"""空域-频域混合损失 ``L_total = L_space + λ·L_spec``（60 [S2]）。

``L_space`` 为对总强度归一化图像的逐像素 L1；``L_spec`` 为按径向频率
五倍频程分带的等权谱 L1，2D FFT 除以 ``N²`` 使频域损失与空域 L1 同量级
（60 [S2] 实现约定）。权重 ``λ`` 冻结为 1.0（预注册常数，60 [S2] C3），
构造器拒绝任何其他取值。本模块不包含物理损失（60 [S2] C2：矩损失、
边缘分布损失、前向一致性损失均不用于第一版训练）。
"""

from __future__ import annotations

import torch
from torch import nn

#: 冻结权重 λ = 1.0，预注册常数，不做代理选择（60 [S2] C3）。
FROZEN_LAMBDA = 1.0

#: 退化截止频率（归一化周期/像素，对应 r=4 下采样，60 [S2]）。
F_C = 1.0 / 8.0

#: 奈奎斯特频率（归一化周期/像素）。
F_NYQUIST = 0.5

#: 五个倍频程频带的边界，单位为归一化周期/像素（60 [S2]）。
BAND_EDGES: tuple[float, ...] = (
    0.0,
    F_C / 4.0,
    F_C / 2.0,
    F_C,
    2.0 * F_C,
    F_NYQUIST,
)

#: 默认图像边长（256×256 固定掩膜，60 [S2] 实现约定）。
DEFAULT_IMAGE_SIZE = 256


def build_band_masks(image_size: int) -> torch.Tensor:
    """生成五倍频程环形频带掩膜，形状 ``(5, image_size, image_size)``。

    径向频率取 ``f = √(k_x² + k_y²)``，``k_x``、``k_y`` 为 ``fftfreq``
    归一化频率（周期/像素）；频带定义为 ``[0, f_c/4]`` 与
    ``(f_{b-1}, f_b]``，边界像素归入低频带；``f > f_N`` 的角点像素不计入
    任何频带（60 [S2] 实现约定）。掩膜生成一次、三方案共享。
    """
    freqs = torch.fft.fftfreq(image_size, dtype=torch.float64)
    kx, ky = torch.meshgrid(freqs, freqs, indexing="ij")
    radius = torch.hypot(kx, ky)

    masks = [radius <= BAND_EDGES[1]]
    for i in range(1, 5):
        masks.append((radius > BAND_EDGES[i]) & (radius <= BAND_EDGES[i + 1]))
    return torch.stack(masks)


class HybridLoss(nn.Module):
    """空域-频域混合损失，三方案统一使用（60 [S2] C1/C3）。

    输入 ``H_hat``（网络预测）与 ``H``（总强度归一化真值）形状须为
    ``(image_size, image_size)`` 或带批量维 ``(B, image_size, image_size)``；
    本损失不对输入做归一化，归一化由数据管线负责（60 [S3]）。
    """

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        lambda_spectral: float = FROZEN_LAMBDA,
    ) -> None:
        super().__init__()
        if lambda_spectral != FROZEN_LAMBDA:
            raise ValueError(
                f"λ 冻结为 {FROZEN_LAMBDA}（预注册常数，60 [S2] C3），"
                f"拒绝修改为 {lambda_spectral}"
            )
        self.image_size = int(image_size)
        self.register_buffer("band_masks", build_band_masks(self.image_size))

    @property
    def lambda_spectral(self) -> float:
        """冻结权重 λ，恒为 1.0。"""
        return FROZEN_LAMBDA

    def l_space(self, H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """空域项：逐像素 L1 均值 ``mean(|Ĥ − H|)``（60 [S2]）。"""
        return torch.mean(torch.abs(H_hat - H))

    def l_spec(self, H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """频域项：五倍频程分带谱 L1，每带取均值、五带等权（60 [S2]）。

        2D FFT 除以 ``N²``（``N`` 为图像边长）后取复数模之差；``f > f_N``
        的角点像素不计入任何频带。
        """
        norm = float(H_hat.shape[-1] * H_hat.shape[-2])
        spec_hat = torch.fft.fft2(H_hat) / norm
        spec_ref = torch.fft.fft2(H) / norm
        diff = torch.abs(spec_hat - spec_ref)

        band_means = []
        for band in self.band_masks:
            # dim=-1 只对频带内像素取均值，保留批量维（逐样本损失）
            band_means.append(diff[..., band].mean(dim=-1))
        return torch.stack(band_means).mean(dim=0)

    def forward(self, H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """总损失 ``L_total = L_space + λ·L_spec``，λ 恒为 1.0。"""
        return self.l_space(H_hat, H) + FROZEN_LAMBDA * self.l_spec(H_hat, H)
