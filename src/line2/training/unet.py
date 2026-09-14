"""L3b [S1] Phase 1 网络：FiLM U-Net + Softplus 概率头（spec line2/30 [S1]）。

主干复用 line1 `src/models/unet.py` 的 `UNetBackbone`（残差编码-解码、
GroupNorm+SiLU、FiLM 层），head 替换为 6 通道；输出经 Softplus 后逐投影
归一化（每张和 = 1 的概率图，`30` [S1] C1/C3）。

κ（条件向量，11 维 = 输入图组 10 维 Σ 摘要 + log₁₀N_low）SHALL 仅由输入
图组派生（`30` [S1] C5 防泄露；输入自归一 + 粗矩 FiLM = F8 (a)+(b)）。
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNetBackbone

__all__ = ["Phase1Net", "normalize_input_group"]


def normalize_input_group(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """F8(a) 输入自归一：逐投影除以计数总和。

    返回 (形状图 s, 总计数 t)；s_i = x_i/Σx_i，退化（Σx=0）→ 全零形状图。
    """
    total = counts.sum(dim=(2, 3), keepdim=True)
    s = counts / total.clamp_min(1.0)
    s = torch.where(total > 0, s, torch.zeros_like(s))
    return s, total.squeeze((2, 3))


def condition_vector(
    counts: torch.Tensor, edges: torch.Tensor, log_n_low: torch.Tensor | None = None
) -> torch.Tensor:
    """F8(b) 条件向量 κ ∈ ℝ¹¹：输入图组 10 维 Σ 摘要（对角 4 + 上三角 6）+ log₁₀N_low。

    SHALL 仅由输入派生（`30` [S1] C5 防泄露）。edges: (4, 129) 或 (B, 4, 129)。
    """
    sigma = batch_sigma_from_counts(counts, edges)
    i_idx, j_idx = torch.triu_indices(4, 4)  # 10 个唯一分量（含对角）
    comp = sigma[:, i_idx, j_idx]  # (B, 10)
    if log_n_low is None:
        n_low = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
        log_n_low = torch.log10(n_low.float()).unsqueeze(1)  # (B, 1)
    return torch.cat([comp, log_n_low.to(comp.dtype)], dim=1)


def batch_sigma_from_counts(counts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """由计数图组计算每样本 4×4 Σ（归一化坐标；首现占位，与 `20` 一致）。

    edges：共享 (4, 129) 或逐样本 (B, 4, 129)（混合束/站台 batch；抖动使各束
    bin 边界不同）。逐样本循环（batch ≤ 16，成本可忽略）。
    """
    if edges.dim() == 2:
        edges = edges.unsqueeze(0).expand(counts.shape[0], -1, -1)
    b_dim = counts.shape[0]
    sigma = torch.zeros(b_dim, 4, 4, dtype=counts.dtype, device=counts.device)
    filled = torch.zeros(b_dim, 4, 4, dtype=torch.bool, device=counts.device)
    pairs = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
    for b in range(b_dim):
        centers = 0.5 * (edges[b][:, :-1] + edges[b][:, 1:])  # (4, 128)
        for gi, (ai, aj) in enumerate(pairs):
            p = counts[b, gi].to(counts.dtype)
            total = p.sum()
            if total <= 0:
                continue
            p = p / total
            ci, cj = centers[ai], centers[aj]
            pi = p.sum(dim=1)
            pj = p.sum(dim=0)
            mi = (pi * ci).sum()
            mj = (pj * cj).sum()
            di = ci - mi
            dj = cj - mj
            s_ii = (pi * di**2).sum()
            s_jj = (pj * dj**2).sum()
            s_ij = (p * di[:, None] * dj[None, :]).sum()
            if not filled[b, ai, aj]:
                sigma[b, ai, aj] = s_ij
                sigma[b, aj, ai] = s_ij
                filled[b, ai, aj] = filled[b, aj, ai] = True
            if not filled[b, ai, ai]:
                sigma[b, ai, ai] = s_ii
                filled[b, ai, ai] = True
            if not filled[b, aj, aj]:
                sigma[b, aj, aj] = s_jj
                filled[b, aj, aj] = True
    return sigma


class Phase1Net(nn.Module):
    """Phase 1 联合版网络：6 进 6 出 FiLM U-Net → Softplus 概率头。

    - 输入：形状图 (B, 6, 128, 128) + 条件向量 κ (B, 11)；
    - 输出：概率图 (B, 6, 128, 128)，逐投影 Σp = 1（`30` [S1] C1/C3）；
    - 率语义：rate = p · N_high（数据集常量，`30` [S2]，D7）。
    """

    def __init__(
        self,
        base: int = 48,
        num_levels: int = 4,
        blocks: int = 2,
        cond_dim: int = 11,
        n_proj: int = 6,
        bottleneck_cap: int = 384,
    ) -> None:
        super().__init__()
        if base not in (24, 48, 96):
            raise ValueError(f"base ∈ {{24,48,96}}，得到 {base}")
        if num_levels not in (3, 4, 5):
            raise ValueError(f"num_levels ∈ {{3,4,5}}，得到 {num_levels}")
        self.n_proj = int(n_proj)
        self.backbone = UNetBackbone(
            in_channels=n_proj,
            C0=base,
            num_levels=num_levels,
            num_residual_blocks=blocks,
            bottleneck_cap=bottleneck_cap,
            with_film=True,
            z_dim=cond_dim,
        )
        # head 替换为 n_proj 通道（line1 主干 head 为单通道残差输出）
        self.backbone.head = nn.Conv2d(self.backbone.decoder_levels[-1], n_proj, 1)
        self.softplus = nn.Softplus()

    def forward(self, shape_maps: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(shape_maps, kappa)
        p = self.softplus(raw)
        p = p / p.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        return p

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
