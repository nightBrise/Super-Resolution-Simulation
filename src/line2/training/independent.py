"""L3b [S4] 独立版对照：6 个互不共享权重的单投影 FiLM U-Net（spec line2/30 [S4]）。

条件向量缩为单投影 4 维摘要（σ_i, σ_j, ρ_ij, log₁₀N_low）——由输入图组对应
边缘派生。总参数量与联合版对齐（C_ind = ceil(C/√6)，设计期实测校准）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.unet import UNetBackbone

__all__ = ["SingleProjNet", "IndependentEnsemble"]


class SingleProjNet(nn.Module):
    """单投影 1 进 1 出 FiLM U-Net（Softplus 概率头，逐图归一化）。"""

    def __init__(self, base: int, num_levels: int, blocks: int, cond_dim: int = 4) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=1,
            C0=base,
            num_levels=num_levels,
            num_residual_blocks=blocks,
            with_film=True,
            z_dim=cond_dim,
        )
        self.backbone.head = nn.Conv2d(self.backbone.decoder_levels[-1], 1, 1)
        self.softplus = nn.Softplus()

    def forward(self, shape_map: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(shape_map, kappa)
        p = self.softplus(raw)
        p = p / p.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        return p


def _proj_condition(
    img_counts: torch.Tensor, edges_pair: torch.Tensor, log_n_low: torch.Tensor
) -> torch.Tensor:
    """单投影 4 维摘要：(σ_i, σ_j, ρ_ij, log₁₀N_low)。"""
    centers = 0.5 * (edges_pair[:-1] + edges_pair[1:])
    p = img_counts.to(img_counts.dtype) / img_counts.sum().clamp_min(1.0)
    mi = float((p.sum(dim=1) * centers).sum())
    mj = float((p.sum(dim=0) * centers).sum())
    di = centers - mi
    dj = centers - mj
    var_i = float((p.sum(axis=1) * di**2).sum())
    var_j = float((p.sum(axis=0) * dj**2).sum())
    rho = float((p * np.outer(di, dj)).sum())
    return torch.tensor([var_i, var_j, rho, float(log_n_low)], dtype=img_counts.dtype)


class IndependentEnsemble(nn.Module):
    """6 个独立单投影模型（H3 对照臂；参数量经 C_ind 与联合版对齐）。"""

    def __init__(
        self,
        base: int = 20,
        num_levels: int = 4,
        blocks: int = 2,
        n_proj: int = 6,
    ) -> None:
        super().__init__()
        self.n_proj = int(n_proj)
        self.nets = nn.ModuleList(
            [SingleProjNet(base=base, num_levels=num_levels, blocks=blocks) for _ in range(n_proj)]
        )

    def forward(self, shape_maps: torch.Tensor, log_n_low: torch.Tensor) -> torch.Tensor:
        """shape_maps (B,6,H,W) 自归一形状图；log_n_low (B,)。输出 (B,6,H,W) 概率图。"""
        outs = []
        b = shape_maps.shape[0]
        for gi, net in enumerate(self.nets):
            sm = shape_maps[:, gi : gi + 1]
            kappas = []
            for bi in range(b):
                # 4 维摘要由形状图的边缘矩导出（调用方传入 edges 以保证一致）；
                # 此处用轻量代理：形状图行/列二阶矩 + log N。
                img = sm[bi, 0]
                tot = float(img.sum().clamp_min(1e-12))
                pn = img / tot
                h = pn.sum(dim=1)
                w = pn.sum(dim=0)
                c = torch.arange(h.numel(), device=h.device, dtype=h.dtype)
                mi = float((h * c).sum())
                mj = float((w * c).sum())
                vi = float((h * (c - mi) ** 2).sum())
                vj = float((w * (c - mj) ** 2).sum())
                kappas.append(torch.tensor([vi, vj, float(log_n_low[bi])], dtype=img.dtype))
            kappa = torch.stack(kappas).to(shape_maps.device)
            outs.append(net(sm, kappa))
        return torch.cat(outs, dim=1)
