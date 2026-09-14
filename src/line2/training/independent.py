"""L3b [S4] 独立版对照：6 个互不共享权重的单投影 FiLM U-Net。

H3 对照臂设计（v0.2.1 复审修订）：E1 比较"联合 vs 独立"时，对照变量只剩
"跨投影信息是否流动"。条件 = 单投影 4 维物理摘要 (σ_i, σ_j, ρ_ij, log₁₀N_low)
（与联合版 κ 同源的物理 edges 矩，非 bin-index 代理——H3 公平性）。

参数量对齐（`30` [S4] C1 ±30%）：spec 公式 ceil(C/√6)=20 与 line1 冻结主干
`GroupNorm(min(8,ch), ch)` 的 8 整除约束冲突（构造即崩，审核 B-1）——偏差登记
于 .decisions.md D17，采用 **C_ind=16、blocks=3**（实测 10,092,870 参数，
对联合版 10,708,710 为 −5.8%，±30% 内）。
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNetBackbone

N_BINS = 128

__all__ = ["SingleProjNet", "IndependentEnsemble", "proj_condition", "PROJ_PAIRS"]

PROJ_PAIRS: tuple[tuple[str, str], ...] = (
    ("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"),
)


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


def proj_condition(
    shape_map: torch.Tensor, centers_i: torch.Tensor, centers_j: torch.Tensor,
    log_n_low: float,
) -> torch.Tensor:
    """单投影 4 维条件：(σ_i, σ_j, ρ_ij, log₁₀N_low)（物理 bin 中心矩）。"""
    p = shape_map.to(shape_map.dtype) / shape_map.sum().clamp_min(1e-12)
    mi = float((p.sum(dim=1) * centers_i).sum())
    mj = float((p.sum(dim=0) * centers_j).sum())
    di = centers_i - mi
    dj = centers_j - mj
    var_i = float((p.sum(dim=1) * di**2).sum())
    var_j = float((p.sum(dim=0) * dj**2).sum())
    rho = float((p * di[:, None] * dj[None, :]).sum())
    return torch.tensor([var_i, var_j, rho, log_n_low], dtype=shape_map.dtype)


class IndependentEnsemble(nn.Module):
    """6 个独立单投影模型（H3 对照臂）。C_ind=16 满足 GroupNorm 8 整除约束。"""

    C_IND = 16
    BLOCKS = 3
    BASE = 24

    def __init__(
        self,
        base: int = 24,
        num_levels: int = 4,
        blocks: int = 3,
        n_proj: int = 6,
    ) -> None:
        super().__init__()
        self.n_proj = int(n_proj)
        self.nets = nn.ModuleList(
            [SingleProjNet(base=base, num_levels=num_levels, blocks=blocks) for _ in range(n_proj)]
        )

    def forward(
        self,
        shape_maps: torch.Tensor,
        edges: torch.Tensor,
        log_n_low: torch.Tensor | float,
    ) -> torch.Tensor:
        """shape_maps (B,6,H,W) 自归一形状图；edges (4,129) 或 (B,4,129)。

        逐投影逐束：物理 bin 中心矩 → 4 维 κ → SingleProjNet → (B,1,H,W)，
        六投影拼回 (B,6,H,W)。确定性。
        """
        b = shape_maps.shape[0]
        if edges.dim() == 2:
            edges = edges.unsqueeze(0).expand(b, -1, -1)
        if not torch.is_tensor(log_n_low):
            log_n_low = torch.full((b,), float(log_n_low), dtype=shape_maps.dtype)
        axis_of = {"x": 0, "xp": 1, "y": 2, "yp": 3}
        per_proj: list[torch.Tensor] = []
        for gi, (ai, aj) in enumerate(PROJ_PAIRS):
            ii, jj = axis_of[ai], axis_of[aj]
            per_beam: list[torch.Tensor] = []
            for bi in range(b):
                centers_i = 0.5 * (edges[bi, ii][:-1] + edges[bi, ii][1:])
                centers_j = 0.5 * (edges[bi, jj][:-1] + edges[bi, jj][1:])
                kappa = proj_condition(
                    shape_maps[bi, gi], centers_i, centers_j, float(log_n_low[bi])
                ).to(shape_maps.dtype)
                out_b = self.nets[gi](shape_maps[bi, gi].reshape(1, 1, N_BINS, N_BINS), kappa.reshape(1, -1))
                per_beam.append(out_b[0, 0])  # (128, 128)
            per_proj.append(torch.stack(per_beam, dim=0))  # (B, 128, 128)
        return torch.stack(per_proj, dim=1)  # (B, 6, 128, 128)
