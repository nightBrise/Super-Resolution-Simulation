"""L3b [S4] 独立版对照：6 个互不共享权重的单投影 FiLM U-Net。

H3 对照臂设计（v0.2.1 复审修订）：E1 比较"联合 vs 独立"时，对照变量只剩
"跨投影信息是否流动"。条件 = 单投影 4 维物理摘要 **(var_i, var_j, cov_ij, log₁₀N_low)**
（与联合版 κ 同源的物理 edges 矩，非 bin-index 代理——H3 公平性）；`30` [S4] 的文字
写 ρ_ij（相关系数、无量纲），实现取**协方差**（第三分量与 log₁₀N_low 不同量级）——
差异已登记于 `30_...decisions.md` D17「两臂 κ 的实际定义」。

Phase 1.5（`50` [S4] C1）把该 4 维摘要的**前三个分量**除以 s_i²、s_j²、s_i·s_j（s_i =
公共域半宽）使其无量纲，log 维口径不变；由 ``kappa_version`` 开关承载，v1 逐位不变。

参数量对齐（`30` [S4] C1 ±30%）：spec 公式 ceil(C/√6)=20 与 line1 冻结主干
`GroupNorm(min(8,ch), ch)` 的 8 整除约束冲突（构造即崩，审核 B-1）——偏差登记
于 .decisions.md D11，采用 **C_ind=16、blocks=3**（实测 10,092,870 参数，
对联合版 10,708,710 为 −5.8%，±30% 内）。
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNetBackbone

from .unet import reference_scales

N_BINS = 128

__all__ = [
    "SingleProjNet",
    "IndependentEnsemble",
    "proj_condition",
    "proj_condition_batch",
    "PROJ_PAIRS",
]

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


def proj_condition_batch(
    shape: torch.Tensor, centers_i: torch.Tensor, centers_j: torch.Tensor,
    log_n_low: torch.Tensor, scale_i: torch.Tensor | None = None,
    scale_j: torch.Tensor | None = None,
) -> torch.Tensor:
    """批量 4 维条件：(var_i, var_j, cov_ij, log₁₀N_low)（物理 bin 中心矩）。

    shape (B, H, W)、centers_i (B, H)、centers_j (B, W)、log_n_low (B,) → (B, 4)。
    第三分量是**协方差** Σ p·d_i·d_j（`30` [S4] 的文字写 ρ_ij 相关系数，差异见 D17），
    故其量级与 log₁₀N_low 相差多个数量级。

    传 ``scale_i``/``scale_j``（参考尺度 s_i、s_j，(B,)）时前三个分量除以
    s_i²、s_j²、s_i·s_j 变为无量纲（`50` [S4] C1 的独立臂口径）；缺省不归一（Phase 1 口径）。

    图组存储布局 = `[后一坐标 a_j, 前一坐标 a_i]`（生产端 `six_projections` 的
    `h.T`，见 `10` [S4]）——此处先转置回 `[a_i, a_j]` 语义，下游 `sum(dim=2)`
    才对应 `centers_i`、`outer(ui, uj)` 才与 `p` 轴序自洽（与
    `metrics.reconstruct_sigma` 同口径，`20` [S1] M4）。
    全程张量运算：单次前向里 96 个逐样本子图（6 投影 × batch）若逐样本取 float()
    会引入数百次设备同步，成本远超计算本身。
    """
    p = shape.transpose(1, 2)
    p = p / shape.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    pi = p.sum(dim=2)
    pj = p.sum(dim=1)
    mi = (pi * centers_i).sum(dim=1)
    mj = (pj * centers_j).sum(dim=1)
    di = centers_i - mi.unsqueeze(1)
    dj = centers_j - mj.unsqueeze(1)
    var_i = (pi * di ** 2).sum(dim=1)
    var_j = (pj * dj ** 2).sum(dim=1)
    cov_ij = (p * di.unsqueeze(2) * dj.unsqueeze(1)).sum(dim=(1, 2))
    if scale_i is not None and scale_j is not None:
        var_i = var_i / scale_i.to(var_i.dtype) ** 2
        var_j = var_j / scale_j.to(var_j.dtype) ** 2
        cov_ij = cov_ij / (scale_i.to(cov_ij.dtype) * scale_j.to(cov_ij.dtype))
    return torch.stack([var_i, var_j, cov_ij, log_n_low.to(shape.dtype)], dim=1)


def proj_condition(
    shape_map: torch.Tensor, centers_i: torch.Tensor, centers_j: torch.Tensor,
    log_n_low: float, scale_i: torch.Tensor | None = None,
    scale_j: torch.Tensor | None = None,
) -> torch.Tensor:
    """单图组 4 维条件（逐样本入口；供等价性核验与外部复用）；``scale_*`` 见上。"""
    log_n = torch.full((1,), float(log_n_low), dtype=shape_map.dtype, device=shape_map.device)
    return proj_condition_batch(
        shape_map.unsqueeze(0),
        centers_i.unsqueeze(0),
        centers_j.unsqueeze(0),
        log_n,
        None if scale_i is None else scale_i.reshape(1),
        None if scale_j is None else scale_j.reshape(1),
    )[0]


class IndependentEnsemble(nn.Module):
    """6 个独立单投影模型（H3 对照臂）。C_ind=16 满足 GroupNorm 8 整除约束。

    ``kappa_version="v2"`` 时逐投影条件按公共域半宽无量纲化（`50` [S4] C1）；
    缺省 ``"v1"`` 保持 Phase 1 的物理单位² 口径逐位不变。
    """

    C_IND = 16
    BLOCKS = 3
    BASE = 24

    def __init__(
        self,
        base: int = 24,
        num_levels: int = 4,
        blocks: int = 3,
        n_proj: int = 6,
        kappa_version: str = "v1",
    ) -> None:
        super().__init__()
        self.n_proj = int(n_proj)
        self.kappa_version = str(kappa_version)
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

        逐投影整批：物理 bin 中心矩 → (B,4) κ → SingleProjNet → (B,1,H,W)，
        六投影拼回 (B,6,H,W)。确定性；逐样本与整批等价（FiLM/GroupNorm/Softplus
        与归一化均逐样本独立，见 tests/unit/test_line2_model.py 的等价性用例）。
        """
        # 统一设备：所有输入与模型同设备（防 DDP/CuBLAS/单卡/多卡混合下的 device mismatch）
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = shape_maps.device
        shape_maps = shape_maps.to(device)
        edges = edges.to(device)
        b = shape_maps.shape[0]
        if not torch.is_tensor(log_n_low):
            log_n = torch.full((b,), float(log_n_low), dtype=shape_maps.dtype, device=device)
        else:
            log_n = log_n_low.to(device).reshape(-1).to(shape_maps.dtype)
            if log_n.numel() == 1:
                log_n = log_n.expand(b)
        if edges.dim() == 2:
            edges = edges.unsqueeze(0).expand(b, -1, -1)
        axis_of = {"x": 0, "xp": 1, "y": 2, "yp": 3}
        centers = 0.5 * (edges[:, :, :-1] + edges[:, :, 1:])  # (B, 4, 128)
        scales = reference_scales(edges) if self.kappa_version == "v2" else None  # (B, 4)
        per_proj: list[torch.Tensor] = []
        for gi, (ai, aj) in enumerate(PROJ_PAIRS):
            ii, jj = axis_of[ai], axis_of[aj]
            kappa = proj_condition_batch(
                shape_maps[:, gi], centers[:, ii], centers[:, jj], log_n,
                None if scales is None else scales[:, ii],
                None if scales is None else scales[:, jj],
            )
            out = self.nets[gi](shape_maps[:, gi].unsqueeze(1), kappa)  # (B, 1, 128, 128)
            per_proj.append(out[:, 0])
        return torch.stack(per_proj, dim=1)  # (B, 6, 128, 128)
