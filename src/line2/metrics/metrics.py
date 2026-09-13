"""L2 [S1] 度量族 M1–M6 与图组级评估（spec line2/20 [S1]，冻结 v1.0）。

约定（`20` [S1]）：
- **总计数归一**（B2 修复）：被评图组 x 进入 M1/M2/M3/M4/M5 前 SHALL 重标
  x̃ = x·(Σr/Σx)（逐投影；Σx=0 → :class:`MetricUnderflow`）。归一内置于实现，
  调用方无归一入口；M6 为比值型，不需归一。
- 参照 r = 同束同站 2e5 全量直方图。
- 图组：int/float (6, 128, 128)，通道顺序 = `00` [S1] 六投影枚举；
  **图内轴向（实测约定，2026-09-14 复审钉死）**：axis 0 = 投影对**前一坐标** bin，axis 1 = 后一坐标 bin（projections.py 同步；M4/M5/B-moment 均按此实现，无转置）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "MetricUnderflow",
    "BUCKET_EDGES",
    "rel_rmse",
    "bucketed_rel_rmse",
    "poisson_deviance",
    "reconstruct_sigma",
    "sigma_errors",
    "halo_fraction_gap",
    "hf_energy_ratio_gap",
    "evaluate_group",
]

BUCKET_EDGES = (0.5, 2.0, 10.0)
_EPS = 1e-12
_HF_SIGMA_BINS = 1.0  # M6 高通平滑尺度（bin 单位，预注册禁调，D5）
HALO_NSIGMA = 4.0     # M5 椭圆半径（参照 Σ 定义，D3）


class MetricUnderflow(Exception):
    """被评图组总计数为 0（`20` [S3] C4 下溢标记的数据层来源）。"""


def _normalized(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    sx = float(x.sum())
    if sx <= 0:
        raise MetricUnderflow("被评图组总计数为 0")
    return x.astype(np.float64) * (float(r.sum()) / sx)


def rel_rmse(x: np.ndarray, r: np.ndarray) -> float:
    """M1：总计数归一后的相对 RMSE（单张 128×128）。"""
    xt = _normalized(x, r)
    return float(np.sqrt(np.mean((xt - r.astype(np.float64)) ** 2)) / np.mean(r))


def group_rel_rmse(group: np.ndarray, r: np.ndarray) -> float:
    """M1 图组级：六投影各算相对 RMSE 后取平均（spec [S1] M1"六投影各算后取平均"）。"""
    return float(np.mean([rel_rmse(group[gi], r[gi]) for gi in range(group.shape[0])]))


def bucketed_rel_rmse(x: np.ndarray, r: np.ndarray) -> dict:
    """M2：按参照计数分桶的相对 RMSE。

    分母保护（D12）：桶内 mean(r)=0（如 B0 全零参照格）时以 1.0 为分母
    （计数尺度 ~1 的绝对 RMSE），数值守卫非判据变更。
    """
    xt = _normalized(x, r).astype(np.float64)
    rf = r.astype(np.float64)
    out: dict = {}
    bounds = (0.0, *BUCKET_EDGES, np.inf)
    names = ("B0", "B1", "B2", "B3")
    for name, lo, hi in zip(names, bounds[:-1], bounds[1:]):
        mask = (rf >= lo) & (rf < hi)
        n_cells = int(mask.sum())
        if n_cells < 100:
            out[name] = {"n_cells": n_cells, "insufficient": True}
            continue
        denom = max(float(rf[mask].mean()), 1.0)
        out[name] = {
            "n_cells": n_cells,
            "rel_rmse": float(np.sqrt(np.mean((xt[mask] - rf[mask]) ** 2)) / denom),
            "insufficient": False,
        }
    return out


def poisson_deviance(x: np.ndarray, r: np.ndarray) -> float:
    """M3：Poisson 偏差 D = 2Σ[r·ln(r/μ) − (r−μ)]，r=参照（数据）、μ=归一化被评（模型）。

    r=0 的格子该项解析极限为 +μ；micro 全格求和。
    """
    mu = _normalized(x, r).astype(np.float64)
    c = r.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = np.where(c > 0, c * np.log(c / np.maximum(mu, _EPS)), 0.0)
    term = cross - c + mu
    return float(2.0 * term.sum())


def reconstruct_sigma(group: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """由 (6,128,128) 图组重构 4×4 Σ（归一化坐标；首现占位规则，确定性）。"""
    centers = [0.5 * (edges[k][:-1] + edges[k][1:]) for k in range(4)]
    axis_of = {a: i for i, a in enumerate(("x", "xp", "y", "yp"))}
    sigma = np.zeros((4, 4))
    filled = np.zeros((4, 4), dtype=bool)
    for gi, (ai, aj) in enumerate(
        (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))
    ):
        i, j = axis_of[ai], axis_of[aj]
        img = group[gi].astype(np.float64)
        total = img.sum()
        if total <= 0:
            continue
        p = img / total
        ui, uj = centers[i], centers[j]
        mi = float((p.sum(axis=1) * ui).sum())
        mj = float((p.sum(axis=0) * uj).sum())
        s_ii = float((p.sum(axis=1) * (ui - mi) ** 2).sum())
        s_jj = float((p.sum(axis=0) * (uj - mj) ** 2).sum())
        s_ij = float((p * np.outer(ui - mi, uj - mj)).sum())
        if not filled[i, i]:
            sigma[i, i] = s_ii
            filled[i, i] = True
        if not filled[j, j]:
            sigma[j, j] = s_jj
            filled[j, j] = True
        if not (filled[i, j] and filled[j, i]):
            sigma[i, j] = sigma[j, i] = s_ij
            filled[i, j] = filled[j, i] = True
    if not filled.all():
        raise ValueError("Σ 重构不完整（投影枚举未覆盖全部分量）")
    return sigma


def _corr(sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(sigma))
    return sigma / np.outer(d, d)


def sigma_errors(sigma_eval: np.ndarray, sigma_ref: np.ndarray) -> dict:
    """M4：4 个方差分量相对误差 + 6 个相关系数变化 |Δρ| + ε_4D 相对误差（11 维）。"""
    diag_idx = [(0, 0), (1, 1), (2, 2), (3, 3)]
    diag_rel = [
        float(abs(sigma_eval[i, j] - sigma_ref[i, j]) / abs(sigma_ref[i, j]))
        for i, j in diag_idx
    ]
    rho_e, rho_r = _corr(sigma_eval), _corr(sigma_ref)
    off_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d_rho = [float(abs(rho_e[i, j] - rho_r[i, j])) for i, j in off_idx]
    eps_e = float(np.sqrt(np.linalg.det(sigma_eval)))
    eps_r = float(np.sqrt(np.linalg.det(sigma_ref)))
    return {
        "diag_rel": diag_rel,
        "abs_delta_rho": d_rho,
        "eps4d_rel": float(abs(eps_e - eps_r) / abs(eps_r)),
    }


def _halo_fraction_per_projection(
    img: np.ndarray, sigma2d: np.ndarray, mean2d: np.ndarray, centers0: np.ndarray, centers1: np.ndarray
) -> float:
    p = img.astype(np.float64)
    total = p.sum()
    if total <= 0:
        raise MetricUnderflow("投影总计数为 0")
    p /= total
    du0 = centers0 - mean2d[0]
    du1 = centers1 - mean2d[1]
    s11, s12, s22 = sigma2d[0, 0], sigma2d[0, 1], sigma2d[1, 1]
    det = s11 * s22 - s12 * s12
    inv = np.array([[s22, -s12], [-s12, s11]]) / det
    quad = (
        inv[0, 0] * du0[:, None] ** 2
        + 2.0 * inv[0, 1] * np.outer(du0, du1)
        + inv[1, 1] * du1[None, :] ** 2
    )
    outside = quad > HALO_NSIGMA**2
    return float(p[outside].sum())


def halo_fraction_gap(
    group: np.ndarray, group_ref: np.ndarray, sigma_ref: np.ndarray, edges: np.ndarray
) -> float:
    """M5：六投影晕占比差的平均（椭圆由参照 Σ 子块 + 参照投影均值定义，D3/D9）。"""
    pairs = (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))
    axis_of = {a: i for i, a in enumerate(("x", "xp", "y", "yp"))}
    gaps = []
    for gi, (ai, aj) in enumerate(pairs):
        i, j = axis_of[ai], axis_of[aj]
        # 2×2 子块与投影均值（参照定义——D3/D9 防漂移）
        s = np.array([[sigma_ref[i, i], sigma_ref[i, j]], [sigma_ref[j, i], sigma_ref[j, j]]])
        img_r = group_ref[gi].astype(np.float64)
        ui = 0.5 * (edges[i][:-1] + edges[i][1:])
        uj = 0.5 * (edges[j][:-1] + edges[j][1:])
        pr = img_r / img_r.sum()
        mean_i = float((pr.sum(axis=1) * ui).sum())
        mean_j = float((pr.sum(axis=0) * uj).sum())
        gaps.append(
            _halo_fraction_per_projection(group[gi], s, np.array([mean_i, mean_j]), ui, uj)
            - _halo_fraction_per_projection(img_r, s, np.array([mean_i, mean_j]), ui, uj)
        )
    return float(np.mean(np.abs(gaps)))


def hf_energy_ratio_gap(group: np.ndarray, group_ref: np.ndarray) -> float:
    """M6：固定 σ_smooth=1 bin 高通能量比 |Δ| 的六投影平均（比值型，不需归一）。"""
    gaps = []
    for gi in range(group.shape[0]):
        ratios = []
        for img in (group[gi].astype(np.float64), group_ref[gi].astype(np.float64)):
            hp = img - gaussian_filter(img, _HF_SIGMA_BINS)
            ratios.append(float(np.abs(hp).sum() / max(img.sum(), _EPS)))
        gaps.append(abs(ratios[0] - ratios[1]))
    return float(np.mean(gaps))


def evaluate_group(
    group: np.ndarray, group_ref: np.ndarray, edges: np.ndarray
) -> dict:
    """图组级评估入口（`40` 门禁消费）：全部预注册度量 + 下溢标记。

    返回 dict：{"underflow": bool, "M1": float, "M2": dict, "M3": float,
    "M4": dict, "M5": float, "M6": float}；underflow=True 时其余键缺省。
    """
    try:
        if float(group.sum()) <= 0:
            return {"underflow": True}
        sigma_eval = reconstruct_sigma(group, edges)
    except MetricUnderflow:
        return {"underflow": True}
    sigma_ref = reconstruct_sigma(group_ref, edges)
    return {
        "underflow": False,
        "M1": rel_rmse(group, group_ref),
        "M2": bucketed_rel_rmse(group, group_ref),
        "M3": poisson_deviance(group, group_ref),
        "M4": sigma_errors(sigma_eval, sigma_ref),
        "M5": halo_fraction_gap(group, group_ref, sigma_ref, edges),
        "M6": hf_energy_ratio_gap(group, group_ref),
    }
