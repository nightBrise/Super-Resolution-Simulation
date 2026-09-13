"""L1 [S4] 投影分箱与归一化（spec line2/10 [S4]）。

- 每站每轴 bin 边界 = 该站全量表 [min, max] 外扩 1%（零截断 ⇒ ∑counts = 粒子数）；
- 六投影按 `00` [S1] 枚举顺序，输出 int32 (6, 128, 128)；
- **图内轴向约定（L2 矩重构必须遵守）**：经 `H.T` 转置后，img 的 **axis 0（行）= 投影对的后一坐标 bin，axis 1（列）= 前一坐标 bin**（如 (x,x′) 图：行 = x′ bin，列 = x bin）——`20` M4 由图组重构 Σ 时须按此转置回 (x, x′) 顺序。
- 坐标 ↔ [-1, 1] 归一化按 bin 边界线性映射，可逆。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PROJECTION_ORDER",
    "AXES",
    "N_BINS",
    "axis_edges",
    "station_edges",
    "normalize_axis",
    "denormalize_axis",
    "six_projections",
]

AXES: tuple[str, ...] = ("x", "xp", "y", "yp")
PROJECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"),
)
N_BINS = 128
_PAD_FRAC = 0.01


def axis_edges(lo: float, hi: float, n_bins: int = N_BINS) -> np.ndarray:
    """单轴 bin 边界：[min, max] 外扩 1%（退化轴报错）。"""
    if not (hi > lo):
        raise ValueError(f"退化轴（max <= min）：[{lo}, {hi}]")
    pad = (hi - lo) * _PAD_FRAC
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def station_edges(table: np.ndarray) -> np.ndarray:
    """站级 (4, 129) 边界矩阵——由该站全量表定标（[S4] C1 前提）。"""
    return np.stack([axis_edges(float(table[:, i].min()), float(table[:, i].max()))
                     for i in range(len(AXES))])


def normalize_axis(coords: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lo, hi = float(edges[0]), float(edges[-1])
    return 2.0 * (coords - lo) / (hi - lo) - 1.0


def denormalize_axis(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lo, hi = float(edges[0]), float(edges[-1])
    return lo + (u + 1.0) / 2.0 * (hi - lo)


def six_projections(table: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(N, 4) 物理坐标表 → (6, 128, 128) int32 计数图组。

    分箱范围（edges）由调用方以该站全量表定标并传入——覆盖一切子样本粒子，
    保证 ∑counts = 子样本粒子数（spec [S4] C1）。
    """
    idx = {a: i for i, a in enumerate(AXES)}
    out = np.zeros((len(PROJECTION_ORDER), N_BINS, N_BINS), dtype=np.int32)
    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        ei = edges[idx[ai]]
        ej = edges[idx[aj]]
        h, _, _ = np.histogram2d(table[:, idx[ai]], table[:, idx[aj]], bins=[ei, ej])
        out[gi] = np.ascontiguousarray(h.T.astype(np.int32))  # img[bin_a, bin_b]
    return out
