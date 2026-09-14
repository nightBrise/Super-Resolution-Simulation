"""L1 [S4] 投影分箱与归一化（spec line2/10 [S4]）。

- 每站每轴 bin 边界 = 该站全量表 [min, max] 外扩 1%（零截断 ⇒ ∑counts = 粒子数）；
- 六投影按 `00` [S1] 枚举顺序，输出 int32 (6, 128, 128)；
- **图内轴向约定（存储布局）**：`np.histogram2d(table[:, ai], table[:, aj], bins=[ei, ej])`
  给出 `h[bin_ai, bin_aj]`，而落盘取 `h.T`，故 img 的 axis 0（行）= 投影对的**后一坐标**
  bin、axis 1（列）= **前一坐标** bin（如 (x,x′) 图：行 = x′ bin，列 = x bin）。
  按 (ai, aj) 语义消费的调用方（`20` 的 M4 Σ 重构/M5 椭圆/B-moment、`30` 的 Σ 摘要）
  SHALL 自行先转置。
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
    输出为存储布局 [bin_aj, bin_ai]（见模块 docstring 的图内轴向约定）。
    """
    idx = {a: i for i, a in enumerate(AXES)}
    out = np.zeros((len(PROJECTION_ORDER), N_BINS, N_BINS), dtype=np.int32)
    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        ei = edges[idx[ai]]
        ej = edges[idx[aj]]
        h, _, _ = np.histogram2d(table[:, idx[ai]], table[:, idx[aj]], bins=[ei, ej])
        out[gi] = np.ascontiguousarray(h.T.astype(np.int32))  # 布局 [bin_aj, bin_ai]
    return out
