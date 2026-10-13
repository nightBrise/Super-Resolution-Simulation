"""L1 [S4] 投影分箱与归一化（spec line2/10 [S4]）。

- 每站每轴 bin 边界 = 该站全量表 [min, max] 外扩 1%（零截断 ⇒ ∑counts = 粒子数）；
- 六投影按 `00` [S1] 枚举顺序，输出 int32 (6, 128, 128)；
- **图内轴向约定（存储布局）**：`np.histogram2d(table[:, ai], table[:, aj], bins=[ei, ej])`
  给出 `h[bin_ai, bin_aj]`，而落盘取 `h.T`，故 img 的 axis 0（行）= 投影对的**后一坐标**
  bin、axis 1（列）= **前一坐标** bin（如 (x,x′) 图：行 = x′ bin，列 = x bin）。
  按 (ai, aj) 语义消费的调用方（`20` 的 M4 Σ 重构/M5 椭圆/B-moment、`30` 的 Σ 摘要）
  SHALL 自行先转置。
- 坐标 ↔ [-1, 1] 归一化按 bin 边界线性映射，可逆。
- **公共域（`50` [S1]/[S2] C2）**：同站同轴的 129 边界跨全部束共享，由 train 束
  该站全量表的池化 [min, max] 外扩 5% 导出；域外粒子按 [S2] C6 夹取至最近边界格
  （:func:`clamp_table`），计数守恒由 :func:`six_projections` 的零截断守卫复核。
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "PROJECTION_ORDER",
    "AXES",
    "N_BINS",
    "POOLED_PAD_FRAC",
    "axis_edges",
    "station_edges",
    "pooled_axis_edges",
    "pooled_station_edges",
    "clamp_table",
    "count_oob_particles",
    "edges_sha256",
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
#: v2 公共域外扩比例（`50` [S1] 公共域定义：train 池化 + 5% 外扩）
POOLED_PAD_FRAC = 0.05


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


def pooled_axis_edges(
    lo: float, hi: float, n_bins: int = N_BINS, pad_frac: float = POOLED_PAD_FRAC
) -> np.ndarray:
    """公共域单轴 bin 边界：池化 [lo, hi] 外扩 ``pad_frac``（`50` [S1] 公共域规则）。"""
    if not (hi > lo):
        raise ValueError(f"退化公共域（max <= min）：[{lo}, {hi}]")
    pad = (hi - lo) * pad_frac
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def pooled_station_edges(
    bounds: np.ndarray, n_bins: int = N_BINS, pad_frac: float = POOLED_PAD_FRAC
) -> np.ndarray:
    """站级 (4, 129) 公共域边界矩阵；``bounds`` = (4, 2) 的逐轴池化 [min, max]。"""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (len(AXES), 2):
        raise ValueError(f"bounds 形状应为 (4, 2)，实得 {bounds.shape}")
    return np.stack([
        pooled_axis_edges(float(b[0]), float(b[1]), n_bins, pad_frac) for b in bounds
    ])


def clamp_table(table: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 (N, ≥4) 粒子表的横向前 4 列夹取进 ``edges``，返回 (夹取后表, 逐轴夹取计数)。

    越界粒子计入最近边界格（`50` [S2] C6 / D13）；夹取后整表落在 edges 内，故
    :func:`six_projections` 的零截断守卫给出计数守恒（∑counts = 输入行数）复核。
    纵向列（如有）逐位保留。

    第二个返回值为**逐轴诊断计数**（双轴越界者在两轴上各计一次，可作保守上界）；
    `50` [S2] C6 的报告/保险丝口径 = 任一轴越界的**粒子数**，见
    :func:`count_oob_particles`。
    """
    out = table.copy()
    counts = np.zeros(len(AXES), dtype=np.int64)
    for axis in range(len(AXES)):
        row = edges[axis]
        column = table[:, axis]
        counts[axis] = int(np.count_nonzero((column < row[0]) | (column > row[-1])))
        np.clip(column, row[0], row[-1], out=out[:, axis])
    return out, counts


def count_oob_particles(table: np.ndarray, edges: np.ndarray) -> int:
    """任一轴越界的**粒子数**（`50` [S2] C6 的 ``n_clamped`` 口径，验收审查 m-1）。

    与 :func:`clamp_table` 返回的逐轴计数不同：双轴越界的粒子在此只计一次
    （分母 = 粒子总数，占比语义为"被夹取的粒子占比"）。
    """
    oob = np.zeros(table.shape[0], dtype=bool)
    for axis in range(len(AXES)):
        row = edges[axis]
        column = table[:, axis]
        oob |= (column < row[0]) | (column > row[-1])
    return int(np.count_nonzero(oob))


def edges_sha256(edges: np.ndarray) -> str:
    """边界数组的规范化 SHA256（小端 float64 连续字节，与平台字节序无关）。

    适用于任一级：单轴 (129,)、站级 (4, 129)、跨站 (2, 4, 129)。
    """
    canonical = np.ascontiguousarray(np.asarray(edges, dtype=np.float64), dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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

    **零截断守卫（`10` [S4] C1 / 决策 D7）**：分箱后逐投影核对计数和是否等于输入
    行数；出现截断（粒子落在 edges 之外）即抛 ``ValueError``，不做静默丢弃。
    v1 逐束自定标口径下该守卫恒不触发，只防误用（例如传入未按该站定标的
    edges）；v2 公共域口径（`50` [S2]）下域外粒子已被 :func:`clamp_table`
    夹取至边界格，本守卫成为**夹取后计数守恒的承重复核**——计数和若仍不等于
    输入行数，说明夹取/分箱实现有误。
    """
    idx = {a: i for i, a in enumerate(AXES)}
    out = np.zeros((len(PROJECTION_ORDER), N_BINS, N_BINS), dtype=np.int32)
    n_rows = int(table.shape[0])
    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        ei = edges[idx[ai]]
        ej = edges[idx[aj]]
        h, _, _ = np.histogram2d(table[:, idx[ai]], table[:, idx[aj]], bins=[ei, ej])
        counted = int(h.sum())
        if counted != n_rows:
            raise ValueError(
                f"投影 ({ai},{aj}) 检测到截断：输入 {n_rows} 行，分箱计数和 {counted}"
                f"（{n_rows - counted} 个粒子落在 edges 之外）——按 `10` [S4] C1 报错而非静默丢弃"
            )
        out[gi] = np.ascontiguousarray(h.T.astype(np.int32))  # 布局 [bin_aj, bin_ai]
    return out
