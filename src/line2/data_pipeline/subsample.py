"""L1 [S2] 统一子采样：无放回、配对（A ⊂ B）、确定性（spec line2/10 [S2]）。

档位 T = {1e3, 3e3, 1e4, 3e4, 1e5}（训练消费 {3e3, 1e4, 1e5}，曲线消费全部），
每档 k = 20 个无放回子样本；2e5 全量表为目标/参照（不抽样）。
"""

from __future__ import annotations

import numpy as np

from .seeds import rng_for

__all__ = ["SUBSAMPLE_TIERS", "SUBSAMPLE_K", "training_tiers", "subsample_index_set"]

SUBSAMPLE_TIERS: tuple[int, ...] = (1_000, 3_000, 10_000, 30_000, 100_000)
SUBSAMPLE_K = 20
_TRAINING_TIERS: tuple[int, ...] = (3_000, 10_000, 100_000)


def training_tiers() -> tuple[int, ...]:
    """训练消费的三档（曲线另消费 {1e3, 3e4}）。"""
    return _TRAINING_TIERS


def subsample_index_set(
    n_total: int, beam_idx: int, station_idx: int, tiers: tuple[int, ...] = SUBSAMPLE_TIERS,
    k: int = SUBSAMPLE_K,
) -> dict[int, list[np.ndarray]]:
    """返回 {N_low: [k 个无放回索引数组]}（spec [S2] C2/C3/C4）。

    每个子样本由独立分支派生 ⇒ 同参数重跑逐位一致（C3）；
    同档内两两索引集不等（C4，n_total ≥ N_low×k 时成立）。
    """
    index_set: dict[int, list[np.ndarray]] = {}
    for tier_idx, n_low in enumerate(tiers):
        if n_low > n_total:
            raise ValueError(f"档位 {n_low} 超过全量粒子数 {n_total}")
        subs: list[np.ndarray] = []
        for sub_idx in range(k):
            rng = rng_for(beam_idx, station_idx, tier_idx, sub_idx)
            idx = rng.choice(n_total, size=n_low, replace=False)
            subs.append(np.sort(idx))
        index_set[n_low] = subs
    return index_set
