"""line2 数据管线的种子派生链（`00` GC2）。

master_seed = 20260913；spawn 链（`00` [S2]/`10` [S2]）：

- 束 i 的抖动参数：``spawn(beam_idx).spawn(JITTER)``
- 束 i 的子采样：``spawn(beam_idx).spawn(station_idx).spawn(tier_idx).spawn(sub_idx)``
- 束级切分：``spawn(SPLIT)``
"""

from __future__ import annotations

import numpy as np

__all__ = ["MASTER_SEED", "JITTER_BRANCH", "SPLIT_BRANCH", "spawn_child"]

MASTER_SEED = 20260913
JITTER_BRANCH = 0
SPLIT_BRANCH = 1


def spawn_child(beam_idx: int, *branches: int) -> np.random.SeedSequence:
    """从 master_seed 派生子 SeedSequence（同参数 ⇒ 同序列，GC2 确定性）。"""
    ss = np.random.SeedSequence(MASTER_SEED).spawn(beam_idx + 1)[beam_idx]
    for b in branches:
        ss = ss.spawn(b + 1)[b]
    return ss


def rng_for(beam_idx: int, *branches: int) -> np.random.Generator:
    return np.random.default_rng(spawn_child(beam_idx, *branches))
