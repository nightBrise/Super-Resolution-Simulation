"""L4 E1 门禁评估（spec line2/40 [S4]）。

- H1（放大存在性）：获胜变体 @1e4 的 5 种子 N_eff 分布
  - 成立：5/5 种子 N_eff > 1e4 且中位数 ≥ 2×1e4
  - 不成立：中位数 ≤ 1e4
  - 不可判定：其它情况
- H3（联合优于独立）：同种子成对比较（5 对）
  - 成立：5/5 对联合 N_eff ≥ 独立
  - 方向成立（弱）：4/5
  - 不成立：≤ 2/5
  - 不可判定：3/5
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "evaluate_h1",
    "evaluate_h3",
    "evaluate_e1_gates",
    "winner_variant",
    "EVAL_FN",
]


def winner_variant(
    joint: Sequence[float], independent: Sequence[float]
) -> str:
    """根据 5 对同种子比较推算 E2 路由的"获胜变体"。

    - 5/5 对联合 ≥ 独立 → joint
    - 4/5 → joint（方向成立）
    - 3/5 → 中位数较高者
    - ≤ 2/5 → independent
    """
    if len(joint) != len(independent):
        raise ValueError("joint 与 independent 种子数不匹配")
    pairs = [(j, i) for j, i in zip(joint, independent) if j is not None and i is not None]
    if not pairs:
        return "indeterminate"
    n = len(pairs)
    joint_wins = sum(1 for j, i in pairs if j >= i)
    if joint_wins >= 5:
        return "joint"
    if joint_wins >= 4:
        return "joint"  # 方向成立
    if joint_wins >= 3:
        return "joint" if np.median([j for j, _ in pairs]) >= np.median([i for _, i in pairs]) else "independent"
    return "independent"


def evaluate_h1(n_eff_values: Sequence[float]) -> str:
    """H1（spec 40 [S4]：放大存在性，获胜变体 @1e4 的 5 种子 N_eff 分布）。

    成立：5/5 N_eff > 1e4 且中位数 ≥ 2e4
    不成立：中位数 ≤ 1e4
    不可判定：其它
    """
    if len(n_eff_values) != 5:
        raise ValueError("H1 需要恰好 5 种子 N_eff")
    n_pos = sum(1 for n in n_eff_values if n > 1e4)
    median = float(np.median(n_eff_values))
    if n_pos == 5 and median >= 2e4:
        return "成立"
    if median <= 1e4:
        return "不成立"
    return "不可判定"


def evaluate_h3(
    joint: Sequence[float], independent: Sequence[float]
) -> str:
    """H3（spec 40 [S4]：联合优于独立，同种子成对 5 对）。

    成立：5/5 对联合 ≥ 独立
    方向成立（弱）：4/5
    不成立：≤ 2/5
    不可判定：3/5
    """
    if len(joint) != 5 or len(independent) != 5:
        raise ValueError("H3 需要各 5 种子 N_eff")
    pairs = [(j, i) for j, i in zip(joint, independent)]
    joint_wins = sum(1 for j, i in pairs if j >= i)
    if joint_wins == 5:
        return "成立"
    if joint_wins == 4:
        return "方向成立"
    if joint_wins <= 2:
        return "不成立"
    return "不可判定"


def evaluate_e1_gates(
    joint: Sequence[float], independent: Sequence[float]
) -> dict:
    """E1 完整门禁：H1（联合） + H3（联合 vs 独立） + 路由推算。"""
    h1 = evaluate_h1(joint)
    h3 = evaluate_h3(joint, independent)
    variant = winner_variant(joint, independent)
    return {"H1": h1, "H3": h3, "winner_variant": variant}


EVAL_FN = evaluate_e1_gates
