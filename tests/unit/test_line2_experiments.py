"""L4 E1 门禁评估的测试（spec line2/40 [S4]）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.line2.experiments.gates import (
    evaluate_e1_gates,
    evaluate_h1,
    evaluate_h3,
    winner_variant,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- H1
def test_h1_positive_strong() -> None:
    """5 种子全部 N_eff > 1e4 且中位数 ≥ 2×1e4 → 成立。"""
    values = [22000.0, 25000.0, 18000.0, 30000.0, 27000.0]
    assert evaluate_h1(values) == "成立"


def test_h1_negative_median_low() -> None:
    """中位数 ≤ 1e4 → 不成立。"""
    values = [1500.0, 800.0, 2000.0, 500.0, 1000.0]
    assert evaluate_h1(values) == "不成立"


def test_h1_indeterminate() -> None:
    """5/5 > 1e4 但中位数 < 2e4 → 不可判定。"""
    values = [12000.0, 11000.0, 13000.0, 15000.0, 10000.0]
    assert evaluate_h1(values) == "不可判定"


# ---------------------------------------------------------------- H3
def test_h3_5_5() -> None:
    """5/5 对联合 ≥ 独立 → 成立。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "成立"


def test_h3_4_5_weak() -> None:
    """4/5 对联合 ≥ 独立 → 方向成立。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "方向成立"


def test_h3_3_5_indeterminate() -> None:
    """3/5 → 不可判定。"""
    j = [20000.0, 8000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "不可判定"


def test_h3_2_5_negative() -> None:
    """≤ 2/5 → 不成立。"""
    j = [8000.0, 9000.0, 7000.0, 12000.0, 6000.0]
    i = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    assert evaluate_h3(j, i) == "不成立"


# ---------------------------------------------------------------- 路由推算
def test_winner_variant_full_joint() -> None:
    j = [22000.0, 25000.0, 18000.0, 30000.0, 27000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert winner_variant(j, i) == "joint"


def test_winner_variant_four_5_joint() -> None:
    """4/5 → joint（方向成立）。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert winner_variant(j, i) == "joint"


def test_winner_variant_three_5() -> None:
    """3/5 → 中位数较高者。"""
    j = [20000.0, 20000.0, 12000.0, 15000.0, 11000.0]  # 中位数 15000
    i = [13000.0, 10000.0, 18000.0, 20000.0, 19000.0]  # 中位数 18000
    assert winner_variant(j, i) == "independent"


def test_winner_variant_two_5_independent() -> None:
    j = [8000.0, 9000.0, 7000.0, 12000.0, 6000.0]
    i = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    assert winner_variant(j, i) == "independent"


# ---------------------------------------------------------------- 端到端
def test_evaluate_e1_gates_endtoend() -> None:
    """5 种子配对端到端：H1 + H3 + 路由。"""
    j = [25000.0, 22000.0, 19000.0, 24000.0, 26000.0]
    i = [11000.0, 12000.0, 10000.0, 13000.0, 11000.0]
    result = evaluate_e1_gates(j, i)
    assert result["H1"] == "成立"        # 5/5 > 1e4, 中位 24000 ≥ 2×1e4
    assert result["H3"] == "成立"        # 5/5 对联合 ≥ 独立
    assert result["winner_variant"] == "joint"
