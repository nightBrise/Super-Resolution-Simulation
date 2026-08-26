"""evaluate/aggregate 的 --split choices 契约测试（EXP-07/08 推理级评估）。

覆盖：--split exp07/exp08 被 argparse 接受（80 [S6] 第一版必做推理级）；
EXP-07/08 只评估方案 B（EXP-07 消融）或 A/B/C（EXP-08）——文件名映射在
evaluate.py 内部（exp07→test_exp07.h5）。只断言 CLI 契约，不跑评估。
"""

from __future__ import annotations

import pytest

from src.evaluation.aggregate import main as aggregate_main
from src.evaluation.evaluate import main as evaluate_main

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("split", ["exp07", "exp08"])
def test_evaluate_accepts_exp07_exp08_splits(split):
    """choices 已接受：config 不存在时干净返回 1，而非 argparse SystemExit(2)。"""
    rc = evaluate_main(["--config", "/nonexistent/config.yaml", "--split", split])
    assert rc == 1


@pytest.mark.parametrize("split", ["exp07", "exp08"])
def test_aggregate_accepts_exp07_exp08_splits(split):
    """choices 已接受：run 目录缺失时按 FileNotFoundError 失败（非 argparse 拒绝）。"""
    with pytest.raises(FileNotFoundError):
        aggregate_main(["--runs", "a,b", "--split", split, "--out", "/tmp/agg_x"])
