"""训练循环集成测试（60 [S10][S12][S14]）。

覆盖规格：
- 60 [S10] C2/C3/C4（早停：每 2,000 步验证；连续 10 次无改善触发；
  不早于最大步数预算的 50%）；
- 60 [S12] C1（日志字段：train loss / out min/max/sum / checkpoint
  path / config hash / data version / spec version）；
- 60 [S12] C2（checkpoint 随附配置、随机种子、数据版本、训练曲线）；
- 60 [S14] C8（seeds.json：master_seed、scheme seeds、data seeds）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.training.train import (
    DEFAULT_VAL_INTERVAL,
    EarlyStopping,
    TrainingLogger,
    format_log_record,
    train,
)
from src.utils.checkpoint import load_checkpoint, write_seeds_json

pytestmark = [pytest.mark.integration, pytest.mark.m3]


# ---------------------------------------------------------------------------
# 早停（60 [S10]）
# ---------------------------------------------------------------------------
def test_validate_interval():
    """每 2,000 步验证一次（60 [S10] C2）。"""
    es = EarlyStopping(patience=10, min_step_fraction=0.5, max_steps=20_000)
    assert es.should_validate(2_000)
    assert es.should_validate(4_000)
    assert not es.should_validate(1_999)
    assert not es.should_validate(2_001)
    assert not es.should_validate(0)


def test_early_stop_after_10_no_improvement():
    """mock val 序列连续 10 次无改善触发（60 [S10] C3）。"""
    es = EarlyStopping(patience=10, min_step_fraction=0.5, max_steps=20_000)
    # 第 2,000 步改善；此后 9 次（4,000–20,000）无改善 → 尚未触发
    assert es.on_validation(2_000, 1.0) is False  # 首次即最优
    for step in range(4_000, 22_000, 2_000):
        stop = es.on_validation(step, 1.0)
        if step < 20_000:
            assert stop is False
        else:
            assert stop is False  # 第 9 次无改善（20,000）
    assert es.no_improve_count == 9
    # 第 22,000 步为第 10 次连续无改善 → 触发（step ≥ 50% 预算）
    assert es.on_validation(22_000, 1.0) is True


def test_improvement_resets_counter():
    """改善重置无改善计数（60 [S10] C3 语义）。"""
    es = EarlyStopping(patience=3, min_step_fraction=0.0, max_steps=10_000)
    es.on_validation(2_000, 1.0)
    es.on_validation(4_000, 1.0)  # 无改善
    es.on_validation(6_000, 0.9)  # 改善 → 重置
    assert es.no_improve_count == 0
    es.on_validation(8_000, 0.9)
    es.on_validation(10_000, 0.9)
    assert es.no_improve_count == 2


def test_not_early_before_50_percent_budget():
    """早停不得早于最大步数预算的 50%（60 [S10] C4）。"""
    es = EarlyStopping(patience=10, min_step_fraction=0.5, max_steps=20_000)
    # 10 次连续无改善，但验证步全部落在 50%（10,000）之前 → 不触发
    es.on_validation(2_000, 1.0)  # 最优
    for step in range(4_000, 24_000, 2_000):
        stop = es.on_validation(step, 1.0)
        if step < 10_000:
            assert stop is False, f"step={step} 早于 50% 预算不得停止"
        elif step >= 22_000:
            assert stop is True, f"step={step} 已过 50% 预算且无改善 10 次应停止"


def test_best_val_loss_tracking():
    """最优验证损失随验证推进更新（checkpoint 保存依据，60 [S10] C2）。"""
    es = EarlyStopping(patience=2, min_step_fraction=0.0, max_steps=10_000)
    es.on_validation(2_000, 1.0)
    es.on_validation(4_000, 0.8)
    es.on_validation(6_000, 0.6)
    assert es.best_val_loss == 0.6


# ---------------------------------------------------------------------------
# 日志（60 [S12] C1）
# ---------------------------------------------------------------------------
def test_log_record_fields():
    """日志行含规定字段（60 [S12] C1）：train loss / out min/max/sum /
    checkpoint path / config hash / data version / spec version。"""
    record = {
        "step": 100,
        "train_loss": 1.23e-3,
        "out_min": 0.0,
        "out_max": 2.5e-3,
        "out_sum": 0.998,
        "checkpoint_path": "/tmp/x/best_val.ckpt",
        "config_hash": "deadbeef",
        "data_version": "v1",
        "spec_version": "v1.0",
    }
    line = format_log_record(record)
    for token in ("train_loss", "out_min", "out_max", "out_sum",
                  "checkpoint_path", "config_hash", "data_version", "spec_version"):
        assert token in line


def test_training_logger_writes_file(tmp_path):
    """TrainingLogger 追加写 logs/train.log。"""
    logger = TrainingLogger(tmp_path)
    logger.log({"step": 1, "train_loss": 0.5, "data_version": "v1", "spec_version": "v1.0"})
    logger.log({"step": 2, "train_loss": 0.4, "data_version": "v1", "spec_version": "v1.0"})
    lines = (tmp_path / "logs" / "train.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "step=1" in lines[0] and "train_loss=0.5" in lines[0]


# ---------------------------------------------------------------------------
# checkpoint 与 seeds.json（60 [S12] C2、60 [S14] C8）
# ---------------------------------------------------------------------------
def test_checkpoint_meta_roundtrip():
    """checkpoint 随附配置/种子/数据版本/训练曲线（60 [S12] C2）。"""
    from src.training.train import _save_checkpoint

    config = {"master_seed": 42, "scheme": "A", "seed_index": 0,
              "scheme_seeds": {"scheme_A_seed_0": 12345}}
    meta = {"config_hash": "h1", "data_version": "v1", "spec_version": "v1.0"}
    curve = [1.0, 0.9, 0.8]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ck.pt"
        model = _make_tiny_model()
        _save_checkpoint(path, model, config, step=123, val_loss=0.5, curve=curve,
                         c_prior_stats=(np.array([1.0]), np.array([2.0])), meta=meta)
        ck = load_checkpoint(path)
        assert ck["config"] == config
        assert ck["master_seed"] == 42
        assert ck["scheme_seed"] == 12345
        assert ck["data_version"] == "v1"
        assert ck["spec_version"] == "v1.0"
        assert ck["step"] == 123
        assert ck["val_loss"] == 0.5
        assert ck["train_curve"] == curve
        assert ck["config_hash"] == "h1"
        assert ck["model_class"] == "TinyNet"


def test_seeds_json_content():
    """seeds.json 记录 master_seed / scheme seeds / data seeds（60 [S14] C8）。"""
    config = {
        "master_seed": 20260825,
        "scheme": "B",
        "seed_index": 1,
        "scheme_seeds": {"scheme_A_seed_0": 1, "scheme_B_seed_1": 2, "scheme_C_seed_0": 3},
        "data_seeds": {"train": 11, "val": 22, "test_id": 33, "test_pb": 44},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_seeds_json(config, tmp)
        seeds = json.loads(path.read_text(encoding="utf-8"))
        assert seeds["master_seed"] == 20260825
        assert seeds["used_scheme_seed"] == 2  # scheme_B_seed_1
        assert seeds["scheme_seeds"] == config["scheme_seeds"]
        assert seeds["data_seeds"] == config["data_seeds"]


def _make_tiny_model():
    """极小的测试模型（checkpoint 往返用，不执行前向）。"""
    from torch import nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network_config = {"C0": 8}
            self.conv = nn.Conv2d(2, 4, 3)

        def forward(self, x):
            return self.conv(x)

    return TinyNet()


def test_train_function_returns_stats_and_artifacts(tmp_path):
    """短训练（CPU、小步数）：统计字段 + checkpoint + seeds.json 落盘。

    仅验证协议契约（日志/落盘/返回结构），不断言损失与研究结果
    （05 [S1] C1）；真正的数值健康检查在 smoke 层（05 [S4]）。
    """
    config = {
        "experiment_id": "EXP-01",
        "scheme": "A",
        "seed_index": 0,
        "run_tag": "run1",
        "master_seed": 20260825,
        "scheme_seeds": {"scheme_A_seed_0": 12345},
        "data_seeds": {"train": 1},
        "training": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
                     "beta1": 0.9, "beta2": 0.999, "batch_size": 2, "max_steps": 4,
                     "log_interval": 2,
                     "early_stopping": {"patience": 10, "min_step_fraction": 0.5,
                                        "val_interval": 2000},
                     "precision": "fp32", "data_loading": {"num_workers": 0}},
        "network": {"C0": 8, "num_levels": 3, "num_residual_blocks": 1},
        "dataset": {"version": "dev1"},
        "gpu": {"device": "cpu"},
    }
    out = tmp_path / "run"
    stats = train(config, out, train_indices=[0, 1, 2, 3], max_steps_override=4)
    assert stats["scheme"] == "A"
    assert stats["steps_run"] == 4
    assert len(stats["train_loss_curve"]) == 4
    assert all(np.isfinite(v) for v in stats["train_loss_curve"])
    assert (out / "checkpoints" / "best_val.ckpt").exists()
    assert (out / "checkpoints" / "last.ckpt").exists()
    assert (out / "seeds.json").exists()
    assert (out / "logs" / "train.log").exists()
    log_text = (out / "logs" / "train.log").read_text(encoding="utf-8")
    assert "config_hash" in log_text and "data_version" in log_text
