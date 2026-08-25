"""Smoke（L2）共享 fixture：GPU 空闲检查与确定性样本索引（05 [S4]）。

- 启动前检查 cuda:0 空闲（有训练占用则跳过并显式标注，05 [S7] C 类）；
- 样本索引由 TEST_MASTER_SEED 经 SeedSequence 派生（05 [S6] C1，
  禁止测试内裸随机调用）；
- smoke 配置为代理变体 C0=24（50 [S7] 代理变体；smoke 非正式实验，
  测试内显式声明，05 [S4]）。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from tests.conftest import TEST_MASTER_SEED  # noqa: E402

#: smoke 训练样本数（05 [S4]：256 样本、batch 4）。
SMOKE_TRAIN_SAMPLES = 256
#: smoke 评估样本数（05 [S4]：16 样本）。
SMOKE_EVAL_SAMPLES = 16


def _cuda0_free() -> bool:
    """cuda:0 可用且空闲（显存占用 < 50%，05 [S2] L2 纪律）。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return False
    free, total = torch.cuda.mem_get_info(0)
    return free > 0.5 * total


@pytest.fixture(scope="module")
def smoke_device():
    """单卡 cuda:0；启动前检查该卡空闲，否则跳过并显式标注（05 [S2][S7]）。"""
    if not _cuda0_free():
        pytest.skip("cuda:0 不可用或被占用（L2 启动前检查空闲；有训练占用时跳过而非抢占）")
    return "cuda:0"


@pytest.fixture(scope="module")
def smoke_indices():
    """确定性样本索引：256 训练样本（train.h5）+ 16 评估样本（test_id.h5）。"""
    ss = np.random.SeedSequence([TEST_MASTER_SEED, 9001, 9002])  # "smoke"、"indices" 的整数分支
    rng = np.random.default_rng(ss)
    train_idx = rng.choice(20000, SMOKE_TRAIN_SAMPLES, replace=False)
    test_idx = rng.choice(1000, SMOKE_EVAL_SAMPLES, replace=False)
    return train_idx.tolist(), test_idx.tolist()


def smoke_config(
    scheme: str,
    seed_index: int = 0,
    max_steps: int = 100,
    batch_size: int = 4,
) -> dict:
    """Smoke 代理配置：C0=24、batch 4、100 步、单卡 cuda:0（05 [S4]）。"""
    return {
        "experiment_id": "EXP-01",
        "scheme": scheme,
        "seed_index": seed_index,
        "run_tag": "run1",
        "config_tag": "smoke",
        "master_seed": TEST_MASTER_SEED,
        "data_seeds": {"train": 11111, "val": 22222, "test_id": 33333, "test_pb": 44444},
        "scheme_seeds": {f"scheme_{s}_seed_{i}": 1000 * (ord(s) - 64) + i for s in "ABC" for i in range(3)},
        "calibration": {"sigma_K": 11.0, "sigma_n": 1.22e-4},
        "network": {"backbone": "unet", "C0": 24, "num_residual_blocks": 2,
                    "num_levels": 5, "scheme_C_film_injection": "bottleneck_and_decoder",
                    "input_channels": 2},
        "training": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
                     "beta1": 0.9, "beta2": 0.999, "batch_size": batch_size,
                     "max_steps": max_steps, "log_interval": 25,
                     "early_stopping": {"patience": 10, "min_step_fraction": 0.5,
                                        "val_interval": 2000},
                     "lambda_spec": 1.0, "precision": "fp32",
                     "data_loading": {"num_workers": 0, "pin_memory": True}},
        "dataset": {"version": "v1", "train_size": 2000, "val_size": 500,
                    "test_id_size": 250, "test_pb_size": 250},
        "gpu": {"device": "cuda:0", "config_class": "proxy"},
        "evaluation": {"tau": 0.05, "trigger_rate": 0.20,
                       "primary_metric": "ε_high^mask",
                       "secondary_metrics": ["ε_z_relative"],
                       "dog": {"sigma_outer": 0.0, "sigma_inner_factor": 0.5}},
    }
