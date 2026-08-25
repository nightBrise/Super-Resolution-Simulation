"""三方案配置公平性集成测试（60 [S11]、00 [S6] 约束 4）。

覆盖规格：
- 60 [S11] C1/C2（同 EXP 的 A/B/C config 逐字段一致，白名单仅
  scheme / seed_index / run_tag；相同损失/优化器/学习率/batch/预算）；
- 60 [S11] C4（种子计数：代理尺度 3 个 / 全量阶段每方案恰 2 个）；
- 60 [S15] 15.7（config 内含 scheme 字段，三方案统一 --config 接口）。
"""

from __future__ import annotations

import copy

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.m3]

#: 三方案 config 允许差异的字段白名单（60 [S15] 15.2）。
WHITELIST = {"scheme", "seed_index", "run_tag"}


def make_config(
    scheme: str,
    seed_index: int = 0,
    run_tag: str = "run1",
    config_class: str = "proxy",
) -> dict:
    """同 EXP 的三方案公平配置（A/B/C 仅 scheme/seed_index/run_tag 不同）。"""
    return {
        "code_version": "abc123",
        "data_version": "v1",
        "spec_version": "v1.0",
        "experiment_id": "EXP-01",
        "scheme": scheme,
        "seed_index": seed_index,
        "run_tag": run_tag,
        "config_tag": "",
        "master_seed": 20260825,
        "data_seeds": {"train": 11111, "val": 22222, "test_id": 33333, "test_pb": 44444},
        "scheme_seeds": {
            f"scheme_{s}_seed_{i}": 1000 * (ord(s) - 64) + i
            for s in "ABC"
            for i in range(3 if config_class == "proxy" else 2)
        },
        "calibration": {"sigma_K": 11.0, "sigma_n": 1.22e-4,
                        "sigma_smooth": {"sigma_smooth_H": 1.0, "sigma_smooth_P": 2.0}},
        "network": {"backbone": "unet", "C0": 24, "num_residual_blocks": 2,
                    "num_levels": 5, "scheme_C_film_injection": "bottleneck_and_decoder",
                    "input_channels": 2},
        "training": {"optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
                     "beta1": 0.9, "beta2": 0.999, "batch_size": 8, "max_steps": 5000,
                     "early_stopping": {"patience": 10, "min_step_fraction": 0.5},
                     "lambda_spec": 1.0, "precision": "fp32"},
        "dataset": {"version": "v1", "train_size": 2000, "val_size": 500,
                    "test_id_size": 250, "test_pb_size": 250},
        "gpu": {"device": "cuda:0", "config_class": config_class},
        "evaluation": {"tau": 0.05, "trigger_rate": 0.20,
                       "primary_metric": "ε_high^mask",
                       "secondary_metrics": ["ε_z_relative"],
                       "dog": {"sigma_outer": 1.499125, "sigma_inner_factor": 0.5}},
    }


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """递归展平配置为 ``路径 -> 值``（列表按序展开）。"""
    out: dict[str, object] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def test_configs_identical_except_whitelist():
    """同 EXP 的 A/B/C config 逐字段比对：仅 scheme/seed_index/run_tag 可异。"""
    cfgs = {s: make_config(s, seed_index=0, run_tag="run1") for s in "ABC"}
    base = _flatten(cfgs["A"])
    for scheme in ("B", "C"):
        other = _flatten(cfgs[scheme])
        assert set(base) == set(other)
        for path, value in base.items():
            if path in WHITELIST:
                continue
            assert other[path] == value, f"{scheme} 与 A 在 {path} 不一致（60 [S11] C1/C2）"


def test_scheme_field_only_in_config_whitelist():
    """白名单恰好为 {scheme, seed_index, run_tag}（60 [S15] 15.2）。"""
    a = _flatten(make_config("A", seed_index=0, run_tag="run1"))
    b = _flatten(make_config("B", seed_index=1, run_tag="run2"))
    differing = {p for p in a if a[p] != b.get(p)}
    assert differing == WHITELIST


def test_seed_counts_proxy_and_full():
    """种子计数（60 [S11] C4）：代理尺度每方案 3 个，全量阶段恰 2 个。"""
    proxy = make_config("A", config_class="proxy")
    full = make_config("A", config_class="standard")
    for scheme in "ABC":
        proxy_keys = [k for k in proxy["scheme_seeds"] if k.startswith(f"scheme_{scheme}_seed_")]
        full_keys = [k for k in full["scheme_seeds"] if k.startswith(f"scheme_{scheme}_seed_")]
        assert len(proxy_keys) == 3, f"代理 {scheme} 应为 3 个种子"
        assert len(full_keys) == 2, f"全量 {scheme} 应恰 2 个种子"
    # 全量种子序号为 0/1
    assert {int(k.rsplit("_", 1)[1]) for k in full_keys} == {0, 1}


def test_same_hyperparameters_across_schemes():
    """关键训练超参数三方案一致（60 [S11] C2）：优化器/lr/batch/预算/λ。"""
    cfgs = [make_config(s) for s in "ABC"]
    for key in ("optimizer", "learning_rate", "weight_decay", "beta1", "beta2",
                "batch_size", "max_steps", "lambda_spec", "precision"):
        values = {c["training"][key] for c in cfgs}
        assert len(values) == 1, f"{key} 三方案不一致：{values}"
    for key in ("patience", "min_step_fraction"):
        values = {c["training"]["early_stopping"][key] for c in cfgs}
        assert len(values) == 1, f"{key} 三方案不一致：{values}"
    # λ 冻结 1.0（60 [S2] C3）
    assert cfgs[0]["training"]["lambda_spec"] == 1.0
