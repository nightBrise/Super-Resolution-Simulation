"""L2 [S4] 容差线产物（H8 的尺子）组装与消费一致性测试（spec line2/20）。

只测 payload 组装与下游读取口径，**不依赖真实数据集**：`deviation_table` 的数值由
算法测试覆盖（`test_line2_metrics.py`），此处覆盖的是"产物格式 ↔ 消费侧"这条接口。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.line2.experiments.gates import OBSERVABLE_NAMES, _h8_from_disk, _load_tolerance
from src.line2.experiments.tolerance_table import (
    K,
    QUANTILE,
    SCOPE,
    TIERS,
    build_payload,
)
from src.line2.metrics.tolerance import N_OBSERVABLES

pytestmark = pytest.mark.unit


def _table(tiers: tuple[int, ...] = TIERS, value: float = 0.5) -> dict[int, np.ndarray]:
    return {int(t): np.full(N_OBSERVABLES, value, dtype=np.float64) for t in tiers}


def _payload(tiers: tuple[int, ...] = TIERS, value: float = 0.5) -> dict:
    return build_payload(
        _table(tiers, value), beam_ids=[1, 2, 3], tiers=tiers, k=K,
        data_dir="studies/line2_particle_sr/data/line2_ds_v1", beams_scope="valtest",
        code_version="a" * 40, data_version="line2_ds_v1", spec_version="line2-spec-v1.0",
        started_at="2026-09-14T04:00:00+00:00", completed_at="2026-09-14T04:30:00+00:00",
        elapsed_s=12.34,
    )


def _fake_run(obs: float, n_eff: float = 30_000.0) -> dict:
    return {
        "run_id": "e1_joint_s0",
        "per_beam": [{"beam_id": 1, "n_eff": n_eff, "n_eff_flag": None,
                      "obs": [obs] * N_OBSERVABLES}],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 组装
def test_build_payload_shape_and_provenance() -> None:
    """`20` [S4]：payload 含 quantile/scope/tiers/k/dev_table，表为 13 维 × 5 档。

    档位键必须是**字符串**——落盘 JSON 后由 `gates._load_tolerance` 按 `int()` 还原。
    """
    payload = _payload()
    assert payload["quantile"] == QUANTILE == 0.95
    assert payload["scope"] == SCOPE == "valtest_pooled"
    assert payload["tiers"] == [1000, 3000, 10000, 30000, 100000]
    assert payload["k"] == K == 20
    assert payload["dev_table"].keys() == {str(t) for t in TIERS}
    assert all(isinstance(key, str) for key in payload["dev_table"])
    for tier, vec in payload["dev_table"].items():
        assert len(vec) == N_OBSERVABLES, tier
        assert all(isinstance(v, float) for v in vec), tier
    assert N_OBSERVABLES == 13 and len(payload["dev_table"]) == 5
    # 溯源字段（`00` GC7 三元组 + 采样口径）齐备且已落成 JSON 原生标量
    assert payload["n_beams"] == 3 and payload["beams_scope"] == "valtest"
    assert payload["code_version"] == "a" * 40 and payload["data_version"] == "line2_ds_v1"
    assert payload["elapsed_s"] == 12.3
    round_trip = json.loads(json.dumps(payload))
    assert round_trip == payload


# ------------------------------------------------------------------ 读回
def test_payload_read_back_by_gate_tolerance_loader(tmp_path: Path) -> None:
    """产物 ↔ 消费一致：`gates._load_tolerance` 把字符串档位还原为整型键的同表。"""
    path = _write(tmp_path / "tolerance.json", _payload(tiers=(1000, 10000, 100000),
                                                        value=0.25))
    table = _load_tolerance(path)
    assert table is not None
    assert sorted(table) == [1000, 10000, 100000]
    assert all(isinstance(tier, int) for tier in table)
    for vec in table.values():
        assert vec.shape == (N_OBSERVABLES,) and float(vec.max()) == 0.25
    # 产物缺失 → None（H8 据此降级，不现场猜容差）
    assert _load_tolerance(tmp_path / "absent.json") is None


# ------------------------------------------------------------------ H8 下游
def test_h8_degrades_when_tolerance_product_is_missing(tmp_path: Path) -> None:
    """无容差线产物 ⇒ H8 判不可判定并记 reason_code（不静默猜一把尺子）。"""
    out = _h8_from_disk({0: _fake_run(obs=0.4)}, (1000, 10000, 100000),
                        tmp_path / "tolerance.json")
    assert out["verdict"] == "不可判定"
    assert out["reason_code"] == "tolerance_missing"
    assert out["pass_rate"] is None and out["per_observable"] == []


def test_h8_consumes_matching_tolerance_product(tmp_path: Path) -> None:
    """同一 payload 供 H8：逐束偏差 0.4 ≤ 容差 0.5 → 全部达标 → 成立。"""
    path = _write(tmp_path / "tolerance.json", _payload(tiers=(1000, 10000, 100000),
                                                        value=0.5))
    out = _h8_from_disk({0: _fake_run(obs=0.4)}, (1000, 10000, 100000), path)
    assert out["verdict"] == "成立" and out["pass_rate"] == 1.0
    assert out["beam_ids"] == [1] and out["n_beams"] == 1
    assert [o["name"] for o in out["per_observable"]] == list(OBSERVABLE_NAMES)
    assert all(o["pass_rate"] == 1.0 and o["n_pass"] == 1 for o in out["per_observable"])


def test_h8_rejects_tolerance_table_missing_curve_tier(tmp_path: Path) -> None:
    """容差表缺曲线档 ⇒ 显式报错（`20` [S4] 最近档映射无法执行），不静默降级。"""
    path = _write(tmp_path / "tolerance.json", _payload(tiers=(1000, 10000), value=0.5))
    with pytest.raises(ValueError, match="缺曲线档"):
        _h8_from_disk({0: _fake_run(obs=0.4)}, (1000, 10000, 100000), path)
