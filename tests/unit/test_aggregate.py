"""aggregate.py 跨方案聚合测试（70 [S7] G2/G3 判定输入）。

覆盖：三个方案的 run 级 metrics.csv 合并 → summary.json 含 prior_gain /
three_class / one_veto（配对行来自跨方案 sample_id 交集）；split 过滤。
只断言契约，不断言研究结果。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.aggregate import aggregate_runs

pytestmark = pytest.mark.unit

GIT_HEAD = "f4b2d07cce1bbd803bc60a97cd774f7065510de2"
SPEC_VER = "v1.0+2026-08-26"
MASTER_SEED = 20260825

VALID_VERDICTS = {"significant_positive", "equivalent", "significant_negative"}

COLUMNS = ["sample_id", "split", "scheme", "e_high_mask", "e_eps_z", "e_I_peak", "F_i", "a_3", "gamma", "b_1"]


def _write_run(run_dir: Path, scheme: str, base: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "code_version": GIT_HEAD,
        "data_version": "v1",
        "spec_version": SPEC_VER,
        "experiment_id": "EXP-02",
        "scheme": scheme,
        "seed_index": 0,
        "master_seed": MASTER_SEED,
        "dataset": {"version": "v1"},
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(cfg, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for i in range(8):
        rows.append(
            {
                "sample_id": f"test-{i:03d}",
                "split": "test_id",
                "scheme": scheme,
                "e_high_mask": f"{base + 0.001 * i:.9f}",
                "e_eps_z": f"{base * 2 + 0.001 * i:.9f}",
                "e_I_peak": f"{base * 3 + 0.001 * i:.9f}",
                "F_i": "0",
                "a_3": "0.1", "gamma": "0.2", "b_1": "0.3",
            }
        )
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def three_runs(tmp_path):
    runs = {}
    for scheme, base in (("A", 0.010), ("B", 0.006), ("C", 0.004)):
        runs[scheme] = tmp_path / f"EXP-02_{scheme}_seed0_run1_D2"
        _write_run(runs[scheme], scheme, base)
    return runs


def test_aggregate_merges_and_builds_prior_gain(three_runs, tmp_path):
    out = tmp_path / "seed0"
    summary = aggregate_runs(list(three_runs.values()), "test_id", out)
    pg = summary["prior_gain"]
    assert "M_A_minus_M_B" in pg and "M_A_minus_M_C" in pg and "M_B_minus_M_C" in pg
    for key in ("M_A_minus_M_B", "M_A_minus_M_C"):
        stat = pg[key]["e_high_mask"]  # 主指标层（70 [S7.2]）
        assert stat["verdict"] in VALID_VERDICTS
        assert len(stat["ci95"]) == 2
    assert summary["three_class"]["M_A_minus_M_B"]["verdict"] in VALID_VERDICTS
    for scheme in ("B", "C"):
        assert summary["one_veto"][scheme]["verdict"] in {
            "no_veto", "noise_fluctuation", "veto", "partial_failure", "local_failure",
        }
    # 合并 metrics.csv：3 方案 × 8 样本
    with open(out / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    assert {r["scheme"] for r in rows} == {"A", "B", "C"}
    assert (out / "summary.json").exists()


def test_aggregate_filters_split(three_runs, tmp_path):
    """run 级 metrics.csv 含非目标 split 行时只保留目标 split。"""
    a = three_runs["A"]
    with open(a / "metrics.csv", "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writerow(
            {"sample_id": "test-999", "split": "test_pb", "scheme": "A",
             "e_high_mask": "0.01", "e_eps_z": "0.02", "e_I_peak": "0.03", "F_i": "0",
             "a_3": "0.1", "gamma": "0.2", "b_1": "0.3"}
        )
    out = tmp_path / "seed0"
    aggregate_runs(list(three_runs.values()), "test_id", out)
    with open(out / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["split"] == "test_id" for r in rows)
    assert len(rows) == 24


def test_aggregate_missing_metrics_raises(three_runs, tmp_path):
    import pytest as _pytest

    (three_runs["A"] / "metrics.csv").unlink()
    with _pytest.raises(FileNotFoundError):
        aggregate_runs(list(three_runs.values()), "test_id", tmp_path / "seed0")
