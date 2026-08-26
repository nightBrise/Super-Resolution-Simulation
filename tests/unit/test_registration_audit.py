"""registration_audit.py 单元测试（90 [S5] N8 预注册对账工具）。

覆盖：config 预注册字段核对（FAIL/WARN）、summary.json 判定枚举核对、
99 判据模块批次解析、final_report 骨架标记核对。只断言对账契约，不涉研究结果。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import registration_audit as ra  # noqa: E402

pytestmark = pytest.mark.unit

GIT_HEAD = "f4b2d07cce1bbd803bc60a97cd774f7065510de2"
SPEC_VER = "v1.0+2026-08-26"


def _write_config(path: Path, overrides: dict | None = None) -> Path:
    cfg = {
        "code_version": GIT_HEAD,
        "data_version": "v1",
        "spec_version": SPEC_VER,
        "evaluation": {
            "tau": 0.05,
            "trigger_rate": 0.2,
            "primary_metric": "ε_high^mask",
            "rho_threshold": 0.1,
            "r_e_max": 10.0,
            "ood_degradation_threshold": 0.2,
            "ci_width_min": 0.05,
            "mde": 0.05,
            "max_expansion_factor": 2.0,
        },
        "degradation": {"snr_hf_threshold": 0.1},
        "network": {"work_scale": 65536.0},
        "training": {"lambda_spec": 1.0},
        "calibration": {"sigma_K": 11.0, "sigma_n": 0.000122},
    }
    for dotted, val in (overrides or {}).items():
        cur = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg), encoding="utf-8")  # JSON 是合法 YAML
    return path


def _write_summary(path: Path) -> Path:
    summary = {
        "version": {"code_version": GIT_HEAD, "data_version": "v1", "spec_version": SPEC_VER},
        "metrics": {},
        "prior_gain": {
            "M_A_minus_M_B": {"verdict": "equivalent", "ci95": [-0.001, 0.002]},
            "M_A_minus_M_C": {"verdict": "significant_positive", "ci95": [0.001, 0.003]},
        },
        "three_class": {"verdict": "significant_positive"},
        "one_veto": {"verdict": "no_veto"},
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _write_99(path: Path) -> Path:
    path.write_text(
        "| 2026-08-26 | v1.0 | 30 | Changed | σ_n 标定口径 | R | 30 | Implemented |\n"
        "| 2026-08-26 | v1.0 | 50 | Added | 网络细节 | R | 50 | Implemented |\n"
        "| 2026-08-26 | v1.0 | 90 | Added | 出口决策 | R | 90 | Proposed |\n",
        encoding="utf-8",
    )
    return path


def test_config_all_pre_registered_values_pass(tmp_path):
    cfg = _write_config(tmp_path / "run" / "config.yaml")
    findings = ra.audit_config(cfg)
    assert not [f for f in findings if f["severity"] == "FAIL"]
    assert all(f["severity"] != "WARN" for f in findings)


def test_config_bad_pre_registered_value_fails(tmp_path):
    cfg = _write_config(tmp_path / "run" / "config.yaml", {"training.lambda_spec": 2.0})
    findings = ra.audit_config(cfg)
    fails = [f for f in findings if f["severity"] == "FAIL"]
    assert any("lambda_spec" in f["message"] for f in fails)


def test_config_placeholder_code_version_fails(tmp_path):
    cfg = _write_config(tmp_path / "run" / "config.yaml", {"code_version": "<git rev-parse HEAD 完整 40 位 hash>"})
    findings = ra.audit_config(cfg)
    fails = [f for f in findings if f["severity"] == "FAIL"]
    assert any("code_version" in f["message"] for f in fails)


def test_summary_valid_verdicts_pass(tmp_path):
    summary = _write_summary(tmp_path / "summary.json")
    findings = ra.audit_summary(summary)
    assert not [f for f in findings if f["severity"] == "FAIL"]


def test_summary_invalid_verdict_fails(tmp_path):
    summary = _write_summary(tmp_path / "summary.json")
    summary.write_text(
        summary.read_text().replace('"verdict": "no_veto"', '"verdict": "bogus"'),
        encoding="utf-8",
    )
    findings = ra.audit_summary(summary)
    fails = [f for f in findings if f["severity"] == "FAIL"]
    assert any("one_veto" in f["message"] for f in fails)


def test_scan_99_keeps_criteria_modules(tmp_path):
    rows = ra.scan_99(_write_99(tmp_path / "99_change_log.md"))
    infos = [r for r in rows if r["severity"] == "INFO"]
    modules = {r["message"].split(" | ")[1] for r in infos}
    assert modules == {"30", "90"}  # 50 非判据模块被过滤


def test_run_audit_end_to_end(tmp_path):
    run = tmp_path / "results" / "EXP-02_A_seed0_run1_D2"
    _write_config(run / "config.yaml")
    _write_summary(run / "summary.json")
    findings, table = ra.run_audit(tmp_path / "results", _write_99(tmp_path / "99_change_log.md"), None)
    assert not [f for f in findings if f["severity"] == "FAIL"]
    assert any("INFO" in row for row in table)


def test_main_exit_code_on_fail(tmp_path, capsys):
    _write_config(tmp_path / "EXP-02_A_seed0_run1_D2" / "config.yaml", {"training.lambda_spec": 2.0})
    rc = ra.main(["--results-dir", str(tmp_path), "--99", str(_write_99(tmp_path / "99.md"))])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out
