"""Smoke 评估测试（05 [S4] test_smoke_eval，L2 单卡 GPU）。

用随机权重对 16 样本（v1 test_id 子集）跑评估——metrics.csv / summary.json
生成且通过 schema 校验；同时计算 L_up 零学习退化基线（G1(b) 依赖，
70 [S5.3]）。A 与 B 各评估一次以覆盖先验增益 / 一票否决路径（随机权重下
增益方向不定，只验协议不验结论，05 [S1] C1）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from src.evaluation.evaluate import METRIC_COLUMNS, run_evaluate  # noqa: E402
from src.models.schemes import SchemeA, SchemeB  # noqa: E402
from src.utils.checkpoint import save_checkpoint  # noqa: E402
from tests.smoke.conftest import smoke_config  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.gpu, pytest.mark.m3]

#: metrics.csv 必含列（80 [S8] 列名规范子集 + split 列）。
REQUIRED_COLUMNS = {
    "sample_id", "split", "scheme",
    "a_3", "gamma", "b_1",
    "psnr", "mae", "mse", "ssim",
    "e_eps_z", "e_I_peak", "e_sigma_z", "e_sigma_delta", "e_h_eff",
    "e_high_doG", "R_E", "e_high_mask", "e_peak",
    "e_profile_I", "e_profile_S",
    "Q_hat", "R_E_class", "F_i",
}


def _write_random_checkpoint(path: Path, scheme: str) -> None:
    """随机权重 checkpoint（smoke 非正式评估，05 [S4] 允许随机权重）。"""
    model = SchemeA(C0=24) if scheme == "A" else SchemeB(C0=24)
    save_checkpoint(path, {
        "model_class": type(model).__name__,
        "network_config": model.network_config,
        "model_state": model.state_dict(),
    })


def test_smoke_eval_pipeline(smoke_device, smoke_indices, tmp_path):
    """A/B 各 16 样本评估：metrics.csv/summary.json 生成且 schema 校验。"""
    _, test_idx = smoke_indices
    ckpt_dir = tmp_path / "ckpts"
    out = tmp_path / "eval"

    summaries = {}
    for scheme in ("A", "B"):
        _write_random_checkpoint(ckpt_dir / f"{scheme}.ckpt", scheme)
        summaries[scheme] = run_evaluate(
            smoke_config(scheme), "test_id", out,
            ckpt_dir / f"{scheme}.ckpt", indices=test_idx,
        )

    # ---- metrics.csv schema 校验 -------------------------------------------
    assert (out / "metrics.csv").exists()
    with open(out / "metrics.csv", "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * len(test_idx)  # A + B 各 16 行
    assert set(rows[0].keys()) >= REQUIRED_COLUMNS
    # (sample_id, split, scheme) 唯一
    keys = [(r["sample_id"], r["split"], r["scheme"]) for r in rows]
    assert len(keys) == len(set(keys))
    schemes_in_file = {r["scheme"] for r in rows}
    assert schemes_in_file == {"A", "B"}
    # 方案 A 行 F_i 为空；B 行 F_i ∈ {0, 1}（随机权重下任意值均合法）
    a_fi = {r["F_i"] for r in rows if r["scheme"] == "A"}
    b_fi = {int(r["F_i"]) for r in rows if r["scheme"] == "B"}
    assert a_fi == {""}
    assert b_fi <= {0, 1}
    # c_high 参数列非空（80 [S8] C4：误差 vs c_high 散点图可绘制）
    assert all(r["gamma"] != "" for r in rows)
    # 数值列可解析
    for r in rows:
        float(r["psnr"]), float(r["e_high_mask"]), float(r["R_E"])

    # ---- summary.json schema 校验（80 [S8] 字段规范）----------------------
    assert (out / "summary.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["version"]) >= {"code_version", "data_version", "spec_version"}
    assert summary["split"] == "test_id"
    assert set(summary["metrics"]) >= {"A", "B"}
    for scheme in ("A", "B"):
        met = summary["metrics"][scheme]
        assert "psnr" in met and "e_high_mask" in met
        for m in ("mean", "std"):
            assert m in met["psnr"] and m in met["e_high_mask"]
    # 先验增益（A−B）：配对差 + Wilcoxon p + CI + 三分类
    gain = summary["prior_gain"].get("M_A_minus_M_B")
    assert gain is not None
    for metric in ("e_high_mask", "e_eps_z"):
        assert metric in gain
        stat = gain[metric]
        for key in ("mean", "median", "wilcoxon_p", "ci95", "verdict", "holm_p"):
            assert key in stat
        lo, hi = stat["ci95"]
        assert lo <= hi
        assert stat["verdict"] in ("significant_positive", "equivalent", "significant_negative")
    # 一票否决（B vs A）：两层判据字段 + 四分支标签
    veto = summary["one_veto"].get("B")
    assert veto is not None
    for key in ("P_F", "ci_lower_eps_z", "ci_lower_ipeak", "gain_eps_z",
                "gain_ipeak", "verdict", "n_triggered", "overshoot", "smooth"):
        assert key in veto
    assert veto["verdict"] in ("veto", "noise_fluctuation", "partial_failure", "local_failure")
    assert veto["overshoot"] + veto["smooth"] == veto["n_triggered"]
    # L_up 零学习退化基线（G1(b) 依赖，70 [S5.3]）
    baseline = summary.get("baseline", {}).get("L_up", {})
    assert "e_high_doG_mean" in baseline and "R_E_mean" in baseline
    assert np.isfinite(baseline["e_high_doG_mean"])
