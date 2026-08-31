"""评估管线契约测试（05 [S3] L1）：读 EXP-01 评估产物验证契约，不重训。

覆盖规格：80 [S8]（metrics.csv 列名规范、summary.json 字段规范、C4 c_high
列、C5 增益）、60 [S12] C3（评估默认读 best_val.ckpt）、70 [S7.1] C7
（test_id/test_pb 分列不合并）。测试只断言契约/协议，不断言研究结果。
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.integration, pytest.mark.m3]

REQUIRED_METRIC_COLUMNS = [
    "sample_id", "split", "scheme",
    "a_3", "gamma", "b_1",                      # c_high 参数（80 [S8] C4）
    "psnr", "mae", "mse", "ssim",               # 图像级（70 [S3]）
    "e_eps_z", "e_I_peak", "e_sigma_z", "e_sigma_delta", "e_h_eff",  # 物理级（70 [S4]）
    "e_high_doG", "R_E", "e_high_mask", "e_peak",  # 精细结构（70 [S5]）
    "Q_hat",                                     # 输出总强度（50 [S12] C4）
]


def _find_a_run() -> Path:
    # 迁移后：EXP-01 标定评估产物归入研究线 archive 冷存储。
    # 优先取有 metrics.csv 的 run（run2_R2 档），否则取任一匹配。
    candidates = sorted(glob.glob(
        str(PROJECT_ROOT / "archive" / "line1_substitute_sr" / "misc_runs"
            / "EXP-01_A_seed0_run*_D2")
    ))
    r2 = [c for c in candidates if "run2_R2" in c]
    with_csv = [c for c in candidates if (Path(c) / "metrics.csv").exists()]
    return Path((with_csv or r2 or candidates)[0])


@pytest.fixture(scope="module")
def exp01_a() -> Path:
    run = _find_a_run()
    assert run.exists(), "EXP-01 A 评估产物不存在（先跑 evaluate）"
    return run


def test_metrics_csv_columns(exp01_a):
    """metrics.csv 逐样本含全部规定列（80 [S8] 列名规范）。"""
    with open(exp01_a / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "metrics.csv 为空"
    missing = [c for c in REQUIRED_METRIC_COLUMNS if c not in rows[0]]
    assert not missing, f"metrics.csv 缺列：{missing}"
    # c_high 列非空（90 [S8] C4：保证误差 vs c_high 散点可绘制）
    for row in rows:
        for col in ("a_3", "gamma", "b_1"):
            assert row.get(col) not in (None, ""), f"c_high 列 {col} 有空值"


def test_metrics_split_label(exp01_a):
    """split 列存在且统一（test_id/test_pb 分列，70 [S7.1] C7）。"""
    with open(exp01_a / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    splits = {r["split"] for r in rows}
    assert splits, "split 列为空"
    assert all(s in ("test_id", "test_pb", "test_ood", "exp03", "exp04") for s in splits)


def test_summary_json_contract(exp01_a):
    """summary.json 含 version 三元组、metrics（该方案）、baseline（80 [S8]）。"""
    summary = json.loads((exp01_a / "summary.json").read_text(encoding="utf-8"))
    version = summary.get("version", {})
    for key in ("code_version", "data_version", "spec_version"):
        assert version.get(key), f"summary version 缺 {key}"
    assert "metrics" in summary and "A" in summary["metrics"], "metrics.A 缺失"
    bl = summary.get("baseline", {}).get("L_up", {})
    assert "R_E_mean" in bl, "L_up 基线缺失（baseline.L_up.R_E_mean）"


def test_summary_metrics_fields(exp01_a):
    """metrics.A 的数值指标含 mean/std（80 [S8]）。"""
    summary = json.loads((exp01_a / "summary.json").read_text(encoding="utf-8"))
    m = summary["metrics"]["A"]
    for col in ("psnr", "mae", "e_eps_z", "e_I_peak", "R_E"):
        assert col in m and "mean" in m[col], f"metrics.A.{col} 缺 mean"
        assert "std" in m[col], f"metrics.A.{col} 缺 std"


def test_eval_uses_best_val(exp01_a):
    """评估默认读 best_val.ckpt（60 [S12] C3）：产物中 best_val 存在且
    summary 由它产生（metrics.csv 与 checkpoint 同目录）。"""
    assert (exp01_a / "checkpoints" / "best_val.ckpt").exists(), "缺 best_val.ckpt"
    # evaluate CLI 的 checkpoint 缺省路径 = out_dir/checkpoints/best_val.ckpt
    import yaml
    with open(exp01_a / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg.get("checkpoint", {}).get("best_val_path", "").endswith("best_val.ckpt")


def test_id_pb_not_merged_in_summary(exp01_a):
    """单划分评估时 summary 只含该划分（test_id/test_pb 不合并，70 [S7.1] C7）：
    当前 EXP-01 产物为 test_id 单划分——断言 metrics 行的 split 全部一致。"""
    with open(exp01_a / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len({r["split"] for r in rows}) == 1, "单划分产物不应混入多个 split"


def test_perceptual_and_mask_fields_contract(exp01_a):
    """新字段契约（70 [S3] C3 感知指标 + 70 [S7.1] C2 掩膜成分/Π_leak +
    80 [S4] C3b R_E 守卫，2026-08-26 P0 报批包）。

    新格式产物 MUST 含 ssim_vis/cne 感知指标列与 mask_composition/
    re_guard 摘要节；当前 EXP-01 产物为修订前格式（M3 重生成前），无该列
    时跳过——重生成后本用例自动生效（版本化契约）。
    """
    summary_path = exp01_a / "summary.json"
    with open(exp01_a / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    header = set(rows[0]) if rows else set()
    if "ssim_vis" not in header:
        pytest.skip("旧格式产物（修订前）：新字段契约在 M3 重生成后自动生效")
    # 感知指标列（70 [S3] C3 强制报告）+ 掩膜成分/Π_leak 列（70 [S7.1] C2）
    for col in ("ssim_vis", "cne", "ch_in_mask", "b_in_mask", "pi_leak"):
        assert col in header, f"metrics.csv 缺新列 {col}"
    for r in rows:
        for col in ("ssim_vis", "cne"):
            float(r[col])  # 可解析（感知指标无数据集依赖，任何数据格式都应计算）
    # summary：mask_composition（含三字段 median 与判读标志）+ re_guard（80 [S4] C3b）
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mask_comp = summary.get("mask_composition")
    assert mask_comp is not None, "summary 缺 mask_composition"
    for col in ("ch_in_mask", "b_in_mask", "pi_leak"):
        assert col in mask_comp, f"mask_composition 缺 {col}"
        assert "median" in mask_comp[col]
    assert "b_in_exceeds_ch_in_x1.5" in mask_comp
    assert "pi_leak_gt_0.5" in mask_comp
    guard = summary.get("re_guard")
    assert guard is not None, "summary 缺 re_guard（80 [S4] C3b）"
    assert "r_e_max" in guard and "per_scheme" in guard
    for scheme, stat in guard["per_scheme"].items():
        for key in ("median", "max", "passed", "label"):
            assert key in stat, f"re_guard.{scheme} 缺 {key}"
        assert stat["label"] in ("normal", "sharpening_artifact")
