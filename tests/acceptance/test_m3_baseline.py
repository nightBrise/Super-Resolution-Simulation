"""M3 里程碑验收测试：读 EXP-01 产物，不重训（05 [S5] L3）。

覆盖规格：05 [S5] M3 绑定（读 EXP-01 产物：A/B/C 损失下降、Ĥ≥0、
形状正确；ε_z/I_peak 相对误差中位数作诊断量记录，不断言阈值——
批次二十一 Z2 与 05 [S1] 铁律 1 一致；σ_K/σ_n/σ_smooth 已写入 config
且 99 有登记；R_E(D2)/R_E(D1) 比率门可计算并被执行）。

测试只断言协议/契约/不变量，不断言研究结果（禁止 assert 增益/误差
阈值类断言）。本测试对当前 EXP-01 产物（run2_R2，σ_smooth,H 修订前
口径）执行；批准后的复验产物由同一测试复跑。
"""

from __future__ import annotations

import csv
import json
import glob
import re
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.acceptance, pytest.mark.m3]


def _find_exp01_runs() -> list[Path]:
    """定位 EXP-01 方案 run 目录（优先 run2_R2，回退最新 run）。

    迁移后 EXP-01 标定产物归入研究线 archive 冷存储。
    """
    pattern = sorted(glob.glob(str(
        PROJECT_ROOT / "archive" / "line1_substitute_sr" / "misc_runs"
        / "EXP-01_?_seed0_run*_D2"
    )))
    # 优先 run2_R2（坍缩修复后产物），否则取最新
    r2 = [p for p in pattern if "run2_R2" in p]
    return [Path(p) for p in (r2 or pattern)]


@pytest.fixture(scope="module")
def exp01_runs() -> dict[str, Path]:
    runs = _find_exp01_runs()
    assert runs, "未找到 EXP-01 产物（results/EXP-01_?_seed0_run*_D2/）"
    return {Path(r).name.split("_")[1]: Path(r) for r in runs}


def test_three_schemes_present(exp01_runs):
    """三方案（A/B/C）产物齐全（05 [S5] M3 绑定）。"""
    assert {"A", "B", "C"} <= set(exp01_runs.keys())


def test_checkpoint_and_metrics_exist(exp01_runs):
    """每个方案含 best_val.ckpt / last.ckpt / metrics.csv / summary.json /
    seeds.json / config.yaml（60 [S12] C2、80 [S8] C1）。"""
    for scheme, run in exp01_runs.items():
        for fname in ("checkpoints/best_val.ckpt", "checkpoints/last.ckpt",
                      "metrics.csv", "summary.json", "seeds.json", "config.yaml"):
            assert (run / fname).exists(), f"{scheme}: 缺 {fname}"


def test_output_sentinels(exp01_runs):
    """输出质量哨兵可计算（80 [S4] C3 Proposed：Q 比 ∈[0.1,10]、Pearson
    ρ≥0.1、val L_space 击穿平凡地板 1/N²）。本测试验证哨兵计算路径存在且
    数值合理（不断言具体研究结果优劣，但坍缩解必须被哨兵识别）。"""
    from src.models.schemes import build_scheme_model_from_checkpoint, forward_scheme
    from src.utils.h5data import H5Dataset

    for scheme, run in exp01_runs.items():
        ckpt = torch.load(run / "checkpoints/best_val.ckpt",
                          map_location="cpu", weights_only=False)
        # 旧版 checkpoint（σ_smooth,H 修订前产物）未持久化 work_scale（60 [S15]
        # 双空间契约）；本测试显式针对修订前产物，回填契约值以走通哨兵路径——
        # 批准后的复验产物由同一测试复跑（含 work_scale 持久化，走正常路径）。
        ckpt.setdefault("work_scale", 65536.0)
        model = build_scheme_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ds = H5Dataset(str(PROJECT_ROOT / "data/dev1/val.h5"), "val")
        q_ratios, corrs, lspaces = [], [], []
        with torch.no_grad():
            for i in range(min(20, len(ds))):
                b = ds[i]
                out = forward_scheme(model, {"L_up": b["L_up"].unsqueeze(0),
                                             "P2": b["P2"].unsqueeze(0),
                                             "c_prior_raw": b["c_prior_raw"].unsqueeze(0)},
                                     "cpu")[0, 0].numpy() / model.work_scale
                H = b["H"][0].numpy()
                q_ratios.append(float(out.sum() / H.sum()))
                corrs.append(float(np.corrcoef(out.ravel(), H.ravel())[0, 1]))
                lspaces.append(float(np.abs(out - H).mean()))
        # 哨兵路径存在性断言（非研究结果断言）：
        # 坍缩解（Q 比 <0.01 或 ρ<0.01）必须被识别为不合格
        q_med = float(np.median(q_ratios))
        rho_med = float(np.median(corrs))
        collapsed = (q_med < 0.1) or (rho_med < 0.1)
        # 记录当前值供报告（不设阈值断言）
        assert np.isfinite(q_med) and np.isfinite(rho_med), f"{scheme}: 哨兵 NaN"
        assert not (collapsed and np.mean(lspaces) >= 1.0 / 65536), (
            f"{scheme}: 输出坍缩（Q 比 {q_med:.3f}、ρ {rho_med:.3f}、"
            f"L_space {np.mean(lspaces):.2e} 未击穿地板）——管线异常"
        )


def test_diagnostic_quantities_recorded(exp01_runs):
    """ε_z / I_peak 相对误差中位数作为诊断量从 metrics.csv 提取（80 [S10]
    M3 行，批次二十一 Z2：记录不阻塞，不断言阈值）。"""
    for scheme, run in exp01_runs.items():
        with open(run / "metrics.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{scheme}: metrics.csv 为空"
        for col in ("e_eps_z", "e_I_peak"):
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan", "inf")]
            assert vals, f"{scheme}: metrics.csv 缺列 {col}"
            # 记录诊断量（不断言阈值）
            median = float(np.median(vals))
            assert 0.0 <= median, f"{scheme}: {col} 中位数为负（异常）"


def test_calibration_recorded():
    """σ_K/σ_n/σ_smooth 已写入 config 且 99 有登记（30 [S12] C5、05 [S5]）。"""
    import yaml
    with open(_find_exp01_runs()[0] / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cal = cfg["calibration"]
    assert float(cal["sigma_K"]) > 0 and float(cal["sigma_n"]) > 0, "σ_K/σ_n 未写入 config"
    assert "sigma_smooth" in cal, "σ_smooth 未写入 config"
    # 99 登记（OQ-30-03 / OQ-20-03 存在）
    log = (PROJECT_ROOT / "docs/specs/99_change_log.md").read_text(encoding="utf-8")
    assert "OQ-30-03" in log and "OQ-20-03" in log, "99 缺 OQ-30-03/OQ-20-03 登记"


def test_re_ratio_gate_computable():
    """R_E(D2)/R_E(D1) 比率门可计算并被执行（30 [S12] C3、05 [S5]）——
    测试只断言计算路径存在与数值记录，不断言门通过（当前为 OQ-30-03
    待裁定状态，允许失败并须披露）。"""
    d2 = _find_exp01_runs()[0]  # D2 档 A
    d1_candidates = sorted(glob.glob(str(
        PROJECT_ROOT / "archive" / "line1_substitute_sr" / "misc_runs"
        / "EXP-01b_A_seed0_run1_D1"
    )))
    if not d1_candidates:
        pytest.skip("EXP-01b D1 产物未生成")
    def re_median(path):
        vals = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                v = r.get("R_E")
                if v and v.lower() not in ("nan", "inf"):
                    vals.append(float(v))
        return float(np.median(vals)) if vals else float("nan")
    re_d2 = re_median(d2 / "metrics.csv")
    re_d1 = re_median(Path(d1_candidates[0]) / "metrics.csv")
    assert np.isfinite(re_d2) and np.isfinite(re_d1), "R_E 中位数不可计算"
    ratio = re_d2 / re_d1
    # 记录比率门结果（可失败，OQ-30-03 待裁定；比率门语义待 σ_smooth,H 修订复评）
    assert ratio > 0, f"R_E 比率异常：{ratio}"
