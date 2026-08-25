"""统计判定协议单元测试（70 [S4] 一票否决、[S7] bootstrap/Wilcoxon/Holm/三分类）。

覆盖规格：
- 70 [S7.2]（配对差 bootstrap 95% CI，10,000 次、配对差单元重采样；
  固定种子可复现）；Wilcoxon 确为配对符号秩；
- 70 [S7.3]（三分类标签：显著正 / 等效 / 显著负，"等效" ≠ 无增益）；
- 70 [S7.1] C3（Holm 阶梯下降校正，手算一致）；
- 70 [S4] C6（一票否决四分支：物理幻觉失效 / 噪声波动 / 部分失效 /
  局部失效）；
- 预注册常量（τ=0.05、触发率 0.20、主指标 ε_high^mask）从配置读取且与
  config.yaml.template 一致（05 [S3.3] test_statistics）。

测试只断言判定流程正确，不断言任何研究结果（05 [S1] C1）。
"""

from __future__ import annotations

import yaml
import numpy as np
import pytest

from src.evaluation.metrics import (
    N_BOOTSTRAP,
    PRIMARY_METRIC_COL,
    SECONDARY_METRIC_COL,
    TAU,
    TRIGGER_RATE,
    VETO_LOCAL,
    VETO_NOISE,
    VETO_PARTIAL,
    VETO_VETO,
    bootstrap_ci,
    hallucination_flag,
    holm_correction,
    overshoot_smooth_class,
    paired_wilcoxon,
    prior_gain_stats,
    three_class,
    veto_verdict,
)

pytestmark = [pytest.mark.unit, pytest.mark.m3]

PROJECT_ROOT = pytest.importorskip("pathlib").Path(__file__).resolve().parents[2]


def test_bootstrap_reproducible_fixed_seed():
    """bootstrap 固定种子可复现；重采样次数 = 10,000（70 [S7.2]）。"""
    rng = np.random.default_rng(20260825)
    d = rng.normal(0.03, 0.05, size=200)
    lo1, hi1 = bootstrap_ci(d, seed=42)
    lo2, hi2 = bootstrap_ci(d, seed=42)
    assert lo1 == pytest.approx(lo2, rel=0, abs=1e-15)
    assert hi1 == pytest.approx(hi2, rel=0, abs=1e-15)
    lo3, hi3 = bootstrap_ci(d, seed=43)
    assert (lo1, hi1) != (lo3, hi3)


def test_bootstrap_constant_sequence_point_ci():
    """常数序列 d≡0.03 → CI = [0.03, 0.03]（70 [S7.2]）。"""
    d = np.full(100, 0.03)
    lo, hi = bootstrap_ci(d, seed=0)
    assert lo == pytest.approx(0.03, rel=0, abs=1e-12)
    assert hi == pytest.approx(0.03, rel=0, abs=1e-12)


def test_three_class_labels():
    """三分类标签正确（70 [S7.3]）：CI 整体为正/负/含零。"""
    assert three_class((0.01, 0.05)) == "significant_positive"
    assert three_class((-0.05, -0.01)) == "significant_negative"
    assert three_class((-0.01, 0.02)) == "equivalent"
    # 边界：CI 端点恰为零 → 含零 → 等效
    assert three_class((0.0, 0.05)) == "equivalent"
    assert three_class((-0.05, 0.0)) == "equivalent"


def test_wilcoxon_is_paired_signed_rank():
    """配对 Wilcoxon 确为符号秩检验：小样本手算统计量与 scipy 一致。"""
    from scipy.stats import wilcoxon

    d = np.array([0.5, -0.2, 1.0, 0.05, -0.4, 0.3])
    ours = paired_wilcoxon(d)
    assert ours == pytest.approx(float(wilcoxon(d).pvalue), rel=1e-12)
    # 全零差（无证据）→ p = 1.0，不抛异常
    assert paired_wilcoxon(np.zeros(10)) == 1.0


def test_holm_hand_computed():
    """Holm 手算一致（70 [S7.1] C3）：p=[0.01,0.04,0.03] → [0.03,0.06,0.06]。"""
    adj = holm_correction(np.array([0.01, 0.04, 0.03]))
    expected = np.array([0.03, 0.06, 0.06])
    assert np.allclose(adj, expected, rtol=1e-12)
    # 单调不减（校正后 p 值随原始 p 单调）
    assert adj[0] <= adj[2] <= adj[1]


def test_prior_gain_stats_components():
    """增益统计含均值/中位数/Wilcoxon p/CI95/三分类（70 [S7.2]）。"""
    rng = np.random.default_rng(7)
    d = rng.normal(0.05, 0.02, size=100)
    stat = prior_gain_stats(d, seed=1)
    assert stat["mean"] == pytest.approx(d.mean(), rel=1e-12)
    assert stat["median"] == pytest.approx(np.median(d), rel=1e-12)
    assert 0.0 <= stat["wilcoxon_p"] <= 1.0
    lo, hi = stat["ci95"]
    assert lo <= stat["mean"] <= hi
    assert stat["verdict"] in ("significant_positive", "equivalent", "significant_negative")


def test_hallucination_flag_formula():
    """样本级幻觉标志公式（70 [S4]）：PSNR 优于 A 且 ε_z 或 I_peak 恶化 > τ。"""
    assert hallucination_flag(30.0, 28.0, 0.10, 0.04, 0.05, 0.04, tau=0.05) == 1
    assert hallucination_flag(30.0, 28.0, 0.04, 0.10, 0.10, 0.04, tau=0.05) == 1  # I_peak 恶化 > τ
    assert hallucination_flag(27.0, 28.0, 0.10, 0.04, 0.10, 0.04, tau=0.05) == 0  # PSNR 不优
    assert hallucination_flag(30.0, 28.0, 0.08, 0.04, 0.08, 0.04, tau=0.05) == 0  # 恶化 ≤ τ


def test_veto_four_branches():
    """一票否决四分支标签正确（70 [S4] C6 操作化，05 [S3.3] test_statistics）。"""
    trigger = TRIGGER_RATE + 0.01
    # 物理幻觉失效：触发率达标 + CI 下界 > 0（显著）且净增益双非正
    assert veto_verdict(trigger, 0.01, 0.02, -0.01, -0.02) == VETO_VETO
    # 噪声波动：触发率达标但统计不显著（两个 CI 下界均 ≤ 0）
    assert veto_verdict(trigger, -0.01, 0.0, 0.01, 0.02) == VETO_NOISE
    # 部分失效（混合增益）：显著 + 净增益非双负
    assert veto_verdict(trigger, 0.01, -0.02, 0.02, -0.01) == VETO_PARTIAL
    assert veto_verdict(trigger, 0.01, 0.02, 0.01, -0.02) == VETO_PARTIAL
    # 局部失效：触发率不达标
    assert veto_verdict(TRIGGER_RATE, 0.01, 0.02, -0.01, -0.02) == VETO_LOCAL
    assert veto_verdict(TRIGGER_RATE - 0.01, 0.01, 0.02, -0.01, -0.02) == VETO_LOCAL


def test_overshoot_smooth_classification():
    """过冲/平滑分类按 I_peak 带符号相对误差方向（70 [S4] C8）。"""
    assert overshoot_smooth_class(0.1) == "overshoot"   # 高估 → 过冲型
    assert overshoot_smooth_class(-0.1) == "smooth"     # 低估 → 平滑型
    assert overshoot_smooth_class(0.0) == "exact"


def test_pre_registered_constants_match_template():
    """预注册常量从配置读取且与 config.yaml.template 一致（05 [S3.3]）。"""
    template_path = PROJECT_ROOT / "config.yaml.template"
    with open(template_path, "r", encoding="utf-8") as fh:
        template = yaml.safe_load(fh)
    ev = template["evaluation"]
    assert ev["tau"] == 0.05 == TAU
    assert ev["trigger_rate"] == 0.20 == TRIGGER_RATE
    assert ev["primary_metric"] == "ε_high^mask"
    assert PRIMARY_METRIC_COL == "e_high_mask"
    assert ev["secondary_metrics"] == ["ε_z_relative"]
    assert SECONDARY_METRIC_COL == "e_eps_z"
    # 主/次指标列名与模板标注对应（70 [S7.1] C1）
    assert ev["dog"]["sigma_inner_factor"] == 0.5
    # N_BOOTSTRAP 与重采样次数契约（70 [S7.2]）
    assert N_BOOTSTRAP == 10_000
