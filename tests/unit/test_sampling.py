"""参数采样单元测试。

覆盖规格：20 [S9] C1（压缩因子三态联合采样）、C6（参数数值范围）、
C7（W1–W8 有效域拒绝采样）、C8（掩膜版本与通过/拒绝统计）、
C9（W1–W7 通过者中 W8 比例 ≥ 60%）；联合约束 1（三态区间与
α = (C−1)/a₁ 精确导出）与「参数数值范围」小节（逐参数分布、γ 与 β 反号）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generators.masks import apply_masks, MASK_NAMES
from src.generators.sampling import sample_parameters

pytestmark = [pytest.mark.unit, pytest.mark.m1]

#: 掩膜判定所需的参数键（与 apply_masks 的输入契约一致）。
_MASK_KEYS = (
    "sigma_z",
    "eta",
    "b0",
    "a1",
    "alpha",
    "a2",
    "a3",
    "beta",
    "gamma",
    "b1",
    "C",
)


@pytest.fixture(scope="module")
def batch_2000():
    """固定主种子抽取的 2000 组入选参数（σ_K=9px）。"""
    samples, stats = sample_parameters(2000, master_seed=20260825, sigma_K=9.0)
    arrays = {key: np.array([s[key] for s in samples]) for key in _MASK_KEYS}
    return samples, stats, arrays


def test_samples_within_ranges(batch_2000):
    """逐参数取值范围与符号约定符合 20 [S9]「参数数值范围」小节。"""
    _, _, a = batch_2000
    assert np.all((a["sigma_z"] >= 0.30) & (a["sigma_z"] <= 0.70))
    assert np.all((a["C"] >= -0.5) & (a["C"] <= 0.9))
    assert np.all(np.abs(a["eta"]) <= 0.3)
    assert np.all((a["b0"] >= 0.04) & (a["b0"] <= 0.09))
    assert np.all((np.abs(a["a1"]) >= 0.35) & (np.abs(a["a1"]) <= 0.75))
    assert np.all(np.abs(a["a2"]) <= 0.10)
    assert np.all(np.abs(a["a3"]) <= 0.05)
    assert np.all((np.abs(a["beta"]) >= 0.9) & (np.abs(a["beta"]) <= 2.0))
    assert np.all((np.abs(a["gamma"]) >= 0.1) & (np.abs(a["gamma"]) <= 0.6))
    assert np.all((a["b1"] >= -0.10) & (a["b1"] <= 0.20))
    # γ 符号恒与 β 相反（chicane 二阶与三阶项反号）
    assert np.all(np.sign(a["gamma"]) == -np.sign(a["beta"]))


def test_alpha_exact_from_compression_factor(batch_2000):
    """α = (C−1)/a₁ 为精确导出（20 [S9] 联合约束 1）。"""
    _, _, a = batch_2000
    assert np.allclose(a["alpha"] * a["a1"], a["C"] - 1.0, rtol=0, atol=1e-12)


def test_compression_state_fractions(batch_2000):
    """压缩三态占比各为 1/3 ± 5pp（20 [S9] 联合约束 1 预注册比例）。"""
    samples, _, _ = batch_2000
    counts = {state: 0 for state in ("under", "optimal", "over")}
    for s in samples:
        counts[s["compression_state"]] += 1
    for state, count in counts.items():
        assert abs(count / len(samples) - 1.0 / 3.0) <= 0.05, state


def test_all_samples_pass_masks(batch_2000):
    """入选样本全部通过 W1–W8（20 [S9] C7）。"""
    _, _, a = batch_2000
    results = apply_masks(a, sigma_K=9.0)
    for name in MASK_NAMES:
        assert np.all(results[name]), name


def test_stats_recorded(batch_2000):
    """统计记录含掩膜版本、逐掩膜拒绝计数与三态计数（20 [S9] C8）。"""
    _, stats, _ = batch_2000
    assert stats["mask_version"]
    assert set(stats["per_mask_rejected"]) == set(MASK_NAMES)
    assert stats["n_candidates"] >= stats["n_accepted"] == 2000
    assert sum(stats["state_counts"].values()) == 2000


def test_w8_fraction_above_sixty_percent(batch_2000):
    """W1–W7 通过者中满足精细结构窗口的比例 ≥ 60%（20 [S9] C9）。"""
    _, stats, _ = batch_2000
    assert stats["w8_fraction_among_w1_w7_passers"] >= 0.6


def test_sigma_K_fallback_derived_from_w_fine_median():
    """未给定 σ_K 时以 2×w_fine 中位数估计（20 [S9] W8 备注、30 [S12] 口径）。"""
    _, stats = sample_parameters(16, master_seed=7, sigma_K=None)
    assert np.isfinite(stats["sigma_K_px"])
    assert stats["sigma_K_px"] > 0.0


def test_reproducible_with_same_seed():
    """相同主种子下采样结果逐位一致（可复现性）。"""
    s1, _ = sample_parameters(64, master_seed=99, sigma_K=9.0)
    s2, _ = sample_parameters(64, master_seed=99, sigma_K=9.0)
    for a, b in zip(s1, s2):
        for key in _MASK_KEYS:
            assert a[key] == b[key]


def test_candidate_cap_raises():
    """候选上限内无法集满时抛出 RuntimeError（拒绝采样上限保护）。"""
    with pytest.raises(RuntimeError):
        sample_parameters(512, master_seed=3, sigma_K=9.0, max_candidates=256)


def test_invalid_n_raises():
    """非正样本数拒绝。"""
    with pytest.raises(ValueError):
        sample_parameters(0, master_seed=1, sigma_K=9.0)


def _base() -> dict[str, float]:
    """20 [S9] 自洽性示例参数（全部通过 W1–W8，σ_K=9px）。"""
    return {
        "sigma_z": 0.5,
        "eta": 0.1,
        "b0": 0.06,
        "a1": 0.5,
        "alpha": -1.0,
        "a2": 0.05,
        "a3": -0.03,
        "beta": 1.2,
        "gamma": -0.3,
        "b1": 0.05,
        "C": 0.5,
    }


@pytest.mark.parametrize(
    ("violated_mask", "overrides", "sigma_K"),
    [
        # W1：3|η|σ_z = 1.35 ≥ 1
        ("W1", {"eta": 0.9}, 9.0),
        # W2：1 + 9b₁ = −0.08 < 0
        ("W2", {"b1": -0.12}, 9.0),
        # W3：|α| = 6.125 > 3（σ_z=0.7、|a₁|σ_z=0.16、C=−0.4 保持其余掩膜通过）
        ("W3", {"sigma_z": 0.7, "a1": 0.16 / 0.7, "C": -0.4, "alpha": -1.4 / (0.16 / 0.7)}, 9.0),
        # W4：|a₁|σ_z = 0.355 > 0.35（b₀=0.04、a₂=a₃=0、β/γ 缩小保持 W5/W7/W8 通过）
        ("W4", {"a1": 0.71, "b0": 0.04, "a2": 0.0, "a3": 0.0, "beta": 0.8, "gamma": -0.1}, 9.0),
        # W5：左端 ≈ 1.068 > 1
        ("W5", {"a2": 0.15}, 9.0),
        # W6：|C|σ_z = 0.375 > 0.35
        ("W6", {"C": 0.75, "alpha": -0.5}, 9.0),
        # W7：左端 ≈ 0.906 > 0.9（σ_K=10 使 W8 仍通过）
        ("W7", {"beta": 2.0}, 10.0),
        # W8：w_fine ≈ 4.28px > 0.8×5 = 4.0px
        ("W8", {}, 5.0),
    ],
)
def test_single_mask_counterexample_rejected(violated_mask, overrides, sigma_K):
    """8 组各仅违反一条掩膜的反例被逐条拒绝（掩膜判定的独立性）。"""
    c = _base()
    c.update(overrides)
    results = apply_masks(c, sigma_K=sigma_K)
    failing = [name for name in MASK_NAMES if not results[name]]
    assert failing == [violated_mask]
