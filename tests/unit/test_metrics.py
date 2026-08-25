"""评估指标单元测试（70 [S3]–[S5]）。

覆盖规格：
- 70 [S2] C1（指标前强制总强度归一化：Ĥ'=3Ĥ 与 Ĥ 指标一致）；
- 70 [S3] C1/C2（MAE/MSE/PSNR(MAX=1)/SSIM）；
- 70 [S4]（σ_z/σ_δ/h_eff/ε_z 解析高斯手算、相对误差口径）；
- 70 [S5]（DoG σ_outer 反算、ε_high、ε_peak 掩膜、R_E 四分支分类、
  主指标 ε_high^mask 掩膜构造）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    RE_AS_REPORTED,
    RE_OVER_SMOOTH,
    RE_TEXTURE_HALLUCINATION,
    RE_TRUE_RECOVERY,
    dog_sigma_outer,
    e_high_doG,
    e_high_mask,
    evaluate_sample,
    high_freq_energy_ratio,
    physics_quantities,
    re_joint_class,
    relative_error,
    signed_relative_error,
)
from src.generators.f_beam import pixel_center_coordinates
from src.training.loss import F_C

pytestmark = [pytest.mark.unit, pytest.mark.m3]

N = 256


def _gaussian_density(sigma_z, sigma_delta, mu_z=0.0, mu_delta=0.0, chirp=0.0):
    """归一化二维高斯密度（可选线性 chirp：δ = chirp·z + 独立噪声）。

    解析结果：h_eff = chirp、ε_z = σ_z·σ_δ（与 chirp 无关）。
    """
    coords, _ = pixel_center_coordinates(N)
    Z, D = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(
        -0.5 * ((Z - mu_z) / sigma_z) ** 2
        - 0.5 * ((D - mu_delta - chirp * (Z - mu_z)) / sigma_delta) ** 2
    )
    return kernel / kernel.sum()


@pytest.fixture(scope="module")
def sigma_outer() -> float:
    return dog_sigma_outer(F_C)


@pytest.fixture(scope="module")
def truth_labels(sigma_outer):
    """解析高斯真值及其物理标签（m 由同一图像计算，70 [S4] 公式口径）。"""
    H = _gaussian_density(0.3, 0.2, mu_z=-0.05, mu_delta=0.03, chirp=0.4)
    pq = physics_quantities(H)
    m = {
        "sigma_z": pq["sigma_z"],
        "sigma_delta": pq["sigma_delta"],
        "h_eff": pq["h_eff"],
        "eps_z": pq["eps_z"],
        "I_peak": float(H.sum(axis=1).max()),
        "I_z": H.sum(axis=1),
        "S_delta": H.sum(axis=0),
    }
    return H, m


def test_ideal_metrics_hat_equals_h(truth_labels, sigma_outer):
    """Ĥ == H → 全部指标理想值（05 [S3.3] test_metrics）。"""
    H, m = truth_labels
    met = evaluate_sample(H, H, m, sigma_outer, e_high_baseline=0.0)
    assert met["mae"] == pytest.approx(0.0, abs=1e-12)
    assert met["mse"] == pytest.approx(0.0, abs=1e-12)
    assert met["psnr"] == float("inf")
    assert met["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert met["e_eps_z"] == pytest.approx(0.0, abs=1e-9)
    assert met["e_high_doG"] == pytest.approx(0.0, abs=1e-12)
    assert met["R_E"] == pytest.approx(1.0, rel=1e-6)
    assert met["e_high_mask"] == pytest.approx(0.0, abs=1e-12)
    assert met["e_peak"] == pytest.approx(0.0, abs=1e-12)
    assert met["e_profile_I"] == pytest.approx(0.0, abs=1e-9)
    assert met["e_profile_S"] == pytest.approx(0.0, abs=1e-9)
    assert met["Q_hat"] == pytest.approx(H.sum(), rel=1e-6)


def test_scale_invariance_three_times(truth_labels, sigma_outer):
    """Ĥ'=3Ĥ 与 Ĥ 指标一致（指标前强制归一化，70 [S2] C1）。"""
    H, m = truth_labels
    rng = np.random.default_rng(20260825)
    # 乘性正扰动（保持 Ĥ>0 且非平凡），避免 PSNR=inf 精确比较
    H_hat = H * (1.0 + 0.05 * rng.normal(size=H.shape))
    met1 = evaluate_sample(H_hat, H, m, sigma_outer, e_high_baseline=0.0)
    met2 = evaluate_sample(3.0 * H_hat, H, m, sigma_outer, e_high_baseline=0.0)
    for key in ("mae", "mse", "psnr", "ssim", "e_eps_z", "e_high_doG", "R_E",
                "e_high_mask", "e_peak", "e_profile_I", "e_profile_S"):
        v1, v2 = met1[key], met2[key]
        assert v1 == pytest.approx(v2, rel=1e-6, abs=1e-12), key


def test_gaussian_physics_hand_computed(truth_labels):
    """解析高斯密度 σ_z/σ_δ/h_eff/ε_z 与手算一致（70 [S4]）。

    带线性 chirp h=0.4 的高斯：σ_z=0.3、h_eff=0.4、ε_z=σ_z·σ_δ=0.06
    （发射度与 chirp 无关）；δ 边缘分布 σ_δ^marg = √(σ_δ² + h²σ_z²)。
    """
    H, m = truth_labels
    pq = physics_quantities(H / H.sum())
    assert pq["sigma_z"] == pytest.approx(0.3, rel=1e-2)
    assert pq["sigma_delta"] == pytest.approx(np.sqrt(0.2**2 + 0.4**2 * 0.3**2), rel=1e-2)
    assert pq["h_eff"] == pytest.approx(0.4, rel=1e-2)
    assert pq["eps_z"] == pytest.approx(0.06, rel=1e-2)
    assert pq["I_peak"] == pytest.approx(m["I_peak"], rel=1e-3)
    assert np.allclose(pq["I_z"], m["I_z"], rtol=1e-3, atol=1e-6)


def test_relative_error_definition():
    """相对误差口径：分母取真值幅度，|真值|<floor 时用 floor（70 [S4] C9）。"""
    assert relative_error(0.12, 0.1) == pytest.approx(0.2)
    assert relative_error(0.1, 0.12) == pytest.approx(0.02 / 0.12)
    assert relative_error(0.0, 0.0) == pytest.approx(0.0)
    assert relative_error(1e-5, 1e-8) == pytest.approx(9.99e-6 / 1e-6)
    assert signed_relative_error(0.12, 0.1) == pytest.approx(0.2)
    assert signed_relative_error(0.08, 0.1) == pytest.approx(-0.2)


def test_dog_sigma_outer_inverse(sigma_outer):
    """DoG σ_outer 由 exp(−2π²σ²f_c²)=0.5 反算（70 [S5.1] C6）。"""
    f_c = F_C
    assert sigma_outer == pytest.approx(np.sqrt(np.log(2.0) / (2.0 * np.pi**2 * f_c**2)), rel=1e-12)
    # 反查：该尺度下高斯传递函数确在 f_c 处衰减至 0.5
    assert np.exp(-2.0 * np.pi**2 * sigma_outer**2 * f_c**2) == pytest.approx(0.5, rel=1e-9)


def test_e_high_mask_depends_only_on_truth(sigma_outer):
    """主指标 ε_high^mask 掩膜只依赖真值 H（70 [S7.1] C2），三方案同一掩膜。"""
    from src.evaluation.metrics import c_high_mask_from_hp, high_pass_fft

    H = _gaussian_density(0.3, 0.2, chirp=0.4)
    hp = high_pass_fft(H)
    mask1 = c_high_mask_from_hp(hp, 0.9)
    mask2 = c_high_mask_from_hp(high_pass_fft(H.copy()), 0.9)
    assert np.array_equal(mask1, mask2)
    # 累计能量恰 ≥ 90% 的最小像素集
    energy = np.abs(hp)[mask1].sum()
    assert energy >= 0.9 * np.abs(hp).sum() - 1e-12
    assert mask1.sum() <= np.abs(hp).ravel().size


def test_re_joint_class_four_branches():
    """R_E 联合判读四分支分类（70 [S5.3] 表）。"""
    assert re_joint_class(1.0, 2.0, 1.0) == RE_TRUE_RECOVERY      # ε≤基线 且 R_E∈[0.8,1.2]
    assert re_joint_class(1.0, 2.0, 1.6) == RE_TEXTURE_HALLUCINATION  # ε≤基线 且 R_E>1.5
    assert re_joint_class(1.0, 2.0, 0.4) == RE_OVER_SMOOTH        # R_E<0.5
    assert re_joint_class(3.0, 2.0, 1.0) == RE_AS_REPORTED        # ε>基线（不强行归类）
    # 边界：R_E=0.5 恰为过平滑阈值下界之上 → 不判过度平滑
    assert re_joint_class(3.0, 2.0, 0.5) == RE_AS_REPORTED


def test_r_e_fourier_highpass_bands():
    """R_E 高通频带 = [f_c, f_N]（70 [S5.3] C4，与 60 [S2] 高频带一致）。"""
    from src.evaluation.metrics import high_pass_fft

    H = _gaussian_density(0.3, 0.2, chirp=0.4)
    hp = high_pass_fft(H)
    N = hp.shape[0]
    spectrum = np.fft.fft2(H)
    freqs = np.fft.fftfreq(N)
    kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.hypot(kx, ky)
    kept = (radius > F_C) & (radius <= 0.5)
    assert np.allclose(np.fft.fft2(hp)[~kept], 0.0, atol=1e-10)


def test_peak_mask_definition(sigma_outer):
    """ε_peak 掩膜按 70 [S5.2] C3：8×8 NMS、前 3 峰、≥50% 全局最大。"""
    from src.evaluation.metrics import peak_mask

    H = np.zeros((N, N))
    peaks = [(64, 64, 1.0), (128, 128, 0.9), (192, 192, 0.8), (200, 100, 0.3), (40, 200, 0.2)]
    for (z, d, h) in peaks:
        H[z, d] = h
    H = H / H.sum()
    mask, n_peaks, positions = peak_mask(H)
    assert n_peaks == 3  # 前 3 个峰（高度 ≥ 50% 全局最大的孤立峰）
    assert len(positions) == 3
    # 入选的恰为最高的 3 个峰
    assert (64, 64) in positions and (128, 128) in positions and (192, 192) in positions
    # 高度 < 50% 全局最大的峰不入选（阈值过滤，70 [S5.2]）
    assert (200, 100) not in positions and (40, 200) not in positions
    # 掩膜为 8×8 窗口并集
    assert 0 < mask.sum() <= 3 * 64
    assert mask[64, 64] and mask[192, 192]


def test_evaluate_sample_rejects_nonpositive_total():
    """总强度非正输入不可归一化（70 [S2]）。"""
    from src.evaluation.metrics import evaluate_sample

    m = {"sigma_z": 0.3, "sigma_delta": 0.2, "h_eff": 0.4, "eps_z": 0.06,
         "I_peak": 0.1, "I_z": np.zeros(N), "S_delta": np.zeros(N)}
    with pytest.raises(ValueError):
        evaluate_sample(np.zeros((N, N)), _gaussian_density(0.3, 0.2), m, 1.5)
