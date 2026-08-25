"""高分辨率生成函数 ``f_beam`` 单元测试。

覆盖规格：20 [S2] C1–C3（输入输出与可复现）、[S3] C1–C4/C7（坐标、非负、
总强度、光滑渲染、默认分辨率）、[S4] C2–C7（剖面形状要素）、
[S5] C1（参数分组）、[S7] C1–C3（物理标签与压缩状态）、
[S10] AC1–AC12（Level 1 验收标准）。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.generators.f_beam as f_beam_mod
from src.generators.f_beam import (
    current_profile,
    f_beam,
    local_energy_spread,
    pixel_center_coordinates,
)

pytestmark = [pytest.mark.unit, pytest.mark.m1]


def _clean_c(**overrides) -> dict[str, float]:
    """无折叠、无 chirp 的简化参数组，用于隔离单一物理要素的诊断测试。"""
    c = {
        "A": 1.0,
        "sigma_z": 0.5,
        "n": 1.5,
        "eta": 0.0,
        "b0": 0.06,
        "a1": 0.0,
        "alpha": 0.0,
        "a2": 0.0,
        "a3": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "b1": 0.0,
    }
    c.update(overrides)
    return c


def test_ac1_reproducible(consistent_c, sigma_smooth_H_px):
    """AC1：相同 c 与配置下输出逐位一致。"""
    H1, m1, _ = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    H2, m2, _ = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    assert np.array_equal(H1, H2)
    for key in ("Q", "mu_z", "mu_delta", "sigma_z", "sigma_delta", "C_zdelta"):
        assert m1[key] == m2[key]


def test_ac2_nonnegative(beam_sample):
    """AC2：H_ij ≥ 0 恒成立。"""
    H, _, _ = beam_sample
    assert H.min() >= 0.0


def test_ac3_total_intensity_and_shape(beam_sample):
    """AC3：默认 256×256，ΣH = Q = 1（20 [S3] C3/C7）。"""
    H, m, _ = beam_sample
    assert H.shape == (256, 256)
    assert abs(H.sum() - 1.0) < 1e-9
    assert m["Q"] == pytest.approx(H.sum(), rel=0, abs=1e-12)
    H5, _, _ = f_beam(_clean_c(), sigma_smooth=2.0, q_total=5.0)
    assert H5.sum() == pytest.approx(5.0, rel=1e-9)


def test_ac4_continuous_in_parameters(consistent_c, sigma_smooth_H_px):
    """AC4：参数连续变化时 H 连续变化（ε 减半，变化量近似减半）。"""
    H0, _, _ = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    for key, eps in (("beta", 0.04), ("a2", 0.02), ("sigma_z", 0.02)):
        c_full = dict(consistent_c, **{key: consistent_c[key] + eps})
        c_half = dict(consistent_c, **{key: consistent_c[key] + eps / 2})
        d_full = np.abs(
            f_beam(c_full, sigma_smooth=sigma_smooth_H_px)[0] - H0
        ).sum()
        d_half = np.abs(
            f_beam(c_half, sigma_smooth=sigma_smooth_H_px)[0] - H0
        ).sum()
        ratio = d_half / d_full
        assert 0.3 <= ratio <= 0.7, (key, ratio)


def test_ac5_flat_top_profile():
    """AC5：n > 1 剖面比高斯更平顶（[S4] C2/C3）。"""
    z_probe = np.array([0.25])
    c_gauss = _clean_c(n=1.0)
    c_flat = _clean_c(n=3.0)
    ratio_gauss = current_profile(z_probe, c_gauss)[0]
    ratio_flat = current_profile(z_probe, c_flat)[0]
    assert ratio_flat > ratio_gauss
    assert ratio_flat > 0.95
    assert ratio_gauss < 0.85
    # n=1 且 η=0 时退化为对称高斯（[S4] C2）
    z = np.linspace(-1.0, 1.0, 257)
    expected = np.exp(-(z / c_gauss["sigma_z"]) ** 2)
    assert np.allclose(current_profile(z, c_gauss), expected, rtol=1e-12)


def test_ac6_head_tail_asymmetry():
    """AC6：η ≠ 0 时头尾不对称；投影一阶矩随 η 符号单调。"""
    z_probe = np.array([0.4, -0.4])
    c_pos = _clean_c(eta=0.2)
    values = current_profile(z_probe, c_pos)
    assert values[0] > values[1]  # η>0 时 z>0 一侧剖面更高

    coords, _ = pixel_center_coordinates(256)
    moment = {}
    for eta in (0.2, -0.2):
        H, _, _ = f_beam(_clean_c(eta=eta), sigma_smooth=2.0)
        moment[eta] = float((coords * H.sum(axis=1)).sum())
    assert moment[0.2] - moment[-0.2] > 0.05


def test_ac7_thickness_varies_with_z():
    """AC7：b₁ ≠ 0 时局部厚度随 |z| 变化；b₁ = 0 时为常数（[S4] C5）。"""
    coords, _ = pixel_center_coordinates(256)

    def tail_over_center_ratio(b1: float) -> float:
        H, _, _ = f_beam(_clean_c(b1=b1), sigma_smooth=2.0)
        zc = coords

        def delta_rms(band: np.ndarray) -> float:
            rows = H[band]
            mass = rows.sum()
            mu = (rows * coords[None, :]).sum() / mass
            var = (rows * (coords[None, :] - mu) ** 2).sum() / mass
            return float(np.sqrt(var))

        center = delta_rms(np.abs(zc) <= 0.1)
        tail = delta_rms((np.abs(zc) >= 0.8) & (np.abs(zc) <= 1.2))
        return tail / center

    assert 0.95 <= tail_over_center_ratio(0.0) <= 1.05
    assert tail_over_center_ratio(0.2) > 1.3
    # 解析式在 b₁=0 时为常数
    z = np.linspace(-1.0, 1.0, 64)
    assert np.allclose(
        local_energy_spread(z, _clean_c(b1=0.0)), _clean_c()["b0"], rtol=1e-12
    )


def test_ac8_third_order_s_shape():
    """AC8：a₃ ≠ 0 时出现三阶 S 形趋势（中心线残差的奇对称分量）。"""

    def cubic_moment(a3: float) -> float:
        H, _, _ = f_beam(_clean_c(a1=0.5, a2=0.05, a3=a3), sigma_smooth=2.0)
        zc, _ = pixel_center_coordinates(256)
        rowmass = H.sum(axis=1)
        keep = rowmass > 1e-4 * rowmass.max()
        mu_delta = (H * _center_coords()[None, :]).sum(axis=1) / np.maximum(
            rowmass, 1e-300
        )
        poly = np.polyfit(zc[keep], mu_delta[keep], 2, w=rowmass[keep])
        resid = mu_delta[keep] - np.polyval(poly, zc[keep])
        weight = rowmass[keep]
        return float((weight * resid * np.sign(zc[keep])).sum() / weight.sum())

    assert abs(cubic_moment(0.0)) < 1e-4
    assert cubic_moment(0.05) > 2e-4
    assert cubic_moment(-0.05) < -2e-4


def _center_coords() -> np.ndarray:
    """δ 方向像素中心坐标（与 `pixel_center_coordinates` 同一定义）。"""
    coords, _ = pixel_center_coordinates(256)
    return coords


def test_ac9_folding_creates_fine_structure():
    """AC9：β ≠ 0 或 γ ≠ 0 时出现折叠或细脊（结构带高频能量显著上升）。"""

    def band_energy(c: dict[str, float]) -> float:
        H, _, _ = f_beam(c, sigma_smooth=1.0)
        F = np.fft.fft2(H, norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
        f = np.hypot(kx, ky)
        band = (f > 1.0 / 16.0) & (f <= 0.5)
        return float(np.sum(np.abs(F[band]) ** 2))

    base = _clean_c(a1=0.5, alpha=-1.0, a2=0.05)
    energy_plain = band_energy(base)
    energy_beta = band_energy(dict(base, beta=1.2))
    energy_beta_gamma = band_energy(dict(base, beta=1.2, gamma=-0.4))
    assert energy_beta > 5.0 * energy_plain
    assert energy_beta_gamma > 3.0 * energy_plain


def test_ac10_fine_structure_not_random(consistent_c, sigma_smooth_H_px):
    """AC10：精细结构非纯随机——两次生成逐位一致且结构由参数决定。"""
    H1, _, _ = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    H2, _, _ = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    assert np.array_equal(H1, H2)
    H3, _, _ = f_beam(
        dict(consistent_c, beta=consistent_c["beta"] + 0.1),
        sigma_smooth=sigma_smooth_H_px,
    )
    assert not np.array_equal(H1, H3)


def test_ac11_physical_labels(beam_sample):
    """AC11：m 含 [S7] 全部标签且矩量与 H 独立重算一致。"""
    H, m, _ = beam_sample
    required = {
        "Q",
        "mu_z",
        "mu_delta",
        "sigma_z",
        "sigma_delta",
        "C_zdelta",
        "h_eff",
        "eps_z",
        "I_peak",
        "I_z",
        "S_delta",
        "Level",
        "c",
        "compression_factor",
        "compression_state",
        "render",
    }
    assert required <= set(m.keys())

    coords, _ = pixel_center_coordinates(256)
    Z, D = np.meshgrid(coords, coords, indexing="ij")
    q = H.sum()
    mu_z = np.sum(H * Z) / q
    mu_d = np.sum(H * D) / q
    var_z = np.sum(H * (Z - mu_z) ** 2) / q
    var_d = np.sum(H * (D - mu_d) ** 2) / q
    cov = np.sum(H * (Z - mu_z) * (D - mu_d)) / q

    assert m["sigma_z"] == pytest.approx(np.sqrt(var_z), rel=1e-12)
    assert m["sigma_delta"] == pytest.approx(np.sqrt(var_d), rel=1e-12)
    assert m["C_zdelta"] == pytest.approx(cov, rel=1e-12)
    assert m["h_eff"] == pytest.approx(cov / var_z, rel=1e-12)
    assert m["eps_z"] == pytest.approx(
        np.sqrt(var_z * var_d - cov**2), rel=1e-12
    )
    assert np.allclose(m["I_z"], H.sum(axis=1), rtol=0, atol=1e-15)
    assert np.allclose(m["S_delta"], H.sum(axis=0), rtol=0, atol=1e-15)
    assert m["I_peak"] == pytest.approx(m["I_z"].max(), rel=0, abs=1e-15)
    assert m["Level"] == 1

    groups = m["c"]
    assert set(groups["c_low"]) == {"A", "sigma_z", "n", "eta", "b0", "a1", "alpha"}
    assert set(groups["c_mid"]) == {"a2", "beta"}
    assert set(groups["c_high"]) == {"a3", "gamma", "b1"}

    assert m["compression_state"] == "under"  # 自洽性示例 C=0.5
    assert m["render"]["normalization"]


def test_compression_state_labels():
    """压缩状态标签由 C = 1 + a₁α 派生（20 [S7] C3）。"""
    c_under = _clean_c(a1=0.5, alpha=-1.0)  # C = 0.5
    c_optimal = _clean_c(a1=0.5, alpha=-2.0)  # C = 0.0
    c_over = _clean_c(a1=0.5, alpha=-3.0)  # C = -0.5
    _, m_u, _ = f_beam(c_under, sigma_smooth=2.0)
    _, m_o, _ = f_beam(c_optimal, sigma_smooth=2.0)
    _, m_v, _ = f_beam(c_over, sigma_smooth=2.0)
    assert (m_u["compression_state"], m_u["compression_factor"]) == ("under", 0.5)
    assert (m_o["compression_state"], m_o["compression_factor"]) == ("optimal", 0.0)
    assert (m_v["compression_state"], m_v["compression_factor"]) == ("over", -0.5)


def test_ac12_no_degradation_logic():
    """AC12：f_beam 不含低分辨率退化逻辑（无下采样、无噪声、不导入 30 模块）。"""
    source = inspect.getsource(f_beam_mod)
    assert "f_deg" not in source
    assert "np.random" not in source
    assert "default_rng" not in source
    assert not hasattr(f_beam_mod, "f_deg")
