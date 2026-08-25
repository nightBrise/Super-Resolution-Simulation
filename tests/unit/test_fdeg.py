"""低分辨率退化函数 ``f_deg`` 单元测试。

覆盖规格：30 [S1] C1–C3、[S2] C1–C3（Blur→Downsample→Noise 固定顺序、
禁止抽稀）、[S3] C1–C4（四元组、L_clean、非负、可复现）、[S4] C1–C4
（退化参数与噪声模型）、[S5] C1–C3（物理坐标一致、块求和保总强度）、
[S6] C8（SNR_hf 定义）、[S9] C1–C3（元数据与定死 SNR）、[S11] D1–D12、
[S12] C1（默认配置）与 [S7]（退化等级 σ_K 标定规则）。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import src.generators.f_deg as f_deg_mod
from src.generators.f_deg import f_deg, sigma_K_for_level, snr_hf

pytestmark = [pytest.mark.unit, pytest.mark.m1]


@pytest.fixture(scope="module")
def dot_image() -> np.ndarray:
    """位于图像中心的单像素亮点，用于点扩散与定位诊断。"""
    H = np.zeros((256, 256))
    H[128, 128] = 1.0
    return H


def test_d7_output_size(beam_sample):
    """D7：下采样后尺寸为 N_H / r（64×64，r=4，30 [S4] C2）。"""
    H, _, _ = beam_sample
    L, L_clean, d, m_L = f_deg(H, sigma_K=4.0, sigma_n=1e-4, seed=1)
    assert L.shape == (64, 64)
    assert L_clean.shape == (64, 64)
    assert m_L["N_H"] == 256
    assert m_L["N_L"] == 64
    L_small, _, _, m_small = f_deg(np.ones((64, 64)), sigma_K=1.0, sigma_n=0.0, r=4)
    assert L_small.shape == (16, 16)
    assert m_small["N_L"] == 16


def test_d4_nonnegative_and_truncation(beam_sample, rng):
    """D4：L ≥ 0；截断为噪声模型组成部分（30 [S4] C4）。"""
    H, _, _ = beam_sample
    sigma_n = 5e-3  # 远大于像素量级，强制截断发生
    L, L_clean, _, m_L = f_deg(H, sigma_K=4.0, sigma_n=sigma_n, seed=3)
    assert L.min() >= 0.0
    assert np.any(L == 0.0)  # 截断确实发生
    assert "max(0" in m_L["truncation"]


def test_l_equals_clipped_noise_sum(beam_sample):
    """噪声实现为 L = max(0, L_clean + n)，n 由种子派生（30 [S4] C4）。"""
    H, _, _ = beam_sample
    seed = 11
    L, L_clean, _, _ = f_deg(H, sigma_K=4.0, sigma_n=1e-3, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    noise = rng.normal(0.0, 1e-3, L_clean.shape)
    assert np.array_equal(L, np.maximum(0.0, L_clean + noise))


def test_d6_reproducible(beam_sample):
    """D6：相同 H、d 与种子下输出逐位可复现；不同种子产生不同噪声。"""
    H, _, _ = beam_sample
    out_a = f_deg(H, sigma_K=4.0, sigma_n=1e-3, seed=5)
    out_b = f_deg(H, sigma_K=4.0, sigma_n=1e-3, seed=5)
    out_c = f_deg(H, sigma_K=4.0, sigma_n=1e-3, seed=6)
    assert np.array_equal(out_a[0], out_b[0])
    assert np.array_equal(out_a[1], out_b[1])
    assert not np.array_equal(out_a[0], out_c[0])
    assert np.array_equal(out_a[1], out_c[1])  # L_clean 与噪声种子无关（D5 辅助）


def test_d5_lclean_output(beam_sample):
    """D5：L_clean 可单独输出且不含噪声（30 [S3] C2）。"""
    H, _, _ = beam_sample
    _, L_clean, _, _ = f_deg(H, sigma_K=4.0, sigma_n=0.0, seed=0)
    assert np.isfinite(L_clean).all()


def test_block_sum_conserves_total_intensity(beam_sample):
    """[S5] C3：r×r 块求和保总强度 ΣL_clean = Σ(K*H) ≈ ΣH（禁止块均值）。"""
    H, _, _ = beam_sample
    sigma_K = 4.0
    _, L_clean, _, _ = f_deg(H, sigma_K=sigma_K, sigma_n=0.0, seed=0)
    blurred = gaussian_filter(H, sigma=sigma_K, mode="nearest")
    assert L_clean.sum() == pytest.approx(blurred.sum(), rel=1e-12)
    assert L_clean.sum() == pytest.approx(H.sum(), rel=1e-6)


def test_d2_blur_before_downsample_anti_aliasing(dot_image):
    """D2：先模糊再下采样——点源在 L_clean 中扩散，排除直接抽稀或无模糊块求和。"""
    sigma_K = 2.0
    _, L_clean, _, _ = f_deg(dot_image, sigma_K=sigma_K, sigma_n=0.0, seed=0)

    decimated = dot_image[::4, ::4]
    assert not np.allclose(L_clean, decimated)  # 非直接抽稀

    support = (L_clean > 1e-3 * L_clean.max()).sum()
    assert support >= 5  # 模糊使点源跨多个低分辨率像素；无模糊块求和仅 1 个


def test_d3_physical_range_and_correspondence(dot_image):
    """D3：L 与 H 覆盖相同物理坐标范围；低分辨率像素对应正确的 H 区域。"""
    H = np.zeros((256, 256))
    H[129, 131] = 1.0  # 落在低分辨率块 (32, 32)：129//4=32、131//4=32
    _, L_clean, _, m_L = f_deg(H, sigma_K=0.5, sigma_n=0.0, seed=0)
    assert m_L["physical_range"] == (-1.0, 1.0)
    peak = np.unravel_index(L_clean.argmax(), L_clean.shape)
    assert peak == (32, 32)
    assert m_L["pixel_scale_relation"] == "delta_L = r * delta_H"


def test_d1_same_source(beam_sample):
    """D1：L 由同一个 H 生成——ΣL_clean 与 ΣH 一致即同源一致性证据。"""
    H, _, _ = beam_sample
    _, L_clean, _, m_L = f_deg(H, sigma_K=4.0, sigma_n=0.0, seed=0)
    assert L_clean.sum() == pytest.approx(H.sum(), rel=1e-6)
    assert m_L["degradation_order"] == "blur -> downsample -> noise"


def test_d8_snr_hf_definition_and_behavior():
    """D8 判据函数：SNR_hf 定义正确且随噪声增强单调下降（30 [S6] C8）。"""
    rng = np.random.default_rng(np.random.SeedSequence(20260825))
    n = 64
    ii, jj = np.mgrid[0:n, 0:n]
    L_clean = np.exp(-(((ii - 32) / 8.0) ** 2 + ((jj - 32) / 8.0) ** 2))

    def hp_norm(x: np.ndarray) -> float:
        F = np.fft.fft2(x, norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        return float(np.sqrt(np.sum(np.abs(F[np.hypot(kx, ky) > 1.0 / 8.0]) ** 2)))

    sigma_n = 5e-2
    noise = rng.normal(0.0, sigma_n, L_clean.shape)
    L = np.maximum(0.0, L_clean + noise)
    expected = hp_norm(L_clean) / hp_norm(L - L_clean)
    assert snr_hf(L, L_clean) == pytest.approx(expected, rel=1e-12)

    snr_low_noise = snr_hf(L_clean + 0.01 * noise, L_clean)
    snr_high_noise = snr_hf(L_clean + 10.0 * noise, L_clean)
    assert snr_low_noise > snr_high_noise


def test_d8_snr_hf_batch_median_convention(beam_sample, sigma_K_px):
    """D8：真实样本上 SNR_hf 逐样本可计算（批量中位数判定见验收测试）。"""
    H, _, _ = beam_sample
    sigma_n = float(H.sum() / 64.0**2 / 3.0)
    values = []
    for seed in (1, 2, 3):
        L, L_clean, _, _ = f_deg(H, sigma_K=sigma_K_px, sigma_n=sigma_n, seed=seed)
        values.append(snr_hf(L, L_clean))
    assert all(np.isfinite(v) and v > 0 for v in values)


def test_d9_large_scale_structure_visible(beam_sample, sigma_K_px):
    """D9：大尺度结构在 L 中仍可识别——低频能量占比高且图像有明显峰值。"""
    H, _, _ = beam_sample
    sigma_n = float(H.sum() / 64.0**2 / 3.0)
    _, L_clean, _, _ = f_deg(H, sigma_K=sigma_K_px, sigma_n=sigma_n, seed=1)

    F = np.fft.fft2(L_clean, norm="ortho")
    kx, ky = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij")
    power = np.abs(F) ** 2
    low = power[np.hypot(kx, ky) <= 1.0 / 8.0].sum()
    assert low / power.sum() > 0.8
    assert L_clean.max() / L_clean.mean() > 3.0


def test_d10_metadata_complete(beam_sample):
    """D10：d 与 m_L 字段完整；SNR 采用定死定义 mean(L_clean)/σ_n（30 [S9] C3）。"""
    H, _, _ = beam_sample
    sigma_n = 1e-4
    L, L_clean, d, m_L = f_deg(H, sigma_K=4.0, sigma_n=sigma_n, seed=2)
    assert set(d) == {"r", "sigma_K", "rho_K", "noise_model", "sigma_n"}
    assert d["rho_K"] == 1.0
    assert d["noise_model"] == "additive_gaussian"
    required = {
        "r",
        "N_H",
        "N_L",
        "sigma_K",
        "rho_K",
        "noise_model",
        "sigma_n",
        "SNR",
        "seed",
        "physical_range",
    }
    assert required <= set(m_L.keys())
    assert m_L["SNR"] == pytest.approx(L_clean.mean() / sigma_n, rel=1e-12)


def test_degradation_level_sigma_K_rules():
    """30 [S7]/[S12]：D1=1×w_fine 中位数、D2=2×、EXP-03=2×D2、EXP-04=D2。"""
    w_median = 4.7
    assert sigma_K_for_level("D1", w_median) == pytest.approx(1.0 * w_median)
    assert sigma_K_for_level("D2", w_median) == pytest.approx(2.0 * w_median)
    assert sigma_K_for_level("EXP-03", w_median) == pytest.approx(4.0 * w_median)
    assert sigma_K_for_level("EXP-04", w_median) == pytest.approx(2.0 * w_median)
    with pytest.raises(ValueError):
        sigma_K_for_level("D3", w_median)  # 30 [S7] C1：不设 D3 等级


def test_input_not_modified(beam_sample):
    """[S2] C3：退化不修改 H 本身。"""
    H, _, _ = beam_sample
    snapshot = H.copy()
    f_deg(H, sigma_K=4.0, sigma_n=1e-3, seed=1)
    assert np.array_equal(H, snapshot)


def test_d11_d12_no_prior_no_network_logic():
    """D11/D12：实现不含先验生成逻辑与网络逻辑。"""
    source = inspect.getsource(f_deg_mod)
    assert "f_prior" not in source
    assert "torch" not in source
    assert "nn." not in source
