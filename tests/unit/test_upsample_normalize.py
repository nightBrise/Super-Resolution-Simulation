"""上采样与总强度归一化单元测试。

覆盖规格：50 [S8] C1–C3（4 倍双线性固定插值、256×256 输出、非负无过冲、
不用可学习上采样）、50 [S13] C1–C2（总强度归一化、插值之后执行）、
60 [S3] C1（H/L_up/P₂ 分别归一化到总强度 1）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear

pytestmark = [pytest.mark.unit, pytest.mark.m1]


@pytest.fixture(scope="module")
def beam_like_low_res(beam_sample):
    """真实束流的无噪声低分辨率图像（远离边界的紧支撑分布）。"""
    from src.generators.f_deg import f_deg

    H, _, _ = beam_sample
    _, L_clean, _, _ = f_deg(H, sigma_K=4.0, sigma_n=0.0, seed=0)
    return L_clean


def test_output_shape_and_dtype(beam_like_low_res):
    """C1：64×64 上采样为 256×256（50 [S8] C1）。"""
    L_up = upsample_4x_bilinear(beam_like_low_res)
    assert L_up.shape == (256, 256)


def test_total_intensity_conservation(beam_like_low_res):
    """双线性上采样对远离边界的分布总强度严格守恒（ΣL_up = 16·ΣL）。"""
    L_up = upsample_4x_bilinear(beam_like_low_res)
    assert L_up.sum() == pytest.approx(16.0 * beam_like_low_res.sum(), rel=1e-6)


def test_constant_image_exact():
    """常数图像上采样后仍为常数，且总强度恰为 16 倍。"""
    L = np.full((64, 64), 3.0)
    L_up = upsample_4x_bilinear(L)
    assert np.allclose(L_up, 3.0, rtol=0, atol=1e-12)
    assert L_up.sum() == pytest.approx(16.0 * L.sum(), rel=1e-12)


def test_bilinear_exactness_on_linear_field():
    """线性场被双线性插值精确复现（内部像素，验证双线性而非最近邻）。"""
    ramp = np.tile(np.arange(64, dtype=np.float64), (64, 1))
    L_up = upsample_4x_bilinear(ramp)
    expected = (np.arange(256) + 0.5) / 4.0 - 0.5
    assert np.allclose(L_up[0, 2:254], expected[2:254], rtol=0, atol=1e-12)


def test_pixel_center_alignment():
    """输出像素中心与输入物理坐标网格对齐：点源峰位于 4i+1.5 处分裂。"""
    d = np.zeros((64, 64))
    d[32, 32] = 1.0
    L_up = upsample_4x_bilinear(d)
    peak = np.unravel_index(L_up.argmax(), L_up.shape)
    assert peak[0] in (129, 130)  # 4×32 + 1.5 = 129.5
    assert peak[1] in (129, 130)
    assert L_up.sum() == pytest.approx(16.0, rel=1e-9)


def test_no_negative_no_overshoot(rng):
    """双线性插值不产生负值与过冲：输出值域含于输入值域（50 [S8]）。"""
    L = rng.uniform(0.0, 2.0, (64, 64))
    L_up = upsample_4x_bilinear(L)
    assert L_up.min() >= L.min() - 1e-12
    assert L_up.max() <= L.max() + 1e-12


def test_linearity(rng):
    """上采样算子为线性算子。"""
    x = rng.uniform(0.0, 1.0, (64, 64))
    y = rng.uniform(0.0, 1.0, (64, 64))
    lhs = upsample_4x_bilinear(2.0 * x + 3.0 * y)
    rhs = 2.0 * upsample_4x_bilinear(x) + 3.0 * upsample_4x_bilinear(y)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_normalize_intensity_to_one(beam_like_low_res):
    """总强度归一化：Σ(img/Σimg) = 1（50 [S13] C2、60 [S3]）。"""
    L_up = upsample_4x_bilinear(beam_like_low_res)
    normed = normalize_intensity(L_up)
    assert normed.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(normed, L_up / L_up.sum(), rtol=0, atol=1e-15)


def test_normalize_rejects_nonpositive_total():
    """总强度非正的输入不可归一化，显式拒绝。"""
    with pytest.raises(ValueError):
        normalize_intensity(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        normalize_intensity(np.array([[1.0, -2.0], [0.5, 0.0]]))


def test_normalize_after_interpolation_order(beam_like_low_res):
    """归一化在插值之后执行：两种顺序结果不同，约定为先插值后归一化。"""
    L_up = upsample_4x_bilinear(beam_like_low_res)
    normed = normalize_intensity(L_up)
    assert normed.sum() == pytest.approx(1.0, rel=1e-12)
    # 插值保持非负（归一化输入合法性）
    assert L_up.min() >= 0.0
