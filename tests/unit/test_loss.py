"""空域-频域混合损失单元测试。

覆盖规格：60 [S2] C1–C3（L_total = L_space + λ·L_spec、五倍频程分带、
λ 冻结 1.0、不使用物理损失）与 [S2] 实现约定（FFT ÷N²、复数模、边界像素
归低频带、角点像素不计入、掩膜固定 256×256 生成一次）。

★ 防泄露用例：`test_fft_normalization_homogeneity`（05 [S3.2]），
验证 fft2÷N² 的齐次性——未 ÷N² 时频域损失量级将差 N² 倍。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

import inspect  # noqa: E402

import src.training.loss as loss_mod  # noqa: E402
from src.training.loss import (  # noqa: E402
    BAND_EDGES,
    F_C,
    F_NYQUIST,
    FROZEN_LAMBDA,
    HybridLoss,
    build_band_masks,
)

pytestmark = [pytest.mark.unit, pytest.mark.m1]

N = 256


@pytest.fixture(scope="module")
def loss_fn():
    """256×256 固定掩膜的混合损失实例。"""
    return HybridLoss(image_size=N)


def test_l_space_hand_computed(loss_fn):
    """L_space = mean(|Ĥ − H|)：常数差手算精确。"""
    H = torch.full((N, N), 1.0 / N**2, dtype=torch.float64)
    H_hat = torch.zeros((N, N), dtype=torch.float64)
    assert loss_fn.l_space(H_hat, H).item() == pytest.approx(1.0 / N**2, rel=1e-12)


def test_equal_inputs_zero_loss(loss_fn):
    """Ĥ == H：L_space == 0 且 L_spec < 1e-12（05 [S3.3] test_loss）。"""
    H = torch.rand(N, N, dtype=torch.float64)
    H = H / H.sum()
    assert loss_fn.l_space(H, H).item() == 0.0
    assert loss_fn.l_spec(H, H).item() < 1e-12
    assert loss_fn(H, H).item() < 1e-12


def test_fft_normalization_homogeneity(loss_fn):
    """★ fft2÷N² 齐次性：L_spec(a·Ĥ, a·H) == a·L_spec(Ĥ, H)（60 [S2] 实现约定）。"""
    g = torch.Generator().manual_seed(20260825)
    H = torch.rand(N, N, dtype=torch.float64, generator=g)
    H_hat = torch.rand(N, N, dtype=torch.float64, generator=g)
    base = loss_fn.l_spec(H_hat, H).item()
    for a in (0.5, 2.0, 7.25):
        scaled = loss_fn.l_spec(a * H_hat, a * H).item()
        assert scaled == pytest.approx(a * base, rel=1e-9)


def test_dc_offset_hand_computed(loss_fn):
    """均匀偏移仅改变 DC 频点：L_spec 手算精确（FFT ÷N² 与分带均值定义）。"""
    H = torch.zeros((N, N), dtype=torch.float64)
    offset = 3e-4
    H_hat = torch.full((N, N), offset, dtype=torch.float64)
    band1_size = int(loss_fn.band_masks[0].sum())
    expected = (offset / band1_size) / 5.0
    assert loss_fn.l_spec(H_hat, H).item() == pytest.approx(expected, rel=1e-9)


def test_band_masks_disjoint_union(loss_fn):
    """5 带掩膜互不相交，且并集恰为所有 f ≤ f_N 像素（角点像素除外）。"""
    masks = loss_fn.band_masks
    assert masks.shape == (5, N, N)
    assert masks.dtype == torch.bool

    union = torch.zeros((N, N), dtype=torch.bool)
    for band in masks:
        assert not torch.any(union & band)  # 两两不相交
        union |= band

    freqs = torch.fft.fftfreq(N, dtype=torch.float64)
    kx, ky = torch.meshgrid(freqs, freqs, indexing="ij")
    radius = torch.hypot(kx, ky)
    assert torch.equal(union, radius <= F_NYQUIST)

    corners = radius > F_NYQUIST
    assert corners.any()  # 2D 网格确存在角点像素
    assert not torch.any(masks[:, corners])


def test_band_edges_frozen(loss_fn):
    """频带边界为 [0, fc/4, fc/2, fc, 2fc, fN]，fc=1/8、fN=0.5（60 [S2]）。"""
    assert BAND_EDGES == (0.0, F_C / 4, F_C / 2, F_C, 2 * F_C, F_NYQUIST)
    assert F_C == 1.0 / 8.0
    assert F_NYQUIST == 0.5


def test_lambda_frozen_at_one():
    """λ 恒为 1.0 且构造器拒绝修改（预注册常数，60 [S2] C3）。"""
    assert FROZEN_LAMBDA == 1.0
    loss_fn = HybridLoss()
    assert loss_fn.lambda_spectral == 1.0
    with pytest.raises(ValueError):
        HybridLoss(lambda_spectral=0.5)
    with pytest.raises(ValueError):
        HybridLoss(lambda_spectral=2.0)


def test_total_loss_is_sum(loss_fn):
    """L_total = L_space + λ·L_spec，λ = 1.0（60 [S2] C1）。"""
    g = torch.Generator().manual_seed(7)
    H = torch.rand(N, N, dtype=torch.float64, generator=g)
    H_hat = torch.rand(N, N, dtype=torch.float64, generator=g)
    total = loss_fn(H_hat, H).item()
    expected = loss_fn.l_space(H_hat, H).item() + loss_fn.l_spec(H_hat, H).item()
    assert total == pytest.approx(expected, rel=1e-12)


def test_batched_input(loss_fn):
    """带批量维输入 (B, N, N) 形状正确。"""
    H = torch.rand(3, N, N, dtype=torch.float64)
    H_hat = torch.rand(3, N, N, dtype=torch.float64)
    out = loss_fn(H_hat, H)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_no_physical_loss_terms():
    """第一版训练不使用物理损失（60 [S2] C2）：模块不定义也不导入
    L_moment / L_marginal / L_forward 符号。"""
    source = inspect.getsource(loss_mod)
    for token in ("L_moment", "L_marginal", "L_forward"):
        assert token not in source
    assert not hasattr(loss_mod, "MomentLoss")
    assert not hasattr(loss_mod, "MarginalLoss")
    assert not hasattr(loss_mod, "ForwardLoss")
