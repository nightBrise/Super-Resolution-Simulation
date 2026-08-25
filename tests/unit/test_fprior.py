"""物理先验生成函数 ``f_prior`` 单元测试。

覆盖规格：40 [S2] C1–C6（同源、非真值、可复现）、[S4] C1–C4（参数分层、
c_prior 不含 A 与 c_high）、[S5] C1–C4（图像先验生成与平滑约束）、
[S6] C1–C5（P1/P2/P3 等级定义）、[S9] C1–C4（公平性：不用 H、不依赖 L、
不含 c_high）、[S10] C1–C3（输出与元数据）、[S12] AC1–AC13
（AC14 质量门为批量校准判据，见验收测试）。

★ 防泄露用例：`test_c_high_invariance` 为本项目最重要的泄露防护测试
（05 [S3.2]），验证固定 c_low+c_mid 时 P2 对 a₃/γ/b₁ 扰动逐位不变。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.generators.f_prior as f_prior_mod
from src.generators.f_prior import f_prior, prior_parameters

pytestmark = [pytest.mark.unit, pytest.mark.m1]


def test_ac1_same_source(consistent_c, sigma_smooth_P_px):
    """AC1：先验由与 H 相同的物理参数 c 生成（同源，记录值与 c 一致）。"""
    _, meta = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    for key, value in meta["c_prior"].items():
        assert value == consistent_c[key], key


def test_ac2_not_equal_to_H(beam_sample, consistent_c, sigma_smooth_P_px):
    """AC2：先验不使用 H 且 P ≠ H（40 [S9] C1）。"""
    H, _, _ = beam_sample
    P, _ = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    assert np.abs(P - H).sum() > 1e-3
    # 签名层面：f_prior 不接受 H 或 L 作为输入
    params = set(inspect.signature(f_prior).parameters)
    assert params == {"c", "level", "grid", "sigma_smooth"}


def test_c_high_invariance(consistent_c, sigma_smooth_P_px):
    """★★ AC3：固定 c_low+c_mid，分别扰动 a₃/γ/b₁，P2 逐位不变。"""
    P_ref, meta_ref = f_prior(
        consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px
    )
    for key, perturbed in (("a3", 0.05), ("gamma", 0.55), ("b1", 0.19)):
        c_mod = dict(consistent_c)
        c_mod[key] = perturbed
        P_mod, meta_mod = f_prior(c_mod, level="P2", sigma_smooth=sigma_smooth_P_px)
        assert np.array_equal(P_ref, P_mod), key
        assert meta_ref["c_prior"] == meta_mod["c_prior"], key
    # 同时扰动三个 c_high 参数仍逐位不变
    c_all = dict(consistent_c, a3=0.04, gamma=-0.55, b1=-0.09)
    P_all, _ = f_prior(c_all, level="P2", sigma_smooth=sigma_smooth_P_px)
    assert np.array_equal(P_ref, P_all)


def test_c_high_keys_absent_from_record(consistent_c, sigma_smooth_P_px):
    """AC3：P2/P1 的 c_prior 记录不含 c_high = {a₃, γ, b₁}（40 [S4] C3）。"""
    for level, expected_keys in (
        ("P2", {"sigma_z", "n", "eta", "b0", "a1", "alpha", "a2", "beta"}),
        ("P1", {"sigma_z", "n", "eta", "b0", "a1", "alpha"}),
    ):
        _, meta = f_prior(consistent_c, level=level, sigma_smooth=sigma_smooth_P_px)
        assert set(meta["c_prior"]) == expected_keys
        assert not (set(meta["c_prior"]) & {"a3", "gamma", "b1"})
        assert "A" not in meta["c_prior"]  # 总强度归一化下不含 A（40 [S4] C4）


def test_p1_invariant_to_c_mid(consistent_c, sigma_smooth_P_px):
    """P1 只保留 c_low：扰动 a₂/β（及 c_high）时 P1 逐位不变（40 [S6] C2）。"""
    P_ref, _ = f_prior(consistent_c, level="P1", sigma_smooth=sigma_smooth_P_px)
    c_mod = dict(consistent_c, a2=0.09, beta=-1.8, a3=0.04, gamma=0.5, b1=0.1)
    P_mod, _ = f_prior(c_mod, level="P1", sigma_smooth=sigma_smooth_P_px)
    assert np.array_equal(P_ref, P_mod)


def test_p2_parameter_definitions(consistent_c):
    """P2 生成参数：a₂_P=a₂、β_P=β、a₃_P=γ_P=b₁_P=0（40 [S6] C3）。"""
    c_p = prior_parameters(consistent_c, "P2")
    assert c_p["a2"] == consistent_c["a2"]
    assert c_p["beta"] == consistent_c["beta"]
    assert (c_p["a3"], c_p["gamma"], c_p["b1"]) == (0.0, 0.0, 0.0)


def test_p1_parameter_definitions(consistent_c):
    """P1 生成参数：a₂_P=a₃_P=β_P=γ_P=b₁_P=0（40 [S6] C2）。"""
    c_p = prior_parameters(consistent_c, "P1")
    for key in ("a2", "a3", "beta", "gamma", "b1"):
        assert c_p[key] == 0.0


def test_ac4_nonnegative_and_normalized(consistent_c, sigma_smooth_P_px):
    """AC4/AC5：P ≥ 0、ΣP = 1、归一化方式记录于元数据。"""
    P, meta = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    assert P.min() >= 0.0
    assert P.sum() == pytest.approx(1.0, rel=1e-9)
    assert meta["normalization"] == "sum-to-1"


def test_ac6_smoother_than_H(
    beam_sample, consistent_c, sigma_smooth_H_px, sigma_smooth_P_px
):
    """AC6：σ_smooth,P > σ_smooth,H 且 P 的结构带能量低于 H（40 [S5] C3/C4）。

    比较频带取 (1/32, 1/8]：f > 1/8 频带对这两类光滑图像的能量均低于
    离散化噪声水平，不具备判别力；结构带内 P 与 H 的差距由平滑核宽度
    2 倍差决定，实测比值 < 0.6。
    """
    H, _, _ = beam_sample
    P, meta = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    assert sigma_smooth_P_px > sigma_smooth_H_px
    assert meta["smoothing"] == sigma_smooth_P_px

    def band_power(img: np.ndarray) -> float:
        F = np.fft.fft2(img, norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
        f = np.hypot(kx, ky)
        band = (f > 1.0 / 32.0) & (f <= 1.0 / 8.0)
        return float(np.sum(np.abs(F[band]) ** 2))

    assert band_power(P) < 0.75 * band_power(H)


def test_ac7_level_configurable(consistent_c, sigma_smooth_P_px):
    """AC7：先验等级可配置；P0 无图像先验、非法等级拒绝（40 [S6] C1）。"""
    P1, meta1 = f_prior(consistent_c, level="P1", sigma_smooth=sigma_smooth_P_px)
    P2, meta2 = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    P3, meta3 = f_prior(
        consistent_c, level="P3", sigma_smooth=sigma_smooth_P_px * 1.5
    )
    assert meta1["level"] == "P1"
    assert meta2["level"] == "P2"
    assert meta3["level"] == "P3"
    assert not np.array_equal(P1, P2)
    with pytest.raises(ValueError):
        f_prior(consistent_c, level="P0")
    with pytest.raises(ValueError):
        f_prior(consistent_c, level="X9")


def test_ac8_oracle_distinction(consistent_c, sigma_smooth_P_px):
    """AC8：realistic（P1/P2）与 oracle（P3）在元数据中明确区分（40 [S12] AC8）。"""
    _, meta2 = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    _, meta3 = f_prior(
        consistent_c, level="P3", sigma_smooth=sigma_smooth_P_px * 1.5
    )
    assert meta2["oracle"] is False
    assert meta2["prior_kind"] == "realistic"
    assert meta3["oracle"] is True
    assert meta3["prior_kind"] == "oracle"


def test_p3_not_equal_to_H(beam_sample, consistent_c, sigma_smooth_P_px):
    """P3 用完整参数但更强平滑，仍满足 P ≠ H（40 [S6] C5）。"""
    H, _, _ = beam_sample
    P, _ = f_prior(consistent_c, level="P3", sigma_smooth=sigma_smooth_P_px * 1.5)
    assert np.abs(P - H).sum() > 1e-3


def test_ac9_metadata_complete(consistent_c, sigma_smooth_P_px):
    """AC9：元数据完整记录 c_prior、level、type、smoothing、grid size、normalization。"""
    _, meta = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    required = {"level", "type", "oracle", "c_prior", "smoothing", "grid_size", "normalization"}
    assert required <= set(meta.keys())
    assert meta["type"] == "image"
    assert meta["grid_size"] == 256


def test_ac10_reproducible(consistent_c, sigma_smooth_P_px):
    """AC10：给定 c、等级与配置，先验生成逐位可复现。"""
    P1, m1 = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    P2, m2 = f_prior(consistent_c, level="P2", sigma_smooth=sigma_smooth_P_px)
    assert np.array_equal(P1, P2)
    assert m1 == m2


def test_ac11_ac12_no_network_no_loss_logic():
    """AC11/AC12：先验生成代码不含网络结构逻辑与训练损失逻辑。"""
    source = inspect.getsource(f_prior_mod)
    assert "torch" not in source
    assert "nn." not in source
    assert "Loss" not in source


def test_ac13_independent_of_L_noise(consistent_c, sigma_smooth_P_px):
    """AC13：先验不依赖 L 的噪声实现——签名无 L/seed 参数且重复生成一致。"""
    params = set(inspect.signature(f_prior).parameters)
    assert not (params & {"L", "seed", "noise", "sigma_n"})
    source = inspect.getsource(f_prior_mod)
    assert "np.random" not in source
    assert "default_rng" not in source
