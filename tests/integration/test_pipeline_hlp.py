"""f_beam → f_deg → f_prior 全链集成测试。

覆盖规格：20 [S11] C1–C2（20 向 30 提供 H、向 40 提供 c 的接口）、
30 [S10] C1–C4（L 与 H 同一样本、坐标一致、尺寸固定）、40 [S11] C1–C2
（40 从 20 接收 c、不接收 H）；60 [S14] 写入分工（20 写 c/m/H、30 追加
L/m_L、40 追加 P2）在函数级接口上的对应；种子派生统一经
SeedSequence 分支（60 [S14] C4）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generators.f_beam import f_beam
from src.generators.f_deg import f_deg, snr_hf
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.generators.masks import apply_masks, DELTA_PX, fine_structure_width
from src.generators.sampling import sample_parameters

pytestmark = [pytest.mark.integration, pytest.mark.m1]

N_CHAIN_SAMPLES = 4
SIGMA_K_PX = 9.0


@pytest.fixture(scope="module")
def chain_samples():
    """固定主种子采样的 4 组参数及其全链生成结果。"""
    params_list, _ = sample_parameters(
        N_CHAIN_SAMPLES, master_seed=20260825, sigma_K=SIGMA_K_PX
    )
    seed_seq = np.random.SeedSequence(20260825).spawn(N_CHAIN_SAMPLES)
    chains = []
    for params, ss in zip(params_list, seed_seq):
        sigma_smooth_h = 0.5 * float(fine_structure_width(params) / DELTA_PX)
        H, m, c_rec = f_beam(params, sigma_smooth=sigma_smooth_h)
        sigma_n = float(H.sum() / 64.0**2 / 3.0)
        noise_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
        L, L_clean, d, m_L = f_deg(
            H, sigma_K=SIGMA_K_PX, sigma_n=sigma_n, seed=noise_seed
        )
        P, meta = f_prior(params, level="P2", sigma_smooth=2.0 * sigma_smooth_h)
        chains.append(
            {
                "c": params,
                "H": H,
                "m": m,
                "c_rec": c_rec,
                "L": L,
                "L_clean": L_clean,
                "d": d,
                "m_L": m_L,
                "P": P,
                "meta": meta,
            }
        )
    return chains


def test_chain_shapes_and_normalization(chain_samples):
    """全链输出形状与归一化契约：H 256²、L 64²、P 256²，ΣH=ΣP=1。"""
    for ch in chain_samples:
        assert ch["H"].shape == (256, 256)
        assert ch["L"].shape == (64, 64)
        assert ch["L_clean"].shape == (64, 64)
        assert ch["P"].shape == (256, 256)
        assert ch["H"].sum() == pytest.approx(1.0, rel=1e-9)
        assert ch["P"].sum() == pytest.approx(1.0, rel=1e-9)
        assert ch["H"].min() >= 0.0
        assert ch["L"].min() >= 0.0
        assert ch["P"].min() >= 0.0


def test_chain_same_source(chain_samples):
    """H、L、P 来自同一组参数：m 记录的 c 与输入一致，L 总强度与 H 一致。

    ``ΣL_clean = Σ(K*H)`` 精确成立（块求和保总强度，见
    ``test_block_sum_conserves_total_intensity``）；``Σ(K*H)`` 与 ``ΣH``
    之差仅来自模糊核在图像边界处的截断，故此处用较宽容差。
    """
    for ch in chain_samples:
        groups = ch["m"]["c"]
        flat = {**groups["c_low"], **groups["c_mid"], **groups["c_high"]}
        for key, value in flat.items():
            assert value == ch["c"][key], key
        assert ch["L_clean"].sum() == pytest.approx(ch["H"].sum(), rel=1e-3)
        assert ch["m_L"]["physical_range"] == ch["m"]["render"]["coordinate_range"]


def test_chain_l_up_contract(chain_samples):
    """L_up 契约：4×双线性上采样后总强度归一化到 1（50 [S8]/[S13]、60 [S3]）。"""
    for ch in chain_samples:
        L_up = normalize_intensity(upsample_4x_bilinear(ch["L"]))
        assert L_up.shape == (256, 256)
        assert L_up.sum() == pytest.approx(1.0, rel=1e-9)
        assert L_up.min() >= 0.0


def test_chain_prior_invariant_to_c_high(chain_samples):
    """链上 P2 对 c_high 扰动逐位不变（★ 泄露防护的集成层复证）。"""
    for ch in chain_samples:
        c_mod = dict(ch["c"], a3=0.045, gamma=0.52, b1=0.17)
        sigma_smooth_h = 0.5 * float(fine_structure_width(ch["c"]) / DELTA_PX)
        P_mod, _ = f_prior(c_mod, level="P2", sigma_smooth=2.0 * sigma_smooth_h)
        assert np.array_equal(ch["P"], P_mod)


def test_chain_snr_hf_computable(chain_samples):
    """每个链样本的 SNR_hf 逐样本可计算且有限（30 [S6] C8 的接口前提）。"""
    for ch in chain_samples:
        value = snr_hf(ch["L"], ch["L_clean"])
        assert np.isfinite(value)
        assert value > 0.0


def test_chain_masks_passed(chain_samples):
    """链上全部参数组通过 W1–W8（入选样本契约，20 [S9] C7）。"""
    for ch in chain_samples:
        results = apply_masks(ch["c"], sigma_K=SIGMA_K_PX)
        assert all(bool(v) for v in results.values())


def test_write_division_interfaces():
    """写入分工的函数级接口：20 出 H/m/c，30 追加 L 四元组，40 追加 P。"""
    import inspect

    from src.generators.f_beam import f_beam as fb
    from src.generators.f_deg import f_deg as fd
    from src.generators.f_prior import f_prior as fp

    assert set(inspect.signature(fb).parameters) == {"c", "grid", "sigma_smooth", "q_total"}
    assert set(inspect.signature(fd).parameters) == {"H", "sigma_K", "sigma_n", "r", "seed"}
    assert set(inspect.signature(fp).parameters) == {"c", "level", "grid", "sigma_smooth"}
    # 30 不接收 c/m（不引入束流物理），40 不接收 H 与 L（先验公平性）
    assert "c" not in inspect.signature(fd).parameters
    assert "H" not in inspect.signature(fp).parameters
    assert "L" not in inspect.signature(fp).parameters
