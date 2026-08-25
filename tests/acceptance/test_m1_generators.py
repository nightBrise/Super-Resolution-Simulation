"""里程碑 M1 验收测试：20/30/40 全部验收标准与网格收敛。

覆盖规格：20 [S10] AC1–AC12、30 [S11] D1–D12、40 [S12] AC1–AC14、
20 [S3] C5（网格收敛，slow）、30 [S6] C8（SNR_hf 批量中位数判据）、
40 [S12] AC14（先验质量门，M1 用自生成校准批与初始 σ_n/σ_smooth,P）。

初始标定口径（M1 阶段，EXP-01 标定前）：
- σ_K 取 D2 初始值 2×w_fine 批量中位数（30 [S12] C3 标定规则的初始值）；
- σ_n 取尾部区域信噪比 2（30 [S12] C4 登记带 2–5 的下档，配合 [S6] C8
  判据使用；标定值以 EXP-01 为准）；
- σ_smooth,H = 0.5×w_fine（逐样本口径，20 [S3] C4）；
- σ_smooth,P = 2.6×σ_smooth,H（40 [S5] C5 登记候选集 2××[0.7, 0.85, 1.0,
  1.15, 1.3] 的上档候选：初始中心 2× 在本文件 AC14 质量门下比值低于 0.2
  判为先验过度逼近真值，故 M1 自校准取满足质量门的上档候选；
  最终值以 EXP-01d 标定登记为准）。

测试铁律（05 [S1]）：本文件只断言协议与不变量（如「批量中位数 < 0.1」为
规格定死的生成质量门），不断言任何研究结果。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.generators.f_beam as f_beam_mod
from src.generators.f_beam import current_profile, f_beam, pixel_center_coordinates
from src.generators.f_deg import f_deg, snr_hf
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.generators.masks import apply_masks, DELTA_PX, fine_structure_width
from src.generators.sampling import sample_parameters

pytestmark = [pytest.mark.acceptance, pytest.mark.m1]

MASTER_SEED = 20260825

#: M1 初始 σ_n 口径：尾部区域信噪比下档（30 [S12] C4 登记带 2–5）。
INITIAL_TAIL_SNR = 2.0

#: M1 采用的 σ_smooth,P 倍数（相对 σ_smooth,H；40 [S5] C5 登记候选上档）。
SIGMA_SMOOTH_P_MULTIPLE = 2.6


def _wf_px(c) -> float:
    """样本精细结构宽度的像素值。"""
    return float(fine_structure_width(c) / DELTA_PX)


@pytest.fixture(scope="module")
def acceptance_params():
    """固定主种子采样的 24 组入选参数（σ_K=9px）。"""
    params, stats = sample_parameters(24, master_seed=MASTER_SEED, sigma_K=9.0)
    return params, stats


@pytest.fixture(scope="module")
def sigma_K_d2_initial(acceptance_params):
    """D2 初始模糊核：2×w_fine 批量中位数（30 [S12] C3 初始值口径）。"""
    params, _ = acceptance_params
    widths = np.array([_wf_px(c) for c in params])
    return float(2.0 * np.median(widths))


# ---------------------------------------------------------------------------
# 20 [S9]/[S10]：采样掩膜统计与 Level 1 验收
# ---------------------------------------------------------------------------


def test_w8_fraction_monitor(acceptance_params):
    """20 [S9] C9：W1–W7 通过者中满足精细结构窗口的比例 ≥ 60%。"""
    _, stats = acceptance_params
    assert stats["w8_fraction_among_w1_w7_passers"] >= 0.6


def test_ac1_ac2_ac3_ac11_batch(acceptance_params):
    """AC1/AC2/AC3/AC11：批量可复现、非负、总强度归一、标签完整。"""
    params, _ = acceptance_params
    for c in params[:12]:
        sigma_smooth = 0.5 * _wf_px(c)
        H1, m1, _ = f_beam(c, sigma_smooth=sigma_smooth)
        H2, m2, _ = f_beam(c, sigma_smooth=sigma_smooth)
        assert np.array_equal(H1, H2)  # AC1
        assert m1["Q"] == m2["Q"]
        assert H1.min() >= 0.0  # AC2
        assert H1.shape == (256, 256)
        assert H1.sum() == pytest.approx(1.0, rel=1e-9)  # AC3
        assert m1["Q"] == pytest.approx(1.0, rel=1e-9)
        # AC11：[S7] 全部标签且自洽
        assert m1["h_eff"] == pytest.approx(
            m1["C_zdelta"] / m1["sigma_z"] ** 2, rel=1e-12
        )
        assert m1["eps_z"] == pytest.approx(
            np.sqrt(
                m1["sigma_z"] ** 2 * m1["sigma_delta"] ** 2 - m1["C_zdelta"] ** 2
            ),
            rel=1e-12,
        )
        assert np.allclose(m1["I_z"], H1.sum(axis=1), rtol=0, atol=1e-15)
        assert m1["Level"] == 1
        assert m1["compression_state"] in ("under", "optimal", "over")


def test_ac4_continuity_batch(acceptance_params):
    """AC4：批量样本上参数连续变化 → H 连续变化（ε 减半，变化量近似减半）。"""
    params, _ = acceptance_params
    for c in params[:3]:
        sigma_smooth = 0.5 * _wf_px(c)
        H0, _, _ = f_beam(c, sigma_smooth=sigma_smooth)
        eps = 0.02
        d_full = np.abs(
            f_beam(dict(c, a2=c["a2"] + eps), sigma_smooth=sigma_smooth)[0] - H0
        ).sum()
        d_half = np.abs(
            f_beam(dict(c, a2=c["a2"] + eps / 2), sigma_smooth=sigma_smooth)[0] - H0
        ).sum()
        assert 0.3 <= d_half / d_full <= 0.7


def test_ac5_flat_top():
    """AC5：n > 1 剖面比高斯更平顶（解析式，20 [S4] C3）。"""
    z = np.array([0.25])
    c_gauss = {"A": 1.0, "sigma_z": 0.5, "n": 1.0, "eta": 0.0}
    c_flat = {"A": 1.0, "sigma_z": 0.5, "n": 3.0, "eta": 0.0}
    assert current_profile(z, c_flat)[0] > current_profile(z, c_gauss)[0]


def _clean_c(**overrides) -> dict[str, float]:
    """无折叠、无 chirp 的简化参数组，用于隔离单一物理要素。"""
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


def test_ac6_head_tail_asymmetry():
    """AC6：η ≠ 0 时剖面头尾不对称（投影一阶矩随 η 符号反转）。"""
    coords, _ = pixel_center_coordinates(256)
    moment = {}
    for eta in (0.2, -0.2):
        H, _, _ = f_beam(_clean_c(eta=eta), sigma_smooth=2.0)
        moment[eta] = float((coords * H.sum(axis=1)).sum())
    assert moment[0.2] - moment[-0.2] > 0.05


def test_ac7_thickness_variation():
    """AC7：b₁ ≠ 0 时局部厚度随 |z| 增大，b₁ = 0 时恒定。"""
    coords, _ = pixel_center_coordinates(256)

    def ratio(b1: float) -> float:
        H, _, _ = f_beam(_clean_c(b1=b1), sigma_smooth=2.0)

        def rms(band: np.ndarray) -> float:
            rows = H[band]
            mass = rows.sum()
            mu = (rows * coords[None, :]).sum() / mass
            return float(np.sqrt((rows * (coords[None, :] - mu) ** 2).sum() / mass))

        center = rms(np.abs(coords) <= 0.1)
        tail = rms((np.abs(coords) >= 0.8) & (np.abs(coords) <= 1.2))
        return tail / center

    assert 0.95 <= ratio(0.0) <= 1.05
    assert ratio(0.2) > 1.3


def test_ac8_s_shape():
    """AC8：a₃ ≠ 0 时中心线出现三阶 S 形（二次拟合残差的奇对称分量）。"""

    def cubic_moment(a3: float) -> float:
        H, _, _ = f_beam(_clean_c(a1=0.5, a2=0.05, a3=a3), sigma_smooth=2.0)
        zc, _ = pixel_center_coordinates(256)
        rowmass = H.sum(axis=1)
        keep = rowmass > 1e-4 * rowmass.max()
        mu_delta = (H * zc[None, :]).sum(axis=1) / np.maximum(rowmass, 1e-300)
        poly = np.polyfit(zc[keep], mu_delta[keep], 2, w=rowmass[keep])
        resid = mu_delta[keep] - np.polyval(poly, zc[keep])
        weight = rowmass[keep]
        return float((weight * resid * np.sign(zc[keep])).sum() / weight.sum())

    assert abs(cubic_moment(0.0)) < 1e-4
    assert cubic_moment(0.05) > 2e-4


def test_ac9_folding():
    """AC9：β ≠ 0 时结构带高频能量显著上升（折叠/细脊出现）。"""

    def band_energy(c: dict[str, float]) -> float:
        H, _, _ = f_beam(c, sigma_smooth=1.0)
        F = np.fft.fft2(H, norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
        band = (np.hypot(kx, ky) > 1.0 / 16.0) & (np.hypot(kx, ky) <= 0.5)
        return float(np.sum(np.abs(F[band]) ** 2))

    base = _clean_c(a1=0.5, alpha=-1.0, a2=0.05)
    assert band_energy(dict(base, beta=1.2)) > 5.0 * band_energy(base)


def test_ac10_ac12(acceptance_params):
    """AC10：精细结构非随机（两次生成逐位一致）；AC12：无退化逻辑。"""
    params, _ = acceptance_params
    c = params[0]
    H1, _, _ = f_beam(c, sigma_smooth=0.5 * _wf_px(c))
    H2, _, _ = f_beam(c, sigma_smooth=0.5 * _wf_px(c))
    assert np.array_equal(H1, H2)
    source = inspect.getsource(f_beam_mod)
    assert "f_deg" not in source
    assert "default_rng" not in source


@pytest.mark.slow
def test_grid_convergence_512(consistent_c):
    """20 [S3] C5：渲染网格 256 → 512 不得显著改变图像结构（slow）。

    判定：512 渲染块求和降回 256 并与 256 渲染归一化对齐后，
    逐像素归一化 L1 差 < 5e-3、物理标签矩量相对差 < 2e-3、
    电流剖面形状相关 > 0.9999。
    """
    wf = _wf_px(consistent_c)
    H256, m256, _ = f_beam(consistent_c, grid=256, sigma_smooth=0.5 * wf)
    H512, m512, _ = f_beam(consistent_c, grid=512, sigma_smooth=wf)

    H512_down = H512.reshape(256, 2, 256, 2).sum(axis=(1, 3))
    H512_down = H512_down / H512_down.sum()
    assert np.abs(H512_down - H256).sum() < 5e-3

    for key in ("sigma_z", "sigma_delta", "mu_z", "mu_delta", "h_eff", "eps_z"):
        assert m512[key] == pytest.approx(m256[key], rel=2e-3), key

    # I_peak 为逐行求和量，随网格细化按 Δ 缩放；比较剖面形状而非幅值
    iz256 = m256["I_z"] / m256["I_z"].sum()
    iz512 = (m512["I_z"][0::2] + m512["I_z"][1::2])
    iz512 = iz512 / iz512.sum()
    assert np.corrcoef(iz256, iz512)[0, 1] > 0.9999


# ---------------------------------------------------------------------------
# 30 [S11]：退化验收 D1–D12
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def degradation_batch(acceptance_params, sigma_K_d2_initial):
    """24 样本的退化批量：σ_K 取 D2 初始值，σ_n 取尾部信噪比 2 档。"""
    params, _ = acceptance_params
    batch = []
    for i, c in enumerate(params):
        sigma_smooth = 0.5 * _wf_px(c)
        H, m, _ = f_beam(c, sigma_smooth=sigma_smooth)
        _, L_clean0, _, _ = f_deg(H, sigma_K=sigma_K_d2_initial, sigma_n=0.0, seed=0)
        sigma_n = float(L_clean0.mean() / INITIAL_TAIL_SNR)
        L, L_clean, d, m_L = f_deg(
            H, sigma_K=sigma_K_d2_initial, sigma_n=sigma_n, seed=MASTER_SEED + i
        )
        batch.append(
            {"H": H, "m": m, "L": L, "L_clean": L_clean, "d": d, "m_L": m_L}
        )
    return batch


def test_d1_d3_d5_d7_d10_batch(degradation_batch):
    """D1/D3/D5/D7/D10：同源、坐标一致、L_clean 单独输出、尺寸与元数据完整。"""
    for item in degradation_batch:
        assert item["L"].shape == (64, 64)  # D7
        assert item["L_clean"].shape == (64, 64)  # D5
        assert item["m_L"]["physical_range"] == (-1.0, 1.0)  # D3
        # ΣL_clean = Σ(K*H) 精确；与 ΣH 之差来自模糊核边界截断
        assert item["L_clean"].sum() == pytest.approx(item["H"].sum(), rel=1e-3)  # D1
        assert item["m_L"]["SNR"] == pytest.approx(
            item["L_clean"].mean() / item["m_L"]["sigma_n"], rel=1e-12
        )  # D10 / [S9] C3 定死定义
        assert item["m_L"]["degradation_order"] == "blur -> downsample -> noise"


def test_d2_fixed_order(degradation_batch):
    """D2：先模糊再下采样再加噪（元数据与噪声形状证据：噪声为 64×64）。"""
    for item in degradation_batch:
        n_eff = item["L"] - item["L_clean"]
        assert n_eff.shape == (64, 64)
        # 噪声逐像素独立：截断后仍保留非零噪声像素
        assert np.any(n_eff != 0.0)


def test_d4_nonnegative(degradation_batch):
    """D4：L ≥ 0（含非负截断）。"""
    for item in degradation_batch:
        assert item["L"].min() >= 0.0


def test_d6_reproducible(degradation_batch, sigma_K_d2_initial):
    """D6：相同 d 与种子下输出逐位可复现。"""
    item = degradation_batch[0]
    H = item["H"]
    redo = f_deg(
        H,
        sigma_K=sigma_K_d2_initial,
        sigma_n=item["m_L"]["sigma_n"],
        seed=item["m_L"]["seed"],
    )
    assert np.array_equal(redo[0], item["L"])
    assert np.array_equal(redo[1], item["L_clean"])


def test_d8_fine_structure_unresolvable(degradation_batch):
    """D8：SNR_hf 批量中位数 < 0.1（30 [S6] C8 定量判据）。

    σ_K 取 D2 初始值、σ_n 取尾部信噪比 2 档；判定值为逐样本中位数。
    """
    values = np.array([snr_hf(item["L"], item["L_clean"]) for item in degradation_batch])
    assert np.all(np.isfinite(values))
    assert np.median(values) < 0.1, f"SNR_hf 中位数 {np.median(values):.4f} ≥ 0.1"


def test_d9_large_scale_visible(degradation_batch):
    """D9：大尺度结构在 L 中仍可识别——低频能量占比 > 0.8。"""
    for item in degradation_batch:
        F = np.fft.fft2(item["L_clean"], norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij")
        power = np.abs(F) ** 2
        low = power[np.hypot(kx, ky) <= 1.0 / 8.0].sum()
        assert low / power.sum() > 0.8


def test_d11_d12_no_prior_no_network():
    """D11/D12：30 实现不含先验生成逻辑与网络逻辑。"""
    import src.generators.f_deg as f_deg_mod

    source = inspect.getsource(f_deg_mod)
    assert "f_prior" not in source
    assert "torch" not in source


# ---------------------------------------------------------------------------
# 40 [S12]：先验验收 AC1–AC14
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prior_batch(acceptance_params):
    """8 样本的先验批量：P2、σ_smooth,P = 2.6×σ_smooth,H（M1 采用值）。"""
    params, _ = acceptance_params
    batch = []
    for c in params[:8]:
        sigma_smooth_h = 0.5 * _wf_px(c)
        H, _, _ = f_beam(c, sigma_smooth=sigma_smooth_h)
        P, meta = f_prior(
            c, level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
        )
        batch.append({"c": c, "H": H, "P": P, "meta": meta})
    return batch


def test_ac1_ac2_ac4_ac5_ac9_batch(prior_batch):
    """AC1/AC2/AC4/AC5/AC9：同源、P≠H、非负、归一化记录、元数据完整。"""
    for item in prior_batch:
        assert np.abs(item["P"] - item["H"]).sum() > 1e-3  # AC2
        assert item["P"].min() >= 0.0  # AC4
        assert item["P"].sum() == pytest.approx(1.0, rel=1e-9)
        meta = item["meta"]  # AC9
        assert {"level", "type", "c_prior", "smoothing", "grid_size", "normalization"} <= set(meta)
        assert meta["normalization"] == "sum-to-1"  # AC5
        for key, value in meta["c_prior"].items():  # AC1
            assert value == item["c"][key], key


def test_ac3_c_high_invariance_batch(prior_batch):
    """AC3（★★）：批量样本上 P2 对 c_high 扰动逐位不变。"""
    for item in prior_batch:
        sigma_smooth_h = 0.5 * _wf_px(item["c"])
        c_mod = dict(item["c"], a3=0.047, gamma=0.49, b1=0.16)
        P_mod, meta_mod = f_prior(
            c_mod, level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
        )
        assert np.array_equal(item["P"], P_mod)
        assert meta_mod["c_prior"] == item["meta"]["c_prior"]
        assert not (set(meta_mod["c_prior"]) & {"a3", "gamma", "b1"})


def test_ac6_smoother_than_H_batch(prior_batch):
    """AC6：批量样本上 P2 结构带 (1/32, 1/8] 能量低于 H（σ_smooth,P > σ_smooth,H）。"""

    def band_power(img: np.ndarray) -> float:
        F = np.fft.fft2(img, norm="ortho")
        kx, ky = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
        f = np.hypot(kx, ky)
        band = (f > 1.0 / 32.0) & (f <= 1.0 / 8.0)
        return float(np.sum(np.abs(F[band]) ** 2))

    for item in prior_batch:
        assert band_power(item["P"]) < 0.75 * band_power(item["H"])


def test_ac7_ac8_levels(prior_batch):
    """AC7/AC8：等级可配置；P1 不含 c_mid；P3 标记为 oracle。"""
    item = prior_batch[0]
    c, sigma_smooth_h = item["c"], 0.5 * _wf_px(item["c"])
    P1, meta1 = f_prior(c, level="P1", sigma_smooth=2.0 * sigma_smooth_h)
    P3, meta3 = f_prior(c, level="P3", sigma_smooth=3.0 * sigma_smooth_h)
    assert not np.array_equal(P1, item["P"])
    assert set(meta1["c_prior"]) == {"sigma_z", "n", "eta", "b0", "a1", "alpha"}
    assert meta3["oracle"] is True
    assert meta1["oracle"] is False


def test_ac10_reproducible_batch(prior_batch):
    """AC10：批量样本上先验生成逐位可复现。"""
    for item in prior_batch:
        sigma_smooth_h = 0.5 * _wf_px(item["c"])
        P2, _ = f_prior(
            item["c"], level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
        )
        assert np.array_equal(item["P"], P2)


def test_ac11_ac12_ac13_no_network_loss_L():
    """AC11/AC12/AC13：先验代码无网络、无损失、不依赖 L 的噪声实现。"""
    import src.generators.f_prior as f_prior_mod

    source = inspect.getsource(f_prior_mod)
    assert "torch" not in source
    assert "Loss" not in source
    assert "np.random" not in source
    params = set(inspect.signature(f_prior).parameters)
    assert not (params & {"H", "L", "seed", "sigma_n"})


@pytest.fixture(scope="module")
def calibration_batch():
    """AC14 自生成校准批：500 样本 (H, P2, L_up) 与初始 σ_n/σ_smooth,P。"""
    params, _ = sample_parameters(500, master_seed=MASTER_SEED, sigma_K=9.0)
    records = []
    for i, c in enumerate(params):
        sigma_smooth_h = 0.5 * _wf_px(c)
        H, _, _ = f_beam(c, sigma_smooth=sigma_smooth_h)
        _, L_clean0, _, _ = f_deg(H, sigma_K=9.0, sigma_n=0.0, seed=0)
        sigma_n = float(L_clean0.mean() / INITIAL_TAIL_SNR)
        L, _, _, _ = f_deg(H, sigma_K=9.0, sigma_n=sigma_n, seed=MASTER_SEED + i)
        L_up = normalize_intensity(upsample_4x_bilinear(L))
        P, _ = f_prior(
            c, level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
        )
        records.append((H, P, L_up))
    return records


@pytest.mark.timeout(900)
def test_ac14_quality_gate_l1_ratio(calibration_batch):
    """AC14（L1 比值部分）：批量均值 ‖H−P₂‖₁/‖H−L_up‖₁ ∈ (0.2, 0.9)。

    M1 阶段用 500 样本自生成校准批与初始 σ_n/σ_smooth,P 评估
    （40 [S12] AC14；EXP-01d 标定后复评）。
    """
    ratios = np.array(
        [np.abs(H - P).sum() / np.abs(H - L_up).sum() for H, P, L_up in calibration_batch]
    )
    mean_ratio = float(ratios.mean())
    assert 0.2 < mean_ratio < 0.9, f"L1 比值批量均值 {mean_ratio:.4f} 超出 (0.2, 0.9)"


@pytest.mark.timeout(900)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "B 类跨文档冲突：70 [S3] 规定 SSIM 用 data_range=1.0 与高斯权重，"
        "而 60 [S3] 规定图像总强度归一化（像素值 ~1e-3 量级），此时 "
        "SSIM 常数项主导，任意先验的 SSIM(P2, H) 恒 ≈ 1.0，"
        "40 [S12] AC14 的上界 0.95 在该口径下不可满足。已按 05 [S7] "
        "B 类记录，待 99 裁定口径后移除本标记。"
    ),
)
def test_ac14_quality_gate_ssim(calibration_batch):
    """AC14（SSIM 部分）：批量均值 SSIM(P₂, H) ∈ [0.7, 0.95]（70 [S3] 口径）。"""
    from skimage.metrics import structural_similarity

    values = []
    for H, P, _ in calibration_batch[:100]:
        values.append(
            structural_similarity(
                P, H, win_size=7, data_range=1.0, gaussian_weights=True
            )
        )
    mean_ssim = float(np.mean(values))
    assert 0.7 <= mean_ssim <= 0.95, f"SSIM 批量均值 {mean_ssim:.6f} 超出 [0.7, 0.95]"
