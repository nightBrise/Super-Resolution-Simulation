"""评估指标全集（70 规格）。

三个维度：
- 图像级（70 [S3]）：MAE / MSE / PSNR（MAX=1）/ SSIM（skimage，
  ``data_range=1.0``、``gaussian_weights=True``、7×7 窗口）；
- 物理级（70 [S4]）：σ_z / σ_δ / h_eff / ε_z / I_peak 相对误差 +
  电流/能谱剖面 L1 误差；样本级幻觉标志 ``F_i`` 与方案级两层判据
  （触发率 > 20% + 配对差 bootstrap 95% CI 下界 > 0）；
- 精细结构级（70 [S5]）：ε_high（DoG，σ_outer 由
  ``exp(−2π²σ²f_c²)=0.5`` 反算，σ_inner = σ_outer/2）、ε_peak
  （8×8 NMS + 前 3 峰 + 掩膜）、R_E（FFT 高通 f > 1/8）与联合判读
  分类（真实恢复 / 疑似纹理幻觉 / 过度平滑 / 如实报告）。

统计判定（70 [S7]）：配对 Wilcoxon、bootstrap 95% CI（10,000 次配对差
单元重采样）、均值/中位数、三分类（显著正 / 等效 / 显著负）、Holm 校正
（每方案对 × 测试集一族，仅主 + 次指标）、一票否决四分支。

全部指标在 Ĥ 与 H 归一化到总强度 1 之后计算（70 [S2] C1）；本模块的
``evaluate_sample`` 内部强制归一化，调用方无需预归一化。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy import stats
from skimage.metrics import structural_similarity

from src.generators.f_beam import pixel_center_coordinates
from src.training.loss import F_C, F_NYQUIST

#: 图像边长（H/L_up/P2 固定 256×256，50 [S2]）。
IMAGE_SIZE = 256

# ---------------------------------------------------------------------------
# 预注册常量（70 [S4][S7]；与 config.yaml.template 互锁，测试对账）
# ---------------------------------------------------------------------------
#: 一票否决物理量恶化阈值 τ = 0.05（70 [S4] C5，预注册、不做标定）。
TAU = 0.05
#: 方案级触发率阈值 20%（70 [S4] C4）。
TRIGGER_RATE = 0.20
#: 预注册主指标 ε_high^mask 的 metrics.csv 列名（70 [S7.1] C1）。
PRIMARY_METRIC_COL = "e_high_mask"
#: 预注册次指标 ε_z 相对误差的 metrics.csv 列名（70 [S7.1] C1）。
SECONDARY_METRIC_COL = "e_eps_z"
#: c_high 掩膜累计能量比 90%（70 [S7.1] C2）。
MASK_ENERGY_FRACTION = 0.90
#: bootstrap 重采样次数 10,000（70 [S7.2]）。
N_BOOTSTRAP = 10_000
#: bootstrap CI 置信水平 95%。
CI_ALPHA = 0.05

#: R_E 联合判读分类标签（70 [S5.3]）。
RE_TRUE_RECOVERY = "true_recovery"          # 真实恢复
RE_TEXTURE_HALLUCINATION = "texture_hallucination"  # 疑似纹理幻觉
RE_OVER_SMOOTH = "over_smooth"              # 过度平滑
RE_AS_REPORTED = "as_reported"              # 如实报告

#: 一票否决四分支标签（70 [S4] C6）。
VETO_VETO = "veto"                          # 物理幻觉失效
VETO_NOISE = "noise_fluctuation"            # 噪声波动（统计不显著）
VETO_PARTIAL = "partial_failure"            # 部分失效（混合增益）
VETO_LOCAL = "local_failure"                # 局部失效

#: 过冲 / 平滑分类标签（70 [S4] C8）。
OVER_OVERSHOOT = "overshoot"                # 过冲型（I_peak 高估）
OVER_SMOOTH = "smooth"                      # 平滑型（I_peak 低估）
OVER_EXACT = "exact"


# ---------------------------------------------------------------------------
# 归一化（70 [S2]）
# ---------------------------------------------------------------------------
def normalize_density(img: np.ndarray) -> np.ndarray:
    """总强度归一化到 1（视为二维概率密度，70 [S2] C1）。"""
    img = np.asarray(img, dtype=np.float64)
    total = img.sum()
    if total <= 0.0:
        raise ValueError("图像总强度非正，无法归一化（70 [S2]）")
    return img / total


# ---------------------------------------------------------------------------
# 图像级指标（70 [S3]）
# ---------------------------------------------------------------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 1.0) -> float:
    """PSNR = 10·log10(MAX²/MSE)，MAX = 1（70 [S3]，批次二十 Q8）。"""
    err = mse(a, b)
    if err <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_value**2 / err))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """SSIM：skimage，窗口 7×7、gaussian_weights=True（70 [S3] 实现约定）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            structural_similarity(a, b, data_range=data_range, gaussian_weights=True, win_size=7)
        )


# ---------------------------------------------------------------------------
# 物理级指标（70 [S4]）
# ---------------------------------------------------------------------------
def physics_quantities(img: np.ndarray) -> dict[str, float | np.ndarray]:
    """由归一化概率密度计算物理量（与 20 [S4] 标签口径一致）。

    坐标网格为像素中心（``z_i = −1 + (i+0.5)·Δ``），第 0 轴为 ``z``、
    第 1 轴为 ``δ``（20 [S3]）；公式见 70 [S4]（σ_z、σ_δ、h_eff=C_zδ/σ_z²、
    ε_z=√(σ_z²σ_δ²−C_zδ²)、I_peak=max I(z)）。
    """
    img = np.asarray(img, dtype=np.float64)
    coords, _ = pixel_center_coordinates(IMAGE_SIZE)
    Z, D = np.meshgrid(coords, coords, indexing="ij")
    q = img.sum()
    mu_z = float(np.sum(img * Z) / q)
    mu_delta = float(np.sum(img * D) / q)
    var_z = float(np.sum(img * (Z - mu_z) ** 2) / q)
    var_delta = float(np.sum(img * (D - mu_delta) ** 2) / q)
    cov = float(np.sum(img * (Z - mu_z) * (D - mu_delta)) / q)
    return {
        "sigma_z": float(np.sqrt(var_z)),
        "sigma_delta": float(np.sqrt(var_delta)),
        "h_eff": cov / var_z if var_z > 0 else 0.0,
        "eps_z": float(np.sqrt(max(var_z * var_delta - cov**2, 0.0))),
        "I_peak": float(img.sum(axis=1).max()),
        "I_z": img.sum(axis=1),
        "S_delta": img.sum(axis=0),
    }


def relative_error(pred: float, truth: float, floor: float = 1e-6) -> float:
    """相对误差 ``|pred − truth| / |truth|``（70 [S4] C9）。

    ``|truth| < floor`` 时改用绝对误差口径（分母取 ``floor``），避免
    ``h_eff`` 等量过零时相对误差发散；量纲定义保持不变。
    """
    denom = abs(float(truth)) if abs(float(truth)) >= floor else floor
    return abs(float(pred) - float(truth)) / denom


def signed_relative_error(pred: float, truth: float, floor: float = 1e-6) -> float:
    """带符号相对误差 ``(pred − truth) / |truth|``（过冲/平滑分类用）。"""
    denom = abs(float(truth)) if abs(float(truth)) >= floor else floor
    return (float(pred) - float(truth)) / denom


def profile_l1(pred_profile: np.ndarray, truth_profile: np.ndarray) -> float:
    """剖面 L1 误差 ``‖I_Ĥ − I_H‖₁``（70 [S4] 表：电流/能谱剖面误差）。"""
    return float(np.abs(np.asarray(pred_profile) - np.asarray(truth_profile)).sum())


def hallucination_flag(
    psnr_x: float,
    psnr_a: float,
    e_eps_z_x: float,
    e_eps_z_a: float,
    e_ipeak_x: float,
    e_ipeak_a: float,
    tau: float = TAU,
) -> int:
    """样本级幻觉标志 F_i（70 [S4]）。

    ``F_i = 1[ PSNR_X > PSNR_A ∧ (e_εz(X) − e_εz(A) > τ ∨ e_Ipeak(X) − e_Ipeak(A) > τ) ]``，
    只对先验方案（B/C）计算，SHALL NOT 应用于方案 A（70 [S4] C4）。
    """
    if psnr_x <= psnr_a:
        return 0
    if (e_eps_z_x - e_eps_z_a > tau) or (e_ipeak_x - e_ipeak_a > tau):
        return 1
    return 0


def veto_verdict(
    p_f: float,
    ci_lower_eps_z: float,
    ci_lower_ipeak: float,
    gain_eps_z: float,
    gain_ipeak: float,
    trigger_rate: float = TRIGGER_RATE,
) -> str:
    """方案级一票否决四分支判定（70 [S4] C6 操作化）。

    - ``P(F) ≤ 20%`` → ``local_failure``（局部失效，须结合参数空间解读）；
    - ``P(F) > 20%`` 且两个物理量的配对差 CI 下界均 ≤ 0（统计不显著）
      → ``noise_fluctuation``（噪声波动，SHALL NOT 触发方案级否决）；
    - ``P(F) > 20%`` 且至少一个 CI 下界 > 0（恶化统计显著）：
      - 净增益双非正（``G_εz ≤ 0`` 且 ``G_Ipeak ≤ 0``）→ ``veto``
        （物理幻觉失效）；
      - 净增益非双负（混合增益）→ ``partial_failure``（部分失效，
        按物理量报告通过/失败向量）。
    """
    if p_f <= trigger_rate:
        return VETO_LOCAL
    significant = ci_lower_eps_z > 0 or ci_lower_ipeak > 0
    if not significant:
        return VETO_NOISE
    if gain_eps_z <= 0 and gain_ipeak <= 0:
        return VETO_VETO
    return VETO_PARTIAL


def overshoot_smooth_class(signed_e_ipeak: float) -> str:
    """过冲 / 平滑分类（70 [S4] C8）：按 I_peak 带符号相对误差方向。"""
    if signed_e_ipeak > 0:
        return OVER_OVERSHOOT
    if signed_e_ipeak < 0:
        return OVER_SMOOTH
    return OVER_EXACT


# ---------------------------------------------------------------------------
# 精细结构级指标（70 [S5]）
# ---------------------------------------------------------------------------
def dog_sigma_outer(f_c: float = F_C) -> float:
    """DoG 外核尺度：高斯传递函数在 f_c 处衰减至 50%（70 [S5.1] C6）。

    由 ``exp(−2π²σ²f_c²) = 0.5`` 唯一解出 ``σ_outer = √(ln2/(2π²f_c²))``；
    内核 ``σ_inner = σ_outer/2``（一倍频程，配置项 ``sigma_inner_factor``）。
    """
    return float(np.sqrt(np.log(2.0) / (2.0 * np.pi**2 * f_c**2)))


def dog_high_pass(img: np.ndarray, sigma_outer: float) -> np.ndarray:
    """DoG 类高通：``H_high = H − GaussianBlur(H)``（70 [S5.1]）。"""
    blur = gaussian_filter(img, sigma=float(sigma_outer), mode="nearest")
    return np.asarray(img, dtype=np.float64) - blur


def e_high_doG(H_hat: np.ndarray, H: np.ndarray, sigma_outer: float) -> float:
    """高频残差误差 ``ε_high = ‖Ĥ_high − H_high‖₁``（70 [S5.1]）。"""
    return float(
        np.abs(dog_high_pass(H_hat, sigma_outer) - dog_high_pass(H, sigma_outer)).sum()
    )


def peak_mask(
    H: np.ndarray,
    window: int = 8,
    max_peaks: int = 3,
    height_frac: float = 0.5,
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """峰值掩膜 M_peak（70 [S5.2] C3）。

    对真值 ``H`` 以 ``window×window``（8×8）窗口做非极大抑制，取峰高
    ``≥ 全局最大值 50%`` 的前 3 个峰，掩膜为入选峰中心 8×8 窗口的并集；
    返回 ``(mask, n_peaks, positions)``（逐样本峰个数与位置作为诊断量，
    70 [S5.2]）。掩膜只依赖真值 ``H``，三方案共用。
    """
    H = np.asarray(H, dtype=np.float64)
    maxf = maximum_filter(H, size=window)
    is_peak = (H == maxf) & (H > 0.0)
    candidates = np.argwhere(is_peak)
    if len(candidates) == 0:
        return np.zeros(H.shape, dtype=bool), 0, []
    heights = H[is_peak]
    threshold = height_frac * H.max()
    order = np.argsort(-heights)
    kept: list[tuple[int, int]] = []
    half = window // 2
    for idx in order:
        if len(kept) >= max_peaks:
            break
        if heights[idx] < threshold:
            continue
        pos = (int(candidates[idx][0]), int(candidates[idx][1]))
        # 去重：同一 8×8 窗口内的相邻极大只算一个峰区域
        if any(max(abs(pos[0] - z0), abs(pos[1] - d0)) < window for z0, d0 in kept):
            continue
        kept.append(pos)
    mask = np.zeros_like(H, dtype=bool)
    for z0, d0 in kept:
        zs = slice(max(0, z0 - half), min(H.shape[0], z0 - half + window))
        ds = slice(max(0, d0 - half), min(H.shape[1], d0 - half + window))
        mask[zs, ds] = True
    return mask, len(kept), kept


def e_peak(H_hat: np.ndarray, H: np.ndarray, mask: np.ndarray) -> float:
    """局部峰值误差 ``ε_peak = ‖(Ĥ − H) ⊙ M_peak‖₁``（70 [S5.2] C2）。"""
    return float(np.abs(np.asarray(H_hat) - np.asarray(H))[mask].sum())


def high_pass_fft(img: np.ndarray, f_c: float = F_C) -> np.ndarray:
    """傅里叶高通分量：保留径向频率 ``f_c < f ≤ f_N`` 成分后逆变换（70 [S5.3]）。

    频带定义与 60 [S2] 谱损失一致（高频倍频程带并集 ``[f_c, f_N]``）；
    ``f > f_N`` 的 2D 网格角点不计入（60 [S2] 实现约定角点处理）。
    """
    img = np.asarray(img, dtype=np.float64)
    N = img.shape[-1]
    spectrum = np.fft.fft2(img)
    freqs = np.fft.fftfreq(N)
    kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.hypot(kx, ky)
    keep = (radius > f_c) & (radius <= F_NYQUIST)
    spectrum[~keep] = 0.0
    return np.ascontiguousarray(np.real(np.fft.ifft2(spectrum)))


def high_freq_energy_ratio(H_hat: np.ndarray, H: np.ndarray, f_c: float = F_C) -> float:
    """高频能量恢复率 ``R_E = ‖Ĥ_hp‖₁ / ‖H_hp‖₁``（70 [S5.3] C4）。"""
    hp_hat = high_pass_fft(H_hat, f_c)
    hp_ref = high_pass_fft(H, f_c)
    denom = np.abs(hp_ref).sum()
    if denom <= 0.0:
        return float("nan")
    return float(np.abs(hp_hat).sum() / denom)


def re_joint_class(
    e_high_x: float,
    e_high_baseline: float,
    r_e: float,
) -> str:
    """R_E 联合判读分类（70 [S5.3] 表）。

    - ``R_E < 0.5`` → 过度平滑；
    - ``ε_high ≤ 基线`` 且 ``R_E > 1.5`` → 疑似纹理幻觉；
    - ``ε_high ≤ 基线`` 且 ``0.8 ≤ R_E ≤ 1.2`` → 真实恢复；
    - 其余 → 如实报告（不强行归类）。
    """
    if r_e < 0.5:
        return RE_OVER_SMOOTH
    if e_high_x <= e_high_baseline and r_e > 1.5:
        return RE_TEXTURE_HALLUCINATION
    if e_high_x <= e_high_baseline and 0.8 <= r_e <= 1.2:
        return RE_TRUE_RECOVERY
    return RE_AS_REPORTED


def c_high_mask_from_hp(H_hp: np.ndarray, energy_frac: float = MASK_ENERGY_FRACTION) -> np.ndarray:
    """c_high 掩膜 M_{c_high}（70 [S7.1] C2）。

    将 ``|H_hp|`` 像素按幅值降序排列，取累计能量达到 ``‖H_hp‖₁`` 的
    ``energy_frac``（90%）的最小像素集；只依赖真值 ``H``，不依赖任何
    方案的输出，三方案使用同一掩膜。
    """
    H_hp = np.asarray(H_hp, dtype=np.float64)
    mag = np.abs(H_hp)
    flat = mag.ravel()
    total = flat.sum()
    if total <= 0.0:
        return np.zeros_like(H_hp, dtype=bool)
    order = np.argsort(-flat)
    cum = np.cumsum(flat[order])
    k = int(np.searchsorted(cum, energy_frac * total)) + 1
    mask = np.zeros(H_hp.shape, dtype=bool)  # 恒 C 连续，ravel 为视图
    mask.ravel()[order[: min(k, len(flat))]] = True
    return mask


def e_high_mask(H_hat: np.ndarray, H: np.ndarray, mask: np.ndarray, f_c: float = F_C) -> float:
    """预注册主指标 ε_high^mask（70 [S7.1] C1）：掩膜内高频误差一范数。"""
    diff = high_pass_fft(H_hat, f_c) - high_pass_fft(H, f_c)
    return float(np.abs(diff)[mask].sum())


# ---------------------------------------------------------------------------
# 单样本评估入口（70 [S2]–[S5] 全部指标）
# ---------------------------------------------------------------------------
def evaluate_sample(
    H_hat: np.ndarray,
    H: np.ndarray,
    m: dict,
    sigma_outer: float,
    e_high_baseline: float | None = None,
    mask_energy_frac: float = MASK_ENERGY_FRACTION,
) -> dict[str, float | str]:
    """单个样本的全部指标（指标前强制总强度归一化，70 [S2] C1）。

    参数
    ----
    H_hat: 网络输出（原始，未归一化；内部归一化）。
    H: 真值（总强度归一化或原始均可；内部归一化）。
    m: 真值物理标签（σ_z/σ_δ/h_eff/ε_z/I_peak/I_z/S_delta）。
    sigma_outer: DoG 外核尺度（70 [S5.1]）。
    e_high_baseline: ``L_up`` 退化基线的高频残差误差（R_E 联合判读参照，
        70 [S5.3]）；``None`` 时 R_E_class 按无法判读处理。
    """
    H_hat = np.asarray(H_hat, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    Q_hat = float(H_hat.sum())
    p = normalize_density(H_hat)
    H_norm = normalize_density(H)
    m = {k: (v if not hasattr(v, "numpy") else v.numpy()) for k, v in m.items()}

    pq = physics_quantities(p)
    i_z = np.asarray(m["I_z"], dtype=np.float64)
    s_delta = np.asarray(m["S_delta"], dtype=np.float64)

    e_high_val = e_high_doG(p, H_norm, sigma_outer)
    r_e = high_freq_energy_ratio(p, H_norm)
    hp_ref = high_pass_fft(H_norm)
    cmask = c_high_mask_from_hp(hp_ref, mask_energy_frac)
    pmask, n_peaks, positions = peak_mask(H_norm)

    if e_high_baseline is None:
        r_e_class: str = RE_AS_REPORTED
    else:
        r_e_class = re_joint_class(e_high_val, float(e_high_baseline), r_e)

    return {
        "psnr": psnr(p, H_norm),
        "mae": mae(p, H_norm),
        "mse": mse(p, H_norm),
        "ssim": ssim(p, H_norm),
        "e_eps_z": relative_error(pq["eps_z"], m["eps_z"]),
        "e_I_peak": relative_error(pq["I_peak"], m["I_peak"]),
        "e_sigma_z": relative_error(pq["sigma_z"], m["sigma_z"]),
        "e_sigma_delta": relative_error(pq["sigma_delta"], m["sigma_delta"]),
        "e_h_eff": relative_error(pq["h_eff"], m["h_eff"]),
        "e_high_doG": e_high_val,
        "R_E": r_e,
        "e_high_mask": e_high_mask(p, H_norm, cmask),
        "e_peak": e_peak(p, H_norm, pmask),
        "e_profile_I": profile_l1(pq["I_z"], i_z),
        "e_profile_S": profile_l1(pq["S_delta"], s_delta),
        "Q_hat": Q_hat,
        "R_E_class": r_e_class,
        "n_peaks": n_peaks,
        "peak_positions": positions,
        "e_I_peak_signed": signed_relative_error(pq["I_peak"], m["I_peak"]),
    }


# ---------------------------------------------------------------------------
# 统计判定协议（70 [S7]）
# ---------------------------------------------------------------------------
def bootstrap_ci(
    d: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int | None = None,
    alpha: float = CI_ALPHA,
) -> tuple[float, float]:
    """配对差序列的 bootstrap 95% CI（10,000 次配对差单元重采样，70 [S7.2]）。

    固定 ``seed`` 时结果可复现（05 [S6]：随机源由测试种子派生）。
    """
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    for k in range(n_boot):
        means[k] = d[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon(d: np.ndarray) -> float:
    """配对 Wilcoxon 符号秩检验 p 值（70 [S7.2]，双边）。"""
    d = np.asarray(d, dtype=np.float64)
    if d.size < 2 or np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def three_class(ci: tuple[float, float]) -> str:
    """三分类结论（70 [S7.3]）：显著正 / 等效 / 显著负。

    - CI 不含零且整体为正 → ``significant_positive``；
    - CI 含零 → ``equivalent``（证据不足以区分于 A，≠ 无增益）；
    - CI 不含零且整体为负 → ``significant_negative``。
    """
    lo, hi = ci
    if lo > 0:
        return "significant_positive"
    if hi < 0:
        return "significant_negative"
    return "equivalent"


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm 阶梯下降校正（70 [S7.1] C3），返回与输入同序的校正 p 值。"""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, p[idx] * (n - rank))
        running_max = max(running_max, value)
        adj[idx] = running_max
    return adj


def prior_gain_stats(
    d: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> dict[str, float | list[float] | str]:
    """单个指标的完整增益统计（70 [S7.2]/[S7.3]）。

    输入为配对差 ``d_i = M_A(i) − M_X(i)``（正值 = 先验降低误差，70 [S6]）。
    """
    d = np.asarray(d, dtype=np.float64)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "wilcoxon_p": paired_wilcoxon(d),
        "ci95": [lo, hi],
        "verdict": three_class((lo, hi)),
    }
