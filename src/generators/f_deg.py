"""低分辨率退化函数 ``f_deg(H; d) → (L, L_clean, d, m_L)``。

本模块按 30 规格 [S2] 的固定顺序生成低分辨率观测：先高斯模糊（模拟探测
系统有限分辨率）、再 ``r×r`` 块求和下采样（保总强度 ``ΣL = Σ(K*H)``）、
最后加性高斯噪声并做非负截断 ``L = max(0, L_clean + n)``。先模糊再下采样
防止高频结构在下采样中产生混叠（30 [S2] C1）。本模块不包含高分辨率生成、
先验生成、网络结构与训练损失逻辑（30 [S1] C2）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

#: 第一版默认下采样倍数（30 [S12] C1）。
DEFAULT_DOWNSAMPLE = 4

#: 精细结构「不可清楚分辨」的高通频带截止频率（归一化周期/像素，30 [S6] C8）。
HIGH_PASS_CUTOFF = 1.0 / 8.0


def f_deg(
    H: np.ndarray,
    sigma_K: float,
    sigma_n: float,
    r: int = DEFAULT_DOWNSAMPLE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """从高分辨率真值 ``H`` 生成低分辨率观测四元组 ``(L, L_clean, d, m_L)``。

    参数
    ----
    H: 高分辨率真值图像（默认 256×256、``H ≥ 0``），本函数不修改 ``H``。
    sigma_K: 各向同性高斯模糊核宽度，以 H 像素为单位（退化等级 D1/D2 由
        调用方按 30 [S7] 传入对应数值）。
    sigma_n: 加性高斯噪声标准差；``0`` 表示无噪声。
    r: 下采样倍数，默认 4（30 [S12]）。
    seed: 噪声随机种子，经 ``SeedSequence`` 派生；相同 ``H``、``d`` 与
        种子下输出逐位可复现（30 [S3] C4）。

    返回
    ----
    ``L``: 含噪低分辨率观测（非负）；``L_clean``: 仅模糊加下采样、无噪声的
    版本（30 [S3] C2）；``d``: 退化参数 ``{r, σ_K, ρ_K, noise model, σ_n}``；
    ``m_L``: 退化元数据（30 [S9] C1）。
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H 必须为方阵图像")
    n_h = H.shape[0]
    if n_h % r != 0:
        raise ValueError(f"高分辨率尺寸 {n_h} 不能被下采样倍数 {r} 整除")
    if sigma_K < 0 or sigma_n < 0:
        raise ValueError("sigma_K 与 sigma_n 必须非负")
    n_l = n_h // r

    blurred = gaussian_filter(H, sigma=sigma_K, mode="nearest")
    L_clean = blurred.reshape(n_l, r, n_l, r).sum(axis=(1, 3))

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    noise = rng.normal(0.0, sigma_n, L_clean.shape) if sigma_n > 0 else np.zeros_like(
        L_clean
    )
    L = np.maximum(0.0, L_clean + noise)

    d: dict[str, Any] = {
        "r": int(r),
        "sigma_K": float(sigma_K),
        "rho_K": 1.0,
        "noise_model": "additive_gaussian",
        "sigma_n": float(sigma_n),
    }

    snr = float(L_clean.mean() / sigma_n) if sigma_n > 0 else float("inf")
    m_L: dict[str, Any] = {
        "r": int(r),
        "N_H": int(n_h),
        "N_L": int(n_l),
        "sigma_K": float(sigma_K),
        "rho_K": 1.0,
        "noise_model": "additive_gaussian",
        "sigma_n": float(sigma_n),
        "SNR": snr,
        "seed": int(seed),
        "physical_range": (-1.0, 1.0),
        "pixel_scale_relation": "delta_L = r * delta_H",
        "degradation_order": "blur -> downsample -> noise",
        "downsample": "r x r block sum, total intensity preserved",
        "truncation": "L = max(0, L_clean + n)",
    }
    return L, L_clean, d, m_L


def sigma_K_for_level(level: str, w_fine_median_px: float) -> float:
    """按 30 [S7]/[S12] 的标定规则返回以 H 像素为单位的 ``σ_K``。

    D1 取 ``1×w_fine`` 批量中位数（弱退化、标定对比档），D2 取
    ``2×w_fine`` 批量中位数（主实验默认档），EXP-03 取 ``2×D2`` 标定值；
    EXP-04 的 ``σ_K`` 与 D2 相同，其差异在 ``σ_n``，故同样返回 D2 值。
    """
    multipliers = {
        "D1": 1.0,
        "D2": 2.0,
        "EXP-03": 4.0,
        "EXP-04": 2.0,
    }
    if level not in multipliers:
        raise ValueError(f"未知退化等级：{level}（支持 D1 / D2 / EXP-03 / EXP-04）")
    return multipliers[level] * float(w_fine_median_px)


def _highpass_power(image: np.ndarray, f_c: float) -> float:
    """图像高通分量（径向频率 ``f > f_c``）的 ``ℓ²`` 范数平方。

    使用正交归一化 FFT，由 Parseval 定理该值等于空域高通分量的能量平方。
    """
    n = image.shape[0]
    kx, ky = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    f = np.hypot(kx, ky)
    F = np.fft.fft2(image, norm="ortho")
    return float(np.sum(np.abs(F[f > f_c]) ** 2))


def snr_hf(L: np.ndarray, L_clean: np.ndarray, f_c: float = HIGH_PASS_CUTOFF) -> float:
    """高频带信噪比 ``SNR_hf = ‖(L_clean)_hp‖₂ / ‖(n_eff)_hp‖₂``。

    高通频带为傅里叶径向频率 ``f > 1/8``（归一化周期/像素）；等效噪声图像
    取 ``n_eff = L − L_clean``（含非负截断效应）。逐样本计算，批量判定时
    由调用方取中位数（30 [S6] C8）。
    """
    signal_power = _highpass_power(np.asarray(L_clean, dtype=np.float64), f_c)
    noise_power = _highpass_power(
        np.asarray(L, dtype=np.float64) - np.asarray(L_clean, dtype=np.float64), f_c
    )
    if noise_power <= 0.0:
        return float("inf")
    return float(np.sqrt(signal_power / noise_power))
