"""Level 1 物理生成函数 ``f_beam(c) → (H, m, c)``。

本模块按 20 规格 [S4] 定义的五项要素生成高分辨率纵向相空间真值：
非高斯电流剖面、头尾不对称（``η``）、局部厚度变化（``b_1``）、三阶中心线
（``a_1, a_2, a_3``）与二阶/三阶压缩折叠映射（``α, β, γ``）。渲染流程为
「基础密度 ``ρ_0`` → 压缩折叠映射 → 光滑渲染 → 总强度归一化」，不直接绘制
目标形状（20 [S4] C6）。折叠映射 ``z_f = z + αδ + βδ² + γδ³``、
``δ_f = δ`` 的 Jacobian 恒为 1（保面积），反向采样无需密度修正。

本模块不包含低分辨率退化逻辑（属 30 规格）与先验生成逻辑（属 40 规格）；
坐标约定为 ``z, δ ∈ [-1, 1]``、像素中心采样、图像第 0 轴为 ``z``、
第 1 轴为 ``δ``（20 [S3]）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from src.generators.masks import fine_structure_width, DELTA_PX, GRID

#: 复杂度等级：第一版基线为 Level 1，不存在 Level 0（20 [S8] C1）。
LEVEL = 1

#: 参数分组，与 00/20 [S5] C1、40 [S4] C1 一致；A 固定为 1（20 [S9]）。
C_LOW_KEYS: tuple[str, ...] = ("A", "sigma_z", "n", "eta", "b0", "a1", "alpha")
C_MID_KEYS: tuple[str, ...] = ("a2", "beta")
C_HIGH_KEYS: tuple[str, ...] = ("a3", "gamma", "b1")


def pixel_center_coordinates(grid: int) -> tuple[np.ndarray, float]:
    """返回 ``grid`` 个像素中心的一维归一化坐标与像素宽度 ``Δ = 2/grid``。

    像素中心定义为 ``z_i = −1 + (i + 0.5)·Δ``，覆盖 ``[-1, 1]``
    （20 [S3] C1 与任务书坐标约定）。
    """
    delta = 2.0 / grid
    coords = -1.0 + (np.arange(grid) + 0.5) * delta
    return coords, delta


def current_profile(z: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    """电流剖面 ``I(z) = A·exp(−|z/σ_eff(z)|^{2n})``。

    有效宽度 ``σ_eff(z) = σ_z(1 + ηz)``；``n = 1`` 且 ``η = 0`` 时退化为
    对称高斯剖面，``n > 1`` 时呈平顶或陡边形状（20 [S4] C2/C3）。
    """
    sigma_eff = c["sigma_z"] * (1.0 + c["eta"] * z)
    return c["A"] * np.exp(-np.abs(z / sigma_eff) ** (2.0 * c["n"]))


def local_energy_spread(z: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    """局部能量厚度 ``σ_δ(z) = b_0[1 + b_1(z/σ_z)²]``。

    ``b_1 = 0`` 时为常数，``b_1 ≠ 0`` 时随 ``z`` 变化（20 [S4] C5）；
    掩膜 W2 保证该式在 ``|z| ≤ 3σ_z`` 上恒正。
    """
    return c["b0"] * (1.0 + c["b1"] * (z / c["sigma_z"]) ** 2)


def centerline(z: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    """中心线 ``C(z) = a_1 z + a_2 z² + a_3 z³``（三阶多项式 chirp）。"""
    return c["a1"] * z + c["a2"] * z**2 + c["a3"] * z**3


def base_density(grid: int, c: Mapping[str, Any]) -> np.ndarray:
    """基础密度 ``ρ_0(z, δ) = I(z)·exp(−(δ − C(z))²/(2σ_δ(z)²))``。

    在 ``grid × grid`` 的像素中心网格上求值，返回二维数组
    （第 0 轴为 ``z``，第 1 轴为 ``δ``）。
    """
    z, _ = pixel_center_coordinates(grid)
    Z, D = np.meshgrid(z, z, indexing="ij")
    iz = current_profile(Z, c)
    sigma_delta = local_energy_spread(Z, c)
    center = centerline(Z, c)
    return iz * np.exp(-((D - center) ** 2) / (2.0 * sigma_delta**2))


def folding_shift(delta: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    """压缩折叠映射的纵向位移项 ``αδ + βδ² + γδ³``。

    完整映射为 ``z_f = z + αδ + βδ² + γδ³``、``δ_f = δ``（20 [S4] C6）。
    """
    return c["alpha"] * delta + c["beta"] * delta**2 + c["gamma"] * delta**3


def render_level1_density(
    c: Mapping[str, Any],
    grid: int = GRID,
    sigma_smooth: float | None = None,
) -> np.ndarray:
    """按 Level 1 模型渲染总强度归一化的光滑密度图像。

    参数
    ----
    c: 内容参数映射，键见 ``PARAMETER_KEYS``（A 固定为 1）。
    grid: 渲染网格边长，默认 256（20 [S3] C7）。
    sigma_smooth: 高斯平滑核宽度的像素数；``None`` 时取单样本回退值
        ``0.125 × w_fine`` 像素（与 20 [S3] C4 逐样本口径同，2026-08-26
        P0 修订：原 0.5× 使精细结构频率处能量保留仅约 5.2e-5，见 99
        OQ-20-03）。

    返回
    ----
    非负二维数组，逐元素和为 1。渲染使用 ``map_coordinates``
    （order=1、mode='nearest'）把 ``ρ_0`` 反向采样到目标网格：目标 ``(Z, D)``
    处的源坐标为 ``z = Z − (αD + βD² + γD³)``、``δ = D``，像素索引为
    ``(坐标 + 1)/Δ − 0.5``；随后以 ``gaussian_filter`` 平滑（像素单位）。
    """
    coords, delta = pixel_center_coordinates(grid)
    Z, D = np.meshgrid(coords, coords, indexing="ij")

    rho0 = base_density(grid, c)

    z_source = Z - folding_shift(D, c)
    index_z = (z_source + 1.0) / delta - 0.5
    index_d = (D + 1.0) / delta - 0.5
    H = map_coordinates(rho0, np.asarray([index_z, index_d]), order=1, mode="nearest")

    if sigma_smooth is None:
        sigma_smooth = 0.125 * fine_structure_width(c) / DELTA_PX
    H = gaussian_filter(H, sigma=sigma_smooth, mode="nearest")

    total = H.sum()
    if total <= 0.0:
        raise ValueError("渲染结果总强度非正，参数组合可能使束流完全移出窗口")
    return H / total


def compression_state(compression_factor: float) -> str:
    """由压缩因子 ``C = 1 + a_1 α`` 派生压缩状态标签（20 [S7] C3）。

    欠压缩 ``C > 0.1``、最佳压缩 ``|C| ≤ 0.1``、过压缩 ``C < −0.1``；
    区间划分与 20 [S9] 联合约束 1 的三态定义一致。
    """
    if compression_factor > 0.1:
        return "under"
    if compression_factor < -0.1:
        return "over"
    return "optimal"


def f_beam(
    c: Mapping[str, Any],
    grid: int = GRID,
    sigma_smooth: float | None = None,
    q_total: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """生成高分辨率真值 ``H``、物理标签 ``m`` 与参数记录，返回 ``(H, m, c)``。

    ``f_beam`` 为确定性函数：给定相同 ``c`` 与相同配置（``grid``、
    ``sigma_smooth``、``q_total``），输出逐位可复现（20 [S2] C3）。
    本函数不包含低分辨率退化逻辑（20 [S1] C2）也不生成先验（20 [S1] C3）。

    参数
    ----
    c: 内容参数映射，至少包含 ``sigma_z, n, eta, b0, a1, alpha, a2, a3,
        beta, gamma, b1``；``A`` 缺省取 1。
    grid: 渲染网格边长，默认 256（20 [S3] C7）。
    sigma_smooth: 平滑核宽度像素数；``None`` 时按
        ``render_level1_density`` 的单样本回退规则取值。
    q_total: 总强度 ``Q``，默认 1（20 [S3] C3）。
    """
    params = dict(c)
    params.setdefault("A", 1.0)

    H = render_level1_density(params, grid=grid, sigma_smooth=sigma_smooth)
    H = H * q_total

    coords, delta = pixel_center_coordinates(grid)
    Z, D = np.meshgrid(coords, coords, indexing="ij")

    q = float(H.sum())
    mu_z = float(np.sum(H * Z) / q)
    mu_delta = float(np.sum(H * D) / q)
    var_z = float(np.sum(H * (Z - mu_z) ** 2) / q)
    var_delta = float(np.sum(H * (D - mu_delta) ** 2) / q)
    cov_z_delta = float(np.sum(H * (Z - mu_z) * (D - mu_delta)) / q)

    i_z = H.sum(axis=1)
    s_delta = H.sum(axis=0)

    compression_factor = float(1.0 + params["a1"] * params["alpha"])
    if "C" in params:
        compression_factor = float(params["C"])

    m: dict[str, Any] = {
        "Q": q,
        "mu_z": mu_z,
        "mu_delta": mu_delta,
        "sigma_z": float(np.sqrt(var_z)),
        "sigma_delta": float(np.sqrt(var_delta)),
        "C_zdelta": cov_z_delta,
        "h_eff": cov_z_delta / var_z if var_z > 0 else 0.0,
        "eps_z": float(np.sqrt(max(var_z * var_delta - cov_z_delta**2, 0.0))),
        "I_peak": float(i_z.max()),
        "I_z": i_z,
        "S_delta": s_delta,
        "Level": LEVEL,
        "c": {
            "c_low": {key: float(params[key]) for key in C_LOW_KEYS},
            "c_mid": {key: float(params[key]) for key in C_MID_KEYS},
            "c_high": {key: float(params[key]) for key in C_HIGH_KEYS},
        },
        "compression_factor": compression_factor,
        "compression_state": compression_state(compression_factor),
        "render": {
            "grid": int(grid),
            "delta": delta,
            "coordinate_range": (-1.0, 1.0),
            "axis_order": "z-rows, delta-columns",
            "normalization": "sum-to-Q",
            "pixel_definition": "pixel-center sampling + Gaussian smoothing",
            "sigma_smooth_px": (
                float(sigma_smooth)
                if sigma_smooth is not None
                else float(0.125 * fine_structure_width(params) / DELTA_PX)
            ),
        },
    }
    return H, m, params
