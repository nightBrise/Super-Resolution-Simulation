"""图像先验生成函数 ``f_prior(c, level) → (P, meta)``。

本模块按 40 规格 [S5][S6] 生成与真值同源但不等于真值的物理先验图像：
复用 20 规格的 Level 1 物理生成框架，去除高分辨率精细项（``c_high``）并
施加更强平滑（``σ_smooth,P > σ_smooth,H``）。默认等级 P2 保留
``c_low + c_mid``，去掉 ``a_3, γ, b_1``（40 [S6] C3）；先验生成不使用
``H``、不依赖低分辨率观测 ``L`` 的噪声实现（40 [S9] C1/C2）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.generators.f_beam import render_level1_density, C_LOW_KEYS, C_MID_KEYS
from src.generators.masks import fine_structure_width, DELTA_PX, GRID

#: 支持的图像先验等级；P0 表示无图像先验，不生成图像（40 [S6]）。
SUPPORTED_LEVELS: tuple[str, ...] = ("P1", "P2", "P3")

#: 先验记录参数中永不包含的高分辨率精细参数（40 [S4] C3）。
C_HIGH_KEYS: tuple[str, ...] = ("a3", "gamma", "b1")


def prior_parameters(c: Mapping[str, Any], level: str) -> dict[str, float]:
    """按先验等级构造生成参数，显式清零该等级不保留的参数。

    P1 只保留 ``c_low``（中心线 ``a_1 z``、映射 ``z + αδ``）；P2 保留
    ``c_low + c_mid``（中心线 ``a_1 z + a_2 z²``、映射 ``z + αδ + βδ²``、
    厚度恒为 ``b_0``）；P3 使用完整参数，仅用于上限分析（40 [S6]）。
    无论输入如何，``c_high`` 参数在 P1/P2 下恒被置零，保证先验对
    ``c_high`` 扰动逐位不变（40 [S9] C3）。
    """
    if level not in SUPPORTED_LEVELS:
        raise ValueError(
            f"不支持的先验等级 {level!r}；P0 表示无图像先验，"
            f"支持的图像先验等级为 {SUPPORTED_LEVELS}"
        )
    c_prior = dict(c)
    c_prior.setdefault("A", 1.0)
    if level in ("P1", "P2"):
        for key in C_HIGH_KEYS:
            c_prior[key] = 0.0
    if level == "P1":
        c_prior["a2"] = 0.0
        c_prior["beta"] = 0.0
    return c_prior


def f_prior(
    c: Mapping[str, Any],
    level: str = "P2",
    grid: int = GRID,
    sigma_smooth: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """生成先验图像 ``P`` 与元数据，返回 ``(P, meta)``。

    参数
    ----
    c: 与真值 ``H`` 相同的完整内容参数（同源，40 [S2] C1）；P1/P2 等级下
        ``c_high`` 分量被显式清零而不被使用。
    level: 先验等级，取值 ``P1`` / ``P2`` / ``P3``，默认 ``P2``
        （40 [S6] C1）。
    grid: 先验图像边长，与 ``H`` 同尺寸，默认 256（40 [S5] C2）。
    sigma_smooth: 先验平滑核宽度像素数，须满足 ``σ_smooth,P > σ_smooth,H``
        （40 [S5] C3）；``None`` 时取初始值 ``2 × σ_smooth,H``，其中
        ``σ_smooth,H = 0.5 × w_fine`` 像素（40 [S5] C5 初始候选中心）。

    返回
    ----
    ``P``: 非负、总强度归一化（``ΣP = 1``）的先验图像；``meta``: 记录
    ``c_prior``、prior level、prior type、smoothing、grid size 与
    normalization（40 [S10] C2）。
    """
    c_p = prior_parameters(c, level)

    sigma_smooth_h_ref = 0.5 * float(fine_structure_width(c_p) / DELTA_PX)
    if sigma_smooth is None:
        sigma_smooth = 2.0 * sigma_smooth_h_ref

    P = render_level1_density(c_p, grid=grid, sigma_smooth=sigma_smooth)

    record_keys = C_LOW_KEYS if level == "P1" else C_LOW_KEYS + C_MID_KEYS
    c_prior_record = {key: float(c_p[key]) for key in record_keys}
    c_prior_record.pop("A")  # 总强度归一化下 A 退化，不进入先验参数记录

    meta: dict[str, Any] = {
        "level": level,
        "type": "image",
        "oracle": level == "P3",
        "prior_kind": "oracle" if level == "P3" else "realistic",
        "c_prior": c_prior_record,
        "smoothing": float(sigma_smooth),
        "smoothing_H_reference": sigma_smooth_h_ref,
        "grid_size": int(grid),
        "normalization": "sum-to-1",
    }
    return P, meta
