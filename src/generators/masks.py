"""有效域掩膜 W1–W8 与精细结构宽度计算。

本模块实现 20 规格 [S9] 联合约束定义的白名单掩膜：候选参数组合必须全部
满足 W1–W8 才可入选进入数据集。每个检查函数接受参数映射（标量值或
numpy 数组），返回布尔值或布尔数组，便于逐样本判定与向量化批量筛选。
精细结构宽度 ``w_fine = b_0 · S``（其中 ``S = |2βσ_δ + 3γσ_δ²|``、
``σ_δ = √((a_1 σ_z)² + b_0²)``）按 20 [S9] 联合约束 2 的参数级定义计算。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: 高分辨率网格单边像素数（20 [S3] C7 默认分辨率 256×256）。
GRID = 256

#: 一个高分辨率像素的归一化坐标宽度，Δ = 2/256 ≈ 0.0078（20 [S9]）。
DELTA_PX = 2.0 / GRID

#: 掩膜名称到检查函数的登记顺序，`apply_masks` 与统计报表按此顺序输出。
MASK_NAMES: tuple[str, ...] = (
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8",
)

#: 掩膜定义版本号，随数据集元数据记录（20 [S9] C8）。
MASK_VERSION = "20-v1.0-W1W8"


def check_w1(c: Mapping[str, Any]) -> Any:
    """掩膜 W1：``σ_eff(z) = σ_z(1+ηz)`` 在 ``|z| ≤ 3σ_z`` 上恒正。

    条件式为 ``3|η|σ_z < 1``（20 [S9] 联合约束 3）。
    """
    return 3.0 * np.abs(c["eta"]) * c["sigma_z"] < 1.0


def check_w2(c: Mapping[str, Any]) -> Any:
    """掩膜 W2：``σ_δ(z)`` 在 ``|z| ≤ 3σ_z`` 上恒正。

    条件式为 ``1 + 9b_1 > 0``（20 [S9] 联合约束 3）。
    """
    return 1.0 + 9.0 * c["b1"] > 0.0


def check_w3(c: Mapping[str, Any]) -> Any:
    """掩膜 W3：压缩剪切数值稳定，``|α| ≤ 3``（20 [S9] 联合约束 3）。"""
    return np.abs(c["alpha"]) <= 3.0


def check_w4(c: Mapping[str, Any]) -> Any:
    """掩膜 W4：关联能散在窗内，``0.15 ≤ |a_1|σ_z ≤ 0.35``。

    锚点值 0.25 的 0.6–1.4 倍（20 [S9] 联合约束 3）。
    """
    x = np.abs(c["a1"]) * c["sigma_z"]
    return (x >= 0.15) & (x <= 0.35)


def check_w5(c: Mapping[str, Any]) -> Any:
    """掩膜 W5：δ 向窗口覆盖，束流 δ 向延展加局部能散不超出 ``[-1, 1]`` 窗口。

    条件式为 ``2.5|a₁|σ_z + 6.25|a₂|σ_z² + 15.625|a₃|σ_z³ + 2.5b₀ ≤ 1``
    （20 [S9] 联合约束 3）。
    """
    sz = c["sigma_z"]
    value = (
        2.5 * np.abs(c["a1"]) * sz
        + 6.25 * np.abs(c["a2"]) * sz**2
        + 15.625 * np.abs(c["a3"]) * sz**3
        + 2.5 * c["b0"]
    )
    return value <= 1.0


def check_w6(c: Mapping[str, Any]) -> Any:
    """掩膜 W6：压缩后束核在窗内，``|C|σ_z ≤ 0.35``（20 [S9] 联合约束 3）。"""
    return np.abs(c["C"]) * c["sigma_z"] <= 0.35


def _projected_energy_spread(c: Mapping[str, Any]) -> Any:
    """样本投影相对能散的参数级估计 ``σ_δ = √((a_1 σ_z)² + b_0²)``。

    渲染后以 20 [S7] 标签矩为准；该估计仅用于掩膜 W7 与 W8 的判定
    （20 [S9] 联合约束 2）。
    """
    return np.sqrt((c["a1"] * c["sigma_z"]) ** 2 + c["b0"] ** 2)


def fine_structure_width(c: Mapping[str, Any]) -> Any:
    """精细结构宽度 ``w_fine = b_0 · S``，归一化坐标单位。

    非线性剪切强度 ``S = |2βσ_δ + 3γσ_δ²|`` 为折叠映射对 δ 的导数展宽，
    ``σ_δ`` 为参数级投影能散估计（20 [S9] 联合约束 2）。
    """
    s_delta = _projected_energy_spread(c)
    shear = np.abs(2.0 * c["beta"] * s_delta + 3.0 * c["gamma"] * s_delta**2)
    return c["b0"] * shear


def check_w7(c: Mapping[str, Any]) -> Any:
    """掩膜 W7：折叠在窗内。

    条件式为 ``|β|(2.5σ_δ)² + |γ|(2.5σ_δ)³ ≤ 0.9``，其中
    ``σ_δ = √((a_1 σ_z)² + b_0²)``（20 [S9] 联合约束 3）。
    """
    s_delta = _projected_energy_spread(c)
    value = np.abs(c["beta"]) * (2.5 * s_delta) ** 2 + np.abs(c["gamma"]) * (
        2.5 * s_delta
    ) ** 3
    return value <= 0.9


def check_w8(c: Mapping[str, Any], sigma_K: float) -> Any:
    """掩膜 W8：精细结构窗口 ``2.5px < w_fine/Δ_px ≤ 0.8σ_K``。

    ``w_fine`` 以归一化坐标计算后换算为高分辨率像素数，``σ_K`` 以 H 像素
    为单位（20 [S9] 联合约束 2）。等价归一化形式为
    ``0.0195 < w_fine ≤ 0.8σ_K·Δ_px``。
    """
    w_fine_px = fine_structure_width(c) / DELTA_PX
    return (w_fine_px > 2.5) & (w_fine_px <= 0.8 * sigma_K)


#: 掩膜名称到不依赖 σ_K 的检查函数的映射；W8 因依赖 σ_K 单独处理。
_MASK_FUNCTIONS = {
    "W1": check_w1,
    "W2": check_w2,
    "W3": check_w3,
    "W4": check_w4,
    "W5": check_w5,
    "W6": check_w6,
    "W7": check_w7,
}


def apply_masks(c: Mapping[str, Any], sigma_K: float) -> dict[str, Any]:
    """对参数组合逐条评估 W1–W8，返回掩膜名称到布尔结果的映射。

    参数映射的值可以是标量（逐样本判定）或等长 numpy 数组（批量判定）；
    ``sigma_K`` 为以 H 像素为单位的模糊核宽度（30 [S12] 标定规则确定的数值）。
    """
    results = {name: fn(c) for name, fn in _MASK_FUNCTIONS.items()}
    results["W8"] = check_w8(c, sigma_K)
    return results
