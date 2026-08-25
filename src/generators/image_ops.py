"""图像公共算子：4 倍双线性上采样与总强度归一化。

本模块实现网络输入构造所需的两个固定算子：``L_up`` 的 4 倍双线性上采样
（50 [S8]）与总强度归一化（50 [S13]、60 [S3]）。上采样为固定插值算子，
三个方案一致，不使用可学习上采样（50 [S8] C3）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

#: 低分辨率到高分辨率的上采样倍数（256 = 4 × 64，30 [S12]）。
UPSAMPLE_FACTOR = 4


def upsample_4x_bilinear(L: np.ndarray) -> np.ndarray:
    """把 ``64×64`` 低分辨率观测双线性上采样为 ``256×256`` 的 ``L_up``。

    使用 ``scipy.ndimage.zoom``（order=1、grid_mode=True）实现：输出像素
    ``o`` 对应输入坐标 ``(o+0.5)/4 − 0.5``，即输出像素中心与输入物理坐标
    网格严格对齐（``z, δ ∈ [-1, 1]`` 同范围，50 [S8] C1）。双线性插值不产生
    负值与过冲；对远离边界的分布总强度严格守恒（``ΣL_up = 16·ΣL``），
    归一化在插值之后由 ``normalize_intensity`` 执行（50 [S13]）。
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("L 必须为二维图像")
    return zoom(L, UPSAMPLE_FACTOR, order=1, mode="nearest", grid_mode=True)


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """总强度归一化：返回 ``img / Σimg``，使逐元素和为 1。

    归一化约定与 60 [S3] 一致（``ΣH = ΣL_up = ΣP₂ = 1``）；总强度非正时
    抛出 ``ValueError``，因为该输入不构成可归一化的密度图像。
    """
    img = np.asarray(img, dtype=np.float64)
    total = img.sum()
    if total <= 0.0:
        raise ValueError("图像总强度非正，无法做总强度归一化")
    return img / total
