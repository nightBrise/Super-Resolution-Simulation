"""通用工具：HDF5 数据访问、配置解析与 checkpoint 读写。

里程碑 M3 新增（60 [S15] 15.1 文件树 ``src/utils/`` 通用工具）：
- ``h5data``：训练/评估共享的 HDF5 数据集读取、方案 C 参数先验构造与
  训练集标准化统计（60 [S4][S5]）；
- ``config_utils``：config.yaml 加载（复用 60 [S14] 契约）、config hash、
  实验目录命名（60 [S15] 15.2）与设备/精度解析；
- ``checkpoint``：checkpoint 读写（60 [S12] C2）与 seeds.json 落盘
  （60 [S14] C8）。

三个方案（A/B/C）与训练/评估/推理共用同一份数据访问与契约代码，保证
三方案公平（同一份数据、同一统计口径）。
"""

from src.utils.h5data import (
    C_PRIOR_KEYS,
    POS_LOG_KEYS,
    H5Dataset,
    compute_c_prior_stats,
    preprocess_c_prior,
)

__all__ = [
    "C_PRIOR_KEYS",
    "POS_LOG_KEYS",
    "H5Dataset",
    "compute_c_prior_stats",
    "preprocess_c_prior",
]
