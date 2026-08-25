"""网络模块：残差 U-Net 主干与三方案组装（50 规格）。

里程碑 M3 新增：
- ``unet``：残差 U-Net 主干（50 [S7] 主干配置表）；
- ``schemes``：方案 A（无先验）/ B（图像先验）/ C（参数先验 + FiLM）
  组装（50 [S3]–[S5][S8]–[S12]）。
"""

from src.models.schemes import (
    SCHEME_CLASSES,
    SchemeA,
    SchemeB,
    SchemeC,
    SchemeModel,
    build_scheme_model,
    build_scheme_model_from_checkpoint,
)
from src.models.unet import UNetBackbone

__all__ = [
    "UNetBackbone",
    "SchemeA",
    "SchemeB",
    "SchemeC",
    "SchemeModel",
    "SCHEME_CLASSES",
    "build_scheme_model",
    "build_scheme_model_from_checkpoint",
]
