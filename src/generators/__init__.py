"""数据生成模块集合：物理真值、低分辨率退化、物理先验与参数采样。

本包当前包含里程碑 M1 的实现：`f_beam`（20 规格高分辨率真值生成）、
`f_deg`（30 规格低分辨率退化）、`f_prior`（40 规格先验生成）、
`sampling`（20 [S9] 参数采样）与 `masks`（20 [S9] W1–W8 掩膜检查），
以及 `image_ops` 中的上采样与总强度归一化公共算子（50 [S8][S13]）。

以下三个模块属于后续里程碑，本里程碑不实现，仅在此登记其归属
（60 [S15] C4b 规定交叉生成工具归属 `src/generators/`）：

- ``dataset_builder.py``：数据划分与 HDF5 落盘、manifest（里程碑 M2，60 [S8][S14]）。
- ``calibration.py``：EXP-01b/c/d 标定（σ_K / σ_n / σ_smooth，里程碑 M3）。
- ``probe.py``：G0 受控探针法（里程碑 M2，80 [S9] G0(b)）。
"""

from src.generators.f_beam import f_beam, render_level1_density
from src.generators.f_deg import f_deg, snr_hf
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.generators.masks import (
    apply_masks,
    check_w1,
    check_w2,
    check_w3,
    check_w4,
    check_w5,
    check_w6,
    check_w7,
    check_w8,
    fine_structure_width,
)
from src.generators.sampling import sample_parameters

__all__ = [
    "apply_masks",
    "check_w1",
    "check_w2",
    "check_w3",
    "check_w4",
    "check_w5",
    "check_w6",
    "check_w7",
    "check_w8",
    "f_beam",
    "f_deg",
    "f_prior",
    "fine_structure_width",
    "normalize_intensity",
    "render_level1_density",
    "sample_parameters",
    "snr_hf",
    "upsample_4x_bilinear",
]
