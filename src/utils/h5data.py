"""HDF5 数据集访问与方案 C 参数先验构造（60 [S4][S5]、60 [S14]）。

数据集文件契约见 60 [S14]：每划分一个 ``data/<版本>/<split>.h5``，顶层
Dataset 为图像字段（``H``/``L``/``L_clean``/``L_up``/``P2``，float32，
总强度归一化），Group 为参数（``c_low``/``c_mid``/``c_high``）、物理标签
（``m``）、退化元数据（``m_L``）与掩膜（``masks``）。

本模块提供：
- ``H5Dataset``：按索引读取单个样本的完整字段字典（训练/评估/推理共用）；
- ``C_PRIOR_KEYS``：方案 C 参数先验字段序（60 [S3] C2：不含 ``A`` 与
  ``c_high``）；
- ``preprocess_c_prior`` / ``compute_c_prior_stats``：正参数取对数 +
  z-score（60 [S5]），统计量只从训练集计算（60 [S13] AC5，★ 防泄露）。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

#: 方案 C 参数先验字段序（60 [S3] C2）：``(c_low \ {A}) ∪ c_mid``。
C_PRIOR_KEYS: tuple[str, ...] = (
    "sigma_z",
    "n",
    "eta",
    "b0",
    "a1",
    "alpha",
    "a2",
    "beta",
)

#: 取对数后再 z-score 的正参数索引（σ_z、b0，60 [S5] C2）。
POS_LOG_KEYS: tuple[str, ...] = ("sigma_z", "b0")

#: 每个样本返回的全部字段键（训练取子集，评估取全集）。
_FIELD_KEYS = ("H", "L", "L_clean", "L_up", "P2", "sample_id", "seed_i")
_M_KEYS = (
    "sigma_z",
    "sigma_delta",
    "h_eff",
    "eps_z",
    "I_peak",
    "I_z",
    "S_delta",
)
_C_HIGH_KEYS = ("a3", "gamma", "b1")


def _as_tensor(arr) -> torch.Tensor:
    """numpy 数组/标量转 torch 张量（float32 保留原 dtype，其余按元素类型）。"""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return torch.from_numpy(np.asarray(arr))


def preprocess_c_prior(
    raw: np.ndarray,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """参数先验预处理：正参数取对数 → z-score（60 [S5]）。

    参数
    ----
    raw: 形状 ``(..., 8)`` 的原始参数（顺序见 ``C_PRIOR_KEYS``）。
    mu, sigma: 训练集统计量（形状 ``(8,)``）；``None`` 时用恒等标准化
        （μ=0、σ=1，仅作防御性默认；正式训练/评估 SHALL 由训练集统计量
        覆盖，见 60 [S5] C3）。
    """
    x = np.asarray(raw, dtype=np.float64)
    if x.shape[-1] != len(C_PRIOR_KEYS):
        raise ValueError(
            f"参数先验最后一维须为 {len(C_PRIOR_KEYS)}（{C_PRIOR_KEYS}），"
            f"实际 {x.shape[-1]}"
        )
    x = x.copy()
    for i, key in enumerate(C_PRIOR_KEYS):
        if key in POS_LOG_KEYS:
            x[..., i] = np.log(x[..., i])
    if mu is None:
        mu = np.zeros(len(C_PRIOR_KEYS))
    if sigma is None:
        sigma = np.ones(len(C_PRIOR_KEYS))
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (len(C_PRIOR_KEYS),):
        raise ValueError(f"σ 形状须为 ({len(C_PRIOR_KEYS)},)，实际 {sigma.shape}")
    return (x - mu) / sigma


def compute_c_prior_stats(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由训练集原始参数计算标准化统计量 ``(μ, σ)``（60 [S5] C1/C3）。

    只接受训练集样本；验证集与测试集必须复用返回的统计量，不得各自计算
    （60 [S13] AC5，★ 无测试集泄漏）。统计量在取对数后的值上计算。
    """
    x = preprocess_c_prior(raw)  # 恒等标准化下取对数
    return x.mean(axis=0), x.std(axis=0)


class H5Dataset(Dataset):
    """按索引读取 HDF5 数据集中单个样本的字段字典。

    样本字典键（训练只用子集，评估/推理用全集）：
    - 图像（``(1, 256, 256)`` 或 ``(1, 64, 64)`` 张量）：``H``、``L``、
      ``L_clean``、``L_up``、``P2``；
    - 参数先验：``c_prior_raw``（``(8,)``，顺序见 ``C_PRIOR_KEYS``，
      不含 ``A`` 与 ``c_high``）；
    - 评估字段：``c_high``（``(3,)``：a3/γ/b1）、``m``（dict：σ_z、
      σ_δ、h_eff、ε_z、I_peak、I_z、S_delta）、``sample_id``（str）、
      ``split``（str）。
    """

    def __init__(self, h5_path: str | Path, split: str) -> None:
        self.split = split
        self.h5 = h5py.File(str(h5_path), "r")
        n = len(self.h5["sample_id"])
        for key in ("H", "L", "L_up", "P2"):
            if key not in self.h5:
                raise KeyError(f"HDF5 缺少字段 {key}（60 [S14] 契约）")
        self.n = n

    def __len__(self) -> int:
        return self.n

    def _read(self, group: str, key: str, index: int) -> np.ndarray:
        return self.h5[group][key][index]

    def __getitem__(self, index: int) -> dict:
        g = self.h5
        sample: dict = {}
        for key in _FIELD_KEYS:
            if key == "sample_id":
                raw_id = g[key][index]  # HDF5 存 bytes → str（解码而非 repr）
                sample[key] = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                continue
            value = _as_tensor(g[key][index])
            # 图像字段 (H, W) 补通道维 → (1, H, W)（模型/评估统一形状）
            if key in ("H", "L", "L_clean", "L_up", "P2") and value.dim() == 2:
                value = value.unsqueeze(0)
            sample[key] = value
        c_prior_raw = np.stack(
            [self._read("c_low" if k in ("sigma_z", "n", "eta", "b0", "a1", "alpha") else "c_mid", k, index)
             for k in C_PRIOR_KEYS],
            axis=0,
        )
        sample["c_prior_raw"] = _as_tensor(c_prior_raw)
        sample["c_high"] = _as_tensor(np.stack([self._read("c_high", k, index) for k in _C_HIGH_KEYS]))
        m = {k: self._read("m", k, index) for k in _M_KEYS}
        sample["m"] = {k: _as_tensor(v) for k, v in m.items()}
        sample["split"] = self.split
        return sample


def collate_samples(batch: list[dict]) -> dict:
    """把样本字典列表合并为批量字典（图像/参数堆叠，标量字段列队）。"""
    out: dict = {}
    for key in batch[0]:
        if key in ("m",):
            out[key] = {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}
        elif key in ("sample_id", "split"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
