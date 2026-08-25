"""checkpoint 与 seeds.json 读写（60 [S12] C2、60 [S14] C8）。

checkpoint 契约：``torch.save`` 字典，含模型权重、配置（含版本三元组）、
随机种子、数据版本、训练曲线与方案 C 的标准化统计量（60 [S5] C3：验证/
测试复用训练集统计量，评估与训练必须使用同一组）。最终评估默认读
``best_val.ckpt``（60 [S12] C3）。

seeds.json（60 [S14] C8）：``master_seed``、``scheme_*_seed_*``、
``data_seeds`` 全集 + 本次运行实际使用的种子。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: checkpoint 内的模型标识键。
MODEL_CLASS_KEY = "model_class"
NETWORK_CONFIG_KEY = "network_config"
C_PRIOR_STATS_KEYS = ("c_prior_mu", "c_prior_sigma")


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """保存 checkpoint（自动建目录，兼容 numpy 数组值）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.asarray(value, dtype=np.float32))
        saved[key] = value
    torch.save(saved, str(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """加载 checkpoint（map_location='cpu'，张量转 numpy 数组兼容键）。"""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in C_PRIOR_STATS_KEYS:
        if key in state and isinstance(state[key], torch.Tensor):
            state[key] = state[key].numpy()
    return state


def write_seeds_json(config: dict, out_dir: str | Path) -> Path:
    """按 60 [S14] C8 落盘 ``seeds.json``（随 checkpoint 一起保存）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scheme = str(config["scheme"])
    seed_index = int(config.get("seed_index", 0))
    scheme_seeds = config.get("scheme_seeds", {})
    used_key = f"scheme_{scheme}_seed_{seed_index}"
    seeds = {
        "master_seed": int(config.get("master_seed", 0)),
        "used_scheme_seed": int(scheme_seeds.get(used_key, 0)),
        "scheme_seeds": {k: int(v) for k, v in scheme_seeds.items()},
        "data_seeds": {k: int(v) for k, v in config.get("data_seeds", {}).items()},
    }
    path = out_dir / "seeds.json"
    path.write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def checkpoint_meta(config: dict, step: int, val_loss: float | None) -> dict:
    """checkpoint 元数据字段（配置、种子、数据版本、spec 版本）。"""
    return {
        "config": config,
        "config_hash": config.get("_config_hash"),
        "data_version": config.get("_data_version"),
        "spec_version": config.get("_spec_version"),
        "step": int(step),
        "val_loss": val_loss,
    }
