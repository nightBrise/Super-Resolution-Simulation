"""config.yaml 解析工具：版本三元组、config hash、目录命名与设备/精度。

复用 60 [S14] 的数据集配置契约（``dataset_builder.load_config`` 回填
code_version / spec_version）；补充 M3 需要的派生字段：
- ``config_digest``：config 的 sha256 摘要（60 [S12] C1 日志字段）；
- ``run_output_dir``：实验目录命名 ``<EXP>_<arm>_<seed>_<run_tag>``
  （60 [S15] 15.2）；
- ``resolve_device`` / ``resolve_precision``：设备与精度解析（60 [S9]：
  Turing 7.5 不支持 bf16）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.generators.dataset_builder import SPEC_VERSION, git_head, load_config as _load_config

#: 规格版本（与 00 [S6] 全局约束 8、60 [S14] C1 配套）。
SPEC_VERSION = SPEC_VERSION


def load_config(path: str | Path) -> dict:
    """读取 config.yaml 并回填缺失的版本三元组（60 [S15] C11）。"""
    return _load_config(path)


def config_digest(config: dict) -> str:
    """config 的 sha256 摘要（前 16 位 hex，60 [S12] C1 日志字段）。"""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_output_dir(config: dict, project_root: Path | None = None) -> Path:
    """实验目录命名 ``<EXP>_<arm>_<seed>_<run_tag>[_<config_tag>]``（60 [S15] 15.2）。

    例如 ``EXP-01_A_seed0_run1``；``config_tag`` 非空时追加。根目录为
    ``<project_root>/results/``（缺省取调用方项目根）。
    """
    root = project_root or Path(__file__).resolve().parents[2]
    # 研究线根（方案 B）：config `study_root` 非空时走 `studies/<study_root>/results/`，
    # 为空/缺省时兜底 `root/results/`（向后兼容，行为与迁移前一致）。
    study_root = str(config.get("study_root", "")).strip()
    base = (root / "studies" / study_root / "results") if study_root else (root / "results")
    exp = str(config["experiment_id"])
    scheme = str(config["scheme"])
    seed = int(config.get("seed_index", 0))
    run_tag = str(config.get("run_tag", "run1"))
    name = f"{exp}_{scheme}_seed{seed}_{run_tag}"
    config_tag = str(config.get("config_tag", "")).strip()
    if config_tag:
        name = f"{name}_{config_tag}"
    return base / name


def resolve_data_version(config: dict) -> str:
    """数据版本号：优先 ``data_version`` 字段，回退 ``dataset.version``。"""
    v = config.get("data_version") or config.get("dataset", {}).get("version")
    v = str(v).replace("data/", "").strip("/")
    if not v or v.startswith("<"):
        v = str(config["dataset"]["version"])
    return v


def resolve_device(config: dict) -> str:
    """解析运行设备：单卡取 ``gpu.device`` 首个设备；无 GPU 时回退 CPU。"""
    import torch

    devices = [d.strip() for d in str(config.get("gpu", {}).get("device", "cuda:0")).split(",")]
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    device = devices[min(rank, len(devices) - 1)]
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return "cpu"
    return device


def resolve_precision(config: dict) -> str:
    """训练精度解析（60 [S9] precision 说明）：仅 ``fp32`` / ``fp16``。

    Turing（Compute Capability 7.5）不支持 bf16——配置为 bf16 时拒绝启动。
    """
    precision = str(config.get("training", {}).get("precision", "fp32")).lower()
    if precision == "bf16":
        raise ValueError(
            "Turing 7.5 不支持 bf16（60 [S9]）：三方案 SHALL 统一使用 fp32 或 fp16"
        )
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"未知精度 {precision}（仅支持 fp32 / fp16）")
    return precision


def git_head_sha() -> str:
    """当前 git commit hash（code_version）。"""
    return git_head()
