"""`50` [S5]：Phase 1.5 矩阵（2 臂 × 5 种子 = 10 runs）的 config 生成与断言。

run 命名 ``e15_<tag>_s<idx>``（``tag`` ∈ {joint, indep}）、种子 {100..104}（`50` [S5]）。
10 份 config 的字段契约由 :mod:`src.line2.training.config_check` 记录，本模块只负责
**填值 + 落盘**：生成前逐份过 :func:`~src.line2.training.config_check.validate_phase15_matrix`
（种子集合/参数量差/档集/站集/k_train/站 1 粒子数/``val_beams`` 守卫），断言不通过就
不落盘任何 config。

公平性（`40` [S1] C3 的 Phase 1.5 版）：10 份 config 之间只许在 ``variant``/``seed``/
``out_dir`` 与两臂各自的拓扑键（``base``/``blocks`` vs ``ind_base``/``ind_blocks``）上不同；
其余键（数据/预算/损失/优化器/档集/站集）逐字相同，由 :data:`FAIRNESS_KEYS` 显式断言。

用法：
    python -m src.line2.experiments.phase15_configs [--results-dir <dir>] [--data-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.line2.data_pipeline.dataset import git_head
from src.line2.training.config_check import (
    PHASE15_EVAL_EVERY,
    PHASE15_K_TRAIN,
    PHASE15_MAX_STEPS,
    PHASE15_PATIENCE,
    PHASE15_SEEDS,
    PHASE15_STATIONS,
    PHASE15_TIERS,
    validate_phase15_matrix,
)

__all__ = [
    "REPO",
    "DEFAULT_RESULTS",
    "DEFAULT_DATA",
    "CONFIG_DIRNAME",
    "VARIANT_RUN_TAG",
    "PHASE15_VARIANTS",
    "PHASE15_DATA_VERSION",
    "PHASE15_SPEC_VERSION",
    "FAIRNESS_KEYS",
    "VARIANT_TOPOLOGY_KEYS",
    "IDENTITY_KEYS",
    "run_id_for",
    "run_ids",
    "build_config",
    "build_matrix",
    "generate",
    "main",
]

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = REPO / "studies/line2_particle_sr/results"
DEFAULT_DATA = REPO / "studies/line2_particle_sr/data/line2_ds_v2"
CONFIG_DIRNAME = "e15_configs"

#: run 目录命名（`50` [S5]）：joint 臂 ``e15_joint_s<i>``、independent 臂 ``e15_indep_s<i>``
VARIANT_RUN_TAG = {"joint": "joint", "independent": "indep"}
PHASE15_VARIANTS: tuple[str, ...] = ("joint", "independent")
#: 版本三元组的数据/规格两项（code_version 由 :func:`build_config` 取当次 git HEAD）
PHASE15_DATA_VERSION = "line2_ds_v2"
PHASE15_SPEC_VERSION = "line2-spec-v1.1"

#: 10 份 config 之间必须逐字相同的键（除身份键与两臂拓扑键外的一切）
FAIRNESS_KEYS: tuple[str, ...] = (
    "phase", "kappa", "data_dir", "data_version", "spec_version", "tiers", "stations",
    "k_train", "max_steps", "eval_every", "patience", "batch", "lr", "w_psd", "n_high",
    "num_levels", "device", "torch_threads", "eval_batch",
)
#: 两臂各自的拓扑键（`30` [S4] C2：变体间差异只在网络拓扑）
VARIANT_TOPOLOGY_KEYS: dict[str, tuple[str, ...]] = {
    "joint": ("base", "blocks"),
    "independent": ("ind_base", "ind_blocks"),
}
#: 允许跨 run 变化的非拓扑键（身份与落盘位置）
IDENTITY_KEYS: tuple[str, ...] = ("variant", "seed", "out_dir", "code_version")


def run_id_for(variant: str, index: int) -> str:
    """``(臂, 序号)`` → run_id（``e15_joint_s0`` …）；序号与 :data:`PHASE15_SEEDS` 一一对应。"""
    if variant not in VARIANT_RUN_TAG:
        raise ValueError(f"未知臂 {variant!r}（∈ {PHASE15_VARIANTS}）")
    if not 0 <= int(index) < len(PHASE15_SEEDS):
        raise ValueError(f"序号 {index} 越界（0–{len(PHASE15_SEEDS) - 1}）")
    return f"e15_{VARIANT_RUN_TAG[variant]}_s{int(index)}"


def run_ids() -> list[str]:
    """矩阵的 10 个 run_id（臂序 joint → independent，种子序 s0 → s4）。"""
    return [run_id_for(variant, i) for variant in PHASE15_VARIANTS for i in range(len(PHASE15_SEEDS))]


def _repo_relative(value: str | Path) -> str:
    """仓库内路径写成仓库相对形式（与既有 e1/e2 config 的 out_dir 同风格）。"""
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def build_config(
    variant: str,
    seed: int,
    *,
    data_dir: str | Path,
    out_dir: str | Path,
    code_version: str | None = None,
) -> dict:
    """单份 Phase 1.5 config（字段契约见 :mod:`src.line2.training.config_check`）。

    ``code_version`` 缺省取当次 git HEAD（`50` Global Constraints 的版本三元组；树脏时
    仍然是 HEAD，脏树的留痕由编排器的 run_meta 承担）。
    """
    cfg = {
        "phase": "1.5",
        "variant": str(variant),
        "seed": int(seed),
        "kappa": "v2",
        "data_dir": _repo_relative(data_dir),
        "out_dir": _repo_relative(out_dir),
        "data_version": PHASE15_DATA_VERSION,
        "spec_version": PHASE15_SPEC_VERSION,
        "tiers": list(PHASE15_TIERS),
        "stations": list(PHASE15_STATIONS),
        "k_train": PHASE15_K_TRAIN,
        "max_steps": PHASE15_MAX_STEPS,
        "eval_every": PHASE15_EVAL_EVERY,
        "patience": PHASE15_PATIENCE,
        "batch": 16,
        "lr": 3e-4,
        "w_psd": 0.1,
        "n_high": 200_000.0,
        "num_levels": 4,
        "device": "cuda:0",
        "torch_threads": 4,
        "eval_batch": 32,
        "code_version": str(code_version if code_version is not None else git_head()),
    }
    if variant == "joint":
        cfg.update({"base": 48, "blocks": 2})
    elif variant == "independent":
        cfg.update({"ind_base": 16, "ind_blocks": 3})
    else:
        raise ValueError(f"未知臂 {variant!r}（∈ {PHASE15_VARIANTS}）")
    return cfg


def build_matrix(
    *, results_dir: str | Path = DEFAULT_RESULTS, data_dir: str | Path = DEFAULT_DATA,
    code_version: str | None = None,
) -> list[dict]:
    """10 份 config（两臂 × 种子 {100..104}），``out_dir`` 落在 ``<results_dir>/<run_id>``。"""
    results_dir = Path(results_dir)
    code = str(code_version if code_version is not None else git_head())
    return [
        build_config(
            variant,
            seed,
            data_dir=data_dir,
            out_dir=results_dir / run_id_for(variant, index),
            code_version=code,
        )
        for variant in PHASE15_VARIANTS
        for index, seed in enumerate(PHASE15_SEEDS)
    ]


def _assert_fairness(configs: list[dict]) -> dict:
    """10 份 config 的公平性断言；返回逐键取值一览（供 e15_matrix.json 落盘）。"""
    reference = configs[0]
    for cfg in configs:
        run = Path(cfg["out_dir"]).name
        for key in FAIRNESS_KEYS:
            if json.dumps(cfg.get(key), sort_keys=True) != json.dumps(reference.get(key), sort_keys=True):
                raise ValueError(
                    f"{run}：公平性键 {key} 与 {Path(reference['out_dir']).name} 不一致"
                    f"（{cfg.get(key)!r} vs {reference.get(key)!r}）"
                    "——10 份 config 只许在身份键与两臂拓扑键上不同"
                )
        diff = {
            key for key in set(cfg) | set(reference)
            if key not in set(IDENTITY_KEYS) | set(VARIANT_TOPOLOGY_KEYS["joint"])
            | set(VARIANT_TOPOLOGY_KEYS["independent"])
            and cfg.get(key) != reference.get(key)
        }
        if diff:
            raise ValueError(f"{run}：出现白名单外的差异键 {sorted(diff)}")
        # 死键：另一臂的拓扑键留在 config 里会让读者以为该臂还有另一层结构
        other = VARIANT_TOPOLOGY_KEYS["independent" if cfg["variant"] == "joint" else "joint"]
        dead = [key for key in other if key in cfg]
        if dead:
            raise ValueError(f"{run}：{cfg['variant']} 臂携带死键 {dead}")
    return {
        "fairness_keys": list(FAIRNESS_KEYS),
        "identity_keys": list(IDENTITY_KEYS),
        "variant_topology_keys": {k: list(v) for k, v in VARIANT_TOPOLOGY_KEYS.items()},
    }


def generate(
    *,
    results_dir: str | Path = DEFAULT_RESULTS,
    data_dir: str | Path = DEFAULT_DATA,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    code_version: str | None = None,
) -> dict:
    """生成 10 份 config 与矩阵记录 ``e15_matrix.json``（断言不通过则不落盘任何文件）。

    落盘目录默认 ``<results_dir>/e15_configs``（与 E2 的 ``e2_configs`` 同惯例）；
    矩阵记录落 ``<results_dir>/e15_matrix.json``，含 :func:`validate_phase15_matrix`
    的核对汇总（种子集合/参数量差/抽批站号）与逐 config 的路径。``code_version`` 缺省取
    当次 git HEAD。
    """
    results_dir = Path(results_dir)
    configs_dir = Path(out_dir) if out_dir is not None else results_dir / CONFIG_DIRNAME
    matrix_path = results_dir / "e15_matrix.json"

    configs = build_matrix(results_dir=results_dir, data_dir=data_dir, code_version=code_version)
    validation = validate_phase15_matrix(configs, data_dir=Path(data_dir))
    fairness = _assert_fairness(configs)

    entries = []
    for cfg in configs:
        run_id = Path(cfg["out_dir"]).name
        path = configs_dir / f"{run_id}.json"
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entries.append(
            {
                "path": str(path),
                "run_id": run_id,
                "variant": cfg["variant"],
                "seed": cfg["seed"],
                "out_dir": cfg["out_dir"],
                "code_version": cfg["code_version"],
            }
        )
    payload = {
        "phase": "1.5",
        "spec": "`50` [S5] 矩阵：2 臂 × 5 种子 = 10 runs",
        "results_dir": str(results_dir),
        "data_dir": str(data_dir),
        "data_version": PHASE15_DATA_VERSION,
        "spec_version": PHASE15_SPEC_VERSION,
        "seeds": list(PHASE15_SEEDS),
        "variants": list(PHASE15_VARIANTS),
        "tiers": list(PHASE15_TIERS),
        "stations": list(PHASE15_STATIONS),
        "k_train": PHASE15_K_TRAIN,
        **fairness,
        "validation": validation,
        "config_dir": str(configs_dir),
        "matrix_path": str(matrix_path),
        "configs": entries,
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        matrix_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="line2 Phase 1.5 矩阵 config 生成（spec 50 [S5]；先过 [S4] C3 矩阵断言）"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"10 份 config 的落盘目录（默认 <results-dir>/{CONFIG_DIRNAME}）")
    parser.add_argument("--dry-run", action="store_true", help="只算不写")
    args = parser.parse_args(argv)
    payload = generate(
        results_dir=args.results_dir, data_dir=args.data_dir,
        out_dir=args.out_dir, dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "configs": len(payload["configs"]),
                "config_dir": payload["config_dir"],
                "matrix_path": payload["matrix_path"],
                "params_total": payload["validation"]["params_total"],
                "param_rel_diff": payload["validation"]["param_rel_diff"],
                "station_check": payload["validation"]["station_check"],
                "dry_run": payload["dry_run"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
