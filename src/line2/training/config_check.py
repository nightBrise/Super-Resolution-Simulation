"""Phase 1.5 run 配置断言（`50` [S4] C3）。

`50` [S5] 生成 10 个 run 的 config 后调用 :func:`validate_phase15_matrix`，一次性
核对两臂种子集合逐元素相同、两臂参数量差 ≤30%、训练档集/站集/k_train，并抽批核对
批次内不含站 1 粒子；单 run 的入口 :func:`validate_run_config` 由
:func:`~src.line2.training.train.train_phase15` 在训练启动时自动调用。

config 字段即契约（Phase 1.5 相对 Phase 1 的差异集中在 ``phase``/``kappa``/
``tiers``/``stations``/``k_train`` 五项，``n_low`` 不再出现）：

.. code-block:: json

    {"phase": "1.5", "variant": "joint", "seed": 100, "kappa": "v2",
     "tiers": [3000, 10000, 100000], "stations": [0], "k_train": 5,
     "max_steps": 50000, "eval_every": 1000, "patience": 10,
     "data_dir": ".../line2_ds_v2", "out_dir": "...", ...}

``val_beams`` 是 Phase 1 遗留的评估开销控制键：它把 val 集截短，而 `50` [S3] C2 的
val_nll（选型与路由的唯一准绳）冻结在**全 150 束 val 集**上定义，截短后读数不是
val_nll、两臂不可比。故 Phase 1.5 的矩阵断言只接受 ``val_beams`` 缺省或恰等于 150
（:data:`PHASE15_VAL_BEAMS`），见 :func:`validate_phase15_matrix`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .data import Line2Pairs, MixedTierBatcher, dataset_manifest

__all__ = [
    "PHASE15_TIERS",
    "PHASE15_SEEDS",
    "PHASE15_STATIONS",
    "PHASE15_K_TRAIN",
    "PHASE15_VAL_BEAMS",
    "PARAM_TOLERANCE",
    "validate_run_config",
    "validate_phase15_matrix",
]

#: 混合档训练档集（`50` [S4] item 2 预注册）
PHASE15_TIERS = (3_000, 10_000, 100_000)
#: 2 臂 × 5 种子（`50` [S5] 矩阵；两臂逐元素相同）
PHASE15_SEEDS = (100, 101, 102, 103, 104)
#: 训练站集：站 1 粒子不进训练/验证（`50` [S4] item 3）
PHASE15_STATIONS = (0,)
#: 每档每束训练子样本数（本批预注册）
PHASE15_K_TRAIN = 5
#: val 集规模（`50` [S3] C2：val_nll 在全 150 束 val 集上定义）——``val_beams`` 只许缺省或等于此值
PHASE15_VAL_BEAMS = 150
#: 两臂参数量差上限（`30` [S4] C1 沿用）
PARAM_TOLERANCE = 0.30
#: 预注册训练预算与评估周期（`30` [S3] C1 的 max_steps=50000 沿用；`50` [S1] 每 1000 步评估）
PHASE15_MAX_STEPS = 50_000
PHASE15_EVAL_EVERY = 1_000
PHASE15_PATIENCE = 10


def validate_run_config(cfg: Mapping) -> dict:
    """单个 Phase 1.5 run config 的断言；返回该 config 的关键契约字段。

    只做结构断言（不触盘、不建网），故可在训练入口直接调用——种子集合/参数量对齐/
    预算预注册与站 1 粒子的抽批核对见 :func:`validate_phase15_matrix`。
    """
    problems: list[str] = []
    if str(cfg.get("phase")) != "1.5":
        problems.append(f"phase 必须为 '1.5'，得到 {cfg.get('phase')!r}")
    if str(cfg.get("variant")) not in ("joint", "independent"):
        problems.append(f"variant 必须为 joint/independent，得到 {cfg.get('variant')!r}")
    if str(cfg.get("kappa")) != "v2":
        problems.append(f"kappa 必须为 'v2'（`50` [S4] C1），得到 {cfg.get('kappa')!r}")
    tiers = tuple(int(t) for t in cfg.get("tiers", ()))
    if sorted(tiers) != sorted(PHASE15_TIERS):
        problems.append(f"训练档集必须为 {sorted(PHASE15_TIERS)}，得到 {sorted(tiers)}")
    stations = tuple(int(s) for s in cfg.get("stations", ()))
    if stations != PHASE15_STATIONS:
        problems.append(f"训练站集必须为 {list(PHASE15_STATIONS)}，得到 {list(stations)}")
    if int(cfg.get("k_train", -1)) != PHASE15_K_TRAIN:
        problems.append(f"k_train 必须为 {PHASE15_K_TRAIN}，得到 {cfg.get('k_train')!r}")
    if problems:
        label = cfg.get("out_dir", cfg.get("variant"))
        raise ValueError(f"run config 断言失败（{label}）：" + "；".join(problems))
    return {
        "variant": str(cfg["variant"]),
        "seed": int(cfg["seed"]),
        "tiers": list(tiers),
        "stations": list(stations),
        "k_train": int(cfg["k_train"]),
        "kappa": str(cfg["kappa"]),
    }


def _reject_val_beams_override(cfg: Mapping) -> None:
    """拒绝截短 val 集的 ``val_beams``（`50` [S3] C2：val_nll 定义在全 150 束 val 集上）。

    Phase 1 的该键用于降低评估开销；Phase 1.5 的 val_nll 是选型与路由的唯一准绳，
    截短后读数只在被截的子集上定义、两臂不可比，故只接受缺省或恰等于
    :data:`PHASE15_VAL_BEAMS`。守卫放在矩阵断言（而非单 run 入口）里：小规模冒烟
    训练仍可按 Phase 1 口径截短 val 集，正式矩阵不许。
    """
    limit = cfg.get("val_beams")
    if limit is None:
        return
    if int(limit) != PHASE15_VAL_BEAMS:
        raise ValueError(
            f"{cfg.get('out_dir', cfg.get('variant'))}：val_beams={limit!r} 不是 "
            f"{PHASE15_VAL_BEAMS}（`50` [S3] C2 的 val_nll 定义在全 {PHASE15_VAL_BEAMS} 束 "
            "val 集上，截短即不是 val_nll）——Phase 1.5 矩阵 config 只许缺省或等于该值"
        )


def _params_total(configs: Iterable[Mapping]) -> dict[str, int]:
    """逐臂建网数参数（同一臂只建一次：种子不改变拓扑）。"""
    from .train import build_model

    counts: dict[str, int] = {}
    for cfg in configs:
        variant = str(cfg["variant"])
        if variant not in counts:
            model = build_model(dict(cfg))
            counts[variant] = sum(p.numel() for p in model.parameters())
    return counts


def _station_counts(cfg: Mapping, data_dir: Path, batch_probe: int) -> dict[int, int]:
    """抽若干批统计站号出现次数（`50` [S4] C3 的"每批含站 1 粒子数 = 0"）。"""
    manifest = dataset_manifest(data_dir)
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    subs = tuple(range(int(cfg["k_train"])))
    sets = {
        int(tier): Line2Pairs(
            data_dir / "train.h5",
            train_beams,
            n_low=int(tier),
            stations=tuple(int(s) for s in cfg["stations"]),
            subs=subs,
        )
        for tier in cfg["tiers"]
    }
    batcher = MixedTierBatcher(sets, batch_size=int(cfg.get("batch", 16)), seed=int(cfg["seed"]))
    seen: dict[int, int] = {}
    batches = iter(batcher)
    for _ in range(batch_probe):
        for station in next(batches)["station"]:
            seen[int(station)] = seen.get(int(station), 0) + 1
    return seen


def validate_phase15_matrix(
    configs: Iterable[Mapping], data_dir: str | Path | None = None, batch_probe: int = 4
) -> dict:
    """Phase 1.5 矩阵的自动断言（`50` [S4] C3）；返回核对汇总供编排器落盘。

    核对项 = ① 两臂种子集合逐元素相同且 = {100..104}；② 两臂参数量差 ≤30%；
    ③ 每 run 的训练档集/站集/k_train/κ 口径（:func:`validate_run_config`）；
    ④ 抽批核验批次内站 1 粒子数 = 0（``data_dir`` 缺省取首个 config 的 ``data_dir``，
    该目录无 manifest 时记 ``station_check = "unavailable:<path>"`` 并跳过抽批）。
    """
    configs = [dict(cfg) for cfg in configs]
    if not configs:
        raise ValueError("空 config 列表：Phase 1.5 矩阵至少含一个 run")
    summary: dict = {
        "runs": [validate_run_config(cfg) for cfg in configs],
        "params_total": {},
        "param_rel_diff": None,
        "station_counts": None,
        "station_check": None,
    }

    for cfg in configs:
        for key, expected in (
            ("max_steps", PHASE15_MAX_STEPS),
            ("eval_every", PHASE15_EVAL_EVERY),
            ("patience", PHASE15_PATIENCE),
        ):
            if int(cfg.get(key, -1)) != expected:
                raise ValueError(
                    f"{cfg.get('out_dir', cfg['variant'])}：{key} 必须为 {expected}"
                    f"（`50` [S4] item 3 预注册），得到 {cfg.get(key)!r}"
                )
        _reject_val_beams_override(cfg)

    by_variant: dict[str, list[int]] = {}
    for cfg in configs:
        by_variant.setdefault(str(cfg["variant"]), []).append(int(cfg["seed"]))
    arm_seeds = {variant: sorted(seeds) for variant, seeds in by_variant.items()}
    for variant, seeds in arm_seeds.items():
        if seeds != list(PHASE15_SEEDS):
            raise ValueError(
                f"{variant} 臂种子集合必须逐元素等于 {list(PHASE15_SEEDS)}，得到 {seeds}"
            )
    if len(set(map(tuple, arm_seeds.values()))) > 1:
        raise ValueError(f"两臂种子集合逐元素相同失败：{arm_seeds}")

    params = _params_total(configs)
    if set(params) != {"joint", "independent"}:
        raise ValueError(f"矩阵需同时含 joint 与 independent 臂，得到 {sorted(params)}")
    rel = (params["independent"] - params["joint"]) / params["joint"]
    if abs(rel) > PARAM_TOLERANCE:
        raise ValueError(
            f"两臂参数量差 {rel:+.1%} 超过 ±{PARAM_TOLERANCE:.0%}"
            f"（joint {params['joint']} vs independent {params['independent']}）"
        )
    summary["params_total"] = params
    summary["param_rel_diff"] = rel

    data_path = Path(data_dir if data_dir is not None else configs[0]["data_dir"])
    if not (data_path / "manifest.json").is_file():
        summary["station_check"] = f"unavailable:{data_path}"
        return summary
    seen = _station_counts(configs[0], data_path, batch_probe)
    if seen.get(1, 0):
        raise ValueError(
            f"批次含站 1 粒子 {seen[1]} 个（`50` [S4] C3 要求 0）："
            f"stations={configs[0]['stations']}，data_dir={data_path}"
        )
    if not seen.get(0, 0):
        raise ValueError(f"抽批结果不含任何站 0 粒子（核验未生效）：{seen}")
    summary["station_counts"] = seen
    summary["station_check"] = f"ok:{data_path}"
    return summary
