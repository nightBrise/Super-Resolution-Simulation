"""L4 [S1] C4：按 E1 的 val 段结论生成 E2 的 10 份 config（spec line2/40）。

E2 的获胜变体由 E1 的 **val 段**结论按 [S3] 路由规则唯一决定（`40` [S1] C4「脚本读取，
不人工指定」）。数字来源 = 每个 E1 run 目录 `summary.json` 的 ``best_val_n_eff``（两段式
消费 [S3]/D6：路由只吃 val 段，test 段留给 H1/H3/H8 出口门禁）；路由规则本身复用
:func:`~src.line2.experiments.gates.winner_variant`，分支说明由
:func:`~src.line2.experiments.gates.evaluate_h3` 的合法枚举给出（[S3] C3），本模块不另写
一套路由规则。`run_meta.json` 是每个 run 的溯源（`40` [S1] C2），用于交叉校验
variant/seed/n_low 与 config 模板指向同一次训练。

产物（均落入 ``results/``，运行产物不入库）：

- ``<results-dir>/e2_configs/e2_<variant>_n<N>_s<idx>.json``——10 份 E2 config
  （获胜变体 × N_low ∈ {3000, 100000} × 种子 100–104）。除 ``variant``/``seed``/``n_low``/
  ``out_dir`` 外逐字沿用 E1 同变体 config：GC4 公平性要求 E2 与 E1 只在 N_low 档位上不同；
- ``<results-dir>/e2_routing.json``——路由记录（两臂 seed→val 数字表、配对表、获胜变体、
  命中分支、生成 config 清单）。

缺 summary/run_meta、某臂种子不足 5 个、两臂种子集合不同、路由结论为 indeterminate 时
一律报错退出，不静默生成。

用法::

    python -m src.line2.experiments.make_e2_configs --results-dir <dir> \
        [--out-dir <dir>] [--e1-config-dir <dir>] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .gates import (
    VERDICT_DIRECTIONAL,
    VERDICT_FAIL,
    VERDICT_INDETERMINATE,
    VERDICT_PASS,
    evaluate_h3,
    winner_variant,
)

__all__ = [
    "E1_SEEDS",
    "E2_N_LOW",
    "E2_FAIRNESS_KEYS",
    "E2_OVERRIDE_KEYS",
    "E1_TEMPLATE_DIFF_KEYS",
    "ROUTING_BRANCHES",
    "VARIANT_RUN_TAG",
    "SUMMARY_FILENAME",
    "RUN_META_FILENAME",
    "ROUTING_FILENAME",
    "CONFIG_DIRNAME",
    "DEFAULT_RESULTS",
    "DEFAULT_E1_CONFIG_DIR",
    "load_e1_runs",
    "route_winner",
    "load_variant_template",
    "build_e2_config",
    "generate",
    "main",
]

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = REPO / "studies/line2_particle_sr/results"
DEFAULT_E1_CONFIG_DIR = REPO / "docs/specs/line2/e1_configs"

SUMMARY_FILENAME = "summary.json"
RUN_META_FILENAME = "run_meta.json"
ROUTING_FILENAME = "e2_routing.json"
CONFIG_DIRNAME = "e2_configs"

#: `40` [S1]：E1/E2 每变体 5 种子，两臂同组 100–104（`40` D10 冻结）
E1_SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104)
#: `40` [S1] E2 的两档 N_low（{3e3, 1e5}；1e4 档由 E1 承担）
E2_N_LOW: tuple[int, ...] = (3_000, 100_000)

#: run 目录命名（`40` [S1]）：joint 臂 ``e1_joint_s<i>``、independent 臂 ``e1_indep_s<i>``
VARIANT_RUN_TAG = {"joint": "joint", "independent": "indep"}
E1_VARIANTS: tuple[str, ...] = ("joint", "independent")

#: GC4 公平性键（`40` [S1] C3）——E2 与 E1 同变体必须逐字相同；受控变量见 E2_OVERRIDE_KEYS
E2_FAIRNESS_KEYS: tuple[str, ...] = (
    "data_dir", "n_high", "max_steps", "eval_every", "patience",
    "batch", "lr", "w_psd", "device", "torch_threads",
)
#: E2 相对 E1 模板唯一的差异键：n_low 是 E2 的自变量，另三个是身份/落盘键
E2_OVERRIDE_KEYS: tuple[str, ...] = ("variant", "seed", "n_low", "out_dir")
#: E1 同变体 5 份 config 之间允许的差异键（同变体的 5 个种子只差身份与落盘目录）
E1_TEMPLATE_DIFF_KEYS: tuple[str, ...] = ("seed", "out_dir")

#: `40` [S3] 获胜变体路由的全分支表：计数码 → (val 段 H3 合法枚举, 分支说明, 允许的获胜变体)
ROUTING_BRANCHES: dict[str, dict] = {
    "5/5": {
        "h3": VERDICT_PASS,
        "rule": "5/5 对联合 ≥ 独立 → 联合",
        "winners": ("joint",),
    },
    "4/5": {
        "h3": VERDICT_DIRECTIONAL,
        "rule": "4/5 → 联合（方向成立）",
        "winners": ("joint",),
    },
    "3/5": {
        "h3": VERDICT_INDETERMINATE,
        "rule": "3/5 → val N_eff 中位数较高者为获胜变体（H3 不可判定）",
        "winners": ("joint", "independent"),
    },
    "<=2/5": {
        "h3": VERDICT_FAIL,
        "rule": "≤2/5 → 独立",
        "winners": ("independent",),
    },
}
_BRANCH_BY_VERDICT = {spec["h3"]: code for code, spec in ROUTING_BRANCHES.items()}


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺产物：{path}（{what}）——不静默生成 E2 config")
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_run_dirs(results_dir: Path, variant: str) -> list[Path]:
    """某臂的全部 E1 run 目录（``e1_<tag>_s*``；排除同前缀的训练日志文件）。"""
    tag = VARIANT_RUN_TAG[variant]
    dirs = sorted(p for p in results_dir.glob(f"e1_{tag}_s*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"{results_dir}：未找到 {variant} 臂的任何 E1 run 目录（e1_{tag}_s*）"
        )
    return dirs


def _val_record(run_dir: Path, variant: str) -> dict:
    """单个 E1 run 的 val 段数字 + 溯源（summary.json::best_val_n_eff、run_meta.json）。"""
    summary_path = run_dir / SUMMARY_FILENAME
    meta_path = run_dir / RUN_META_FILENAME
    summary = _read_json(summary_path, f"E1 val 段数字（{run_dir.name}）")
    meta = _read_json(meta_path, f"E1 run 溯源（{run_dir.name}，`40` [S1] C2）")
    value = summary.get("best_val_n_eff")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"{summary_path}：best_val_n_eff={value!r} 非有限——val 段数字缺失，路由无输入"
        )
    for src, payload in ((summary_path, summary), (meta_path, meta)):
        if str(payload.get("variant")) != variant:
            raise ValueError(
                f"{src}：variant={payload.get('variant')!r} 与目录命名 {variant!r} 不符"
            )
        if payload.get("seed") is None:
            raise ValueError(f"{src}：缺 seed 字段（同种子成对比较的前提，`40` [S3]）")
    if int(meta["seed"]) != int(summary["seed"]):
        raise ValueError(
            f"{run_dir}：summary 种子 {summary['seed']} ≠ run_meta 种子 {meta['seed']}"
            "——同一 run 的两份溯源必须指向同一次训练"
        )
    if meta.get("n_low") is None:
        raise ValueError(f"{meta_path}：缺 n_low 字段（E2 换档的前提）")
    return {
        "run_id": run_dir.name,
        "seed": int(summary["seed"]),
        "val_n_eff": float(value),
        "n_low": int(meta["n_low"]),
        "summary_path": str(summary_path),
        "run_meta_path": str(meta_path),
    }


def load_e1_runs(results_dir: str | Path) -> dict[str, dict[int, dict]]:
    """读两臂全部 E1 run 的 val 段数字，每臂按 seed 升序。

    返回 ``{variant: {seed: {"run_id", "seed", "val_n_eff", "n_low", "summary_path",
    "run_meta_path"}}}``。缺 summary/run_meta、同臂同种子多份 run、或某臂种子数不等于 5 →
    报错（`40` [S1] E1 每变体 5 种子），不静默生成。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    out: dict[str, dict[int, dict]] = {}
    for variant in E1_VARIANTS:
        bucket: dict[int, dict] = {}
        for run_dir in _arm_run_dirs(results_dir, variant):
            record = _val_record(run_dir, variant)
            if record["seed"] in bucket:
                raise ValueError(
                    f"{variant} 臂种子 {record['seed']} 有多份 run（"
                    f"{bucket[record['seed']]['run_id']} / {record['run_id']}）"
                    "——同种子成对比较要求每个种子唯一（`40` [S3]）"
                )
            bucket[record["seed"]] = record
        if len(bucket) != len(E1_SEEDS):
            raise ValueError(
                f"{variant} 臂只有 {len(bucket)} 个种子 {sorted(bucket)}；"
                f"`40` [S1] 要求每变体 {len(E1_SEEDS)} 个（100–104）"
                "——不静默生成 E2 config"
            )
        out[variant] = {seed: bucket[seed] for seed in sorted(bucket)}
    return out


def _val_table(runs: Mapping[int, dict], seeds: Sequence[int]) -> dict[str, float]:
    return {str(seed): float(runs[seed]["val_n_eff"]) for seed in seeds}


def route_winner(runs: Mapping[str, Mapping[int, dict]]) -> dict:
    """[S3] 获胜变体路由：val 段同种子成对计数 → 获胜变体（规则复用 gates）。

    两臂种子集合不同（不得退化为槽位配对）、种子集合不是冻结的 100–104、或结论为
    ``indeterminate`` 时抛错——脚本读取，不人工挑选。
    """
    joint, indep = runs["joint"], runs["independent"]
    if sorted(joint) != sorted(indep):
        raise ValueError(
            f"两臂种子集合不同：joint {sorted(joint)} vs independent {sorted(indep)}"
            "——按种子值配对不成立（`40` [S3] H3 同种子成对），不退化为槽位配对"
        )
    seeds = sorted(joint)
    if seeds != list(E1_SEEDS):
        raise ValueError(
            f"E1 两臂种子 {seeds} ≠ E2 种子档 {list(E1_SEEDS)}（`40` [S1] 每变体 5 种子，D10）"
            "——E2 的 _s<idx> 与种子的一一对应关系无定义，不生成 config"
        )
    joint_vals = [float(joint[seed]["val_n_eff"]) for seed in seeds]
    indep_vals = [float(indep[seed]["val_n_eff"]) for seed in seeds]
    h3_val = evaluate_h3(joint_vals, indep_vals)  # val 段 H3（[S3] C3 合法枚举）
    branch = _BRANCH_BY_VERDICT[h3_val]
    winner = winner_variant(joint_vals, indep_vals)
    if winner not in E1_VARIANTS:
        raise RuntimeError(
            f"E1 val 段路由结论为 {winner}（[S3] 路由，n_pairs={len(seeds)}）"
            "——E2 的获胜变体无定义，不生成任何 config（须先补足 E1 run）"
        )
    if winner not in ROUTING_BRANCHES[branch]["winners"]:
        raise RuntimeError(
            f"路由结论 {winner} 与分支 {branch}（{ROUTING_BRANCHES[branch]['rule']}）矛盾"
            "——winner_variant 与 [S3] 分支表不一致，请核对 gates"
        )
    n_low_set = {runs[arm][seed]["n_low"] for arm in E1_VARIANTS for seed in seeds}
    if len(n_low_set) != 1:
        raise ValueError(
            f"E1 run 的 n_low 不一致：{sorted(n_low_set)}"
            "——val 段数字跨档不可比（GC4 公平性）"
        )
    return {
        "basis": "val",
        "source": {"summary": f"{SUMMARY_FILENAME}:best_val_n_eff",
                   "run_meta": RUN_META_FILENAME},
        "pairing": {
            "basis": "seed_value",
            "pairs": [[seed, seed] for seed in seeds],
            "seed_sets": {"joint": seeds, "independent": seeds},
        },
        "n_pairs": len(seeds),
        "seeds": seeds,
        "val_n_eff": {
            "joint": _val_table(joint, seeds),
            "independent": _val_table(indep, seeds),
        },
        "val_medians": {
            "joint": float(np.median([joint[seed]["val_n_eff"] for seed in seeds])),
            "independent": float(np.median([indep[seed]["val_n_eff"] for seed in seeds])),
        },
        "joint_wins": int(sum(1 for j, i in zip(joint_vals, indep_vals) if j >= i)),
        "h3_val_verdict": h3_val,
        "winner_branch": branch,
        "branch_rule": ROUTING_BRANCHES[branch]["rule"],
        "winner_variant": winner,
        "e1_n_low": next(iter(n_low_set)),
        "e1_runs": {
            variant: {
                str(seed): {
                    "run_id": runs[variant][seed]["run_id"],
                    "summary": runs[variant][seed]["summary_path"],
                    "run_meta": runs[variant][seed]["run_meta_path"],
                }
                for seed in seeds
            }
            for variant in E1_VARIANTS
        },
    }


def load_variant_template(
    runs: Mapping[int, dict], variant: str, e1_config_dir: str | Path
) -> tuple[dict, str]:
    """读某变体 5 份 E1 config 作为 E2 模板，返回 ``(模板 config, 模板路径)``。

    每份 config 的 variant/seed/n_low 必须与该 run 的目录名与 run_meta 一致（挡住陈旧的
    config 副本，例如换种子前的旧目录），且 5 份之间除 ``seed``/``out_dir`` 外逐字一致
    （`40` [S1] C3）——否则同变体内不存在可复用的单一模板。
    """
    e1_config_dir = Path(e1_config_dir)
    configs: dict[int, tuple[dict, str]] = {}
    for seed in sorted(runs):
        path = e1_config_dir / f"{runs[seed]['run_id']}.json"
        cfg = _read_json(path, f"E1 config 模板（{variant} 臂，seed {seed}）")
        if str(cfg.get("variant")) != variant:
            raise ValueError(
                f"{path}：variant={cfg.get('variant')!r} ≠ {variant!r}（模板须取自获胜变体）"
            )
        if int(cfg.get("seed", -1)) != int(seed):
            raise ValueError(
                f"{path}：config 种子 {cfg.get('seed')!r} ≠ run_meta 种子 {seed}"
                "——config 与 run 不是同一次训练（陈旧副本）"
            )
        if int(cfg.get("n_low", -1)) != int(runs[seed]["n_low"]):
            raise ValueError(
                f"{path}：config n_low {cfg.get('n_low')!r} ≠ run_meta n_low "
                f"{runs[seed]['n_low']}——E2 换档的基准档不一致"
            )
        configs[seed] = (cfg, str(path))
    reference, reference_path = configs[sorted(configs)[0]]
    for seed in sorted(configs):
        cfg, path = configs[seed]
        diff = sorted(
            {k for k in set(cfg) | set(reference) if cfg.get(k) != reference.get(k)}
            - set(E1_TEMPLATE_DIFF_KEYS)
        )
        if diff:
            raise ValueError(
                f"{path} 与 {reference_path}：同变体 config 在 {diff} 上不一致（`40` [S1] C3）"
                "——不存在可复用的单一模板"
            )
    return dict(reference), reference_path


def build_e2_config(
    template: Mapping, variant: str, seed: int, n_low: int, out_dir: str
) -> dict:
    """E2 config = 模板逐字复制，只改 ``variant``/``seed``/``n_low``/``out_dir``（GC4）。"""
    cfg = dict(template)
    cfg["variant"] = variant
    cfg["seed"] = int(seed)
    cfg["n_low"] = int(n_low)
    cfg["out_dir"] = out_dir
    return cfg


def _repo_relative(path: Path) -> str:
    """仓库内路径写成仓库相对形式（与 `docs/specs/line2/e1_configs` 的 out_dir 同风格）。"""
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def generate(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    e1_config_dir: str | Path = DEFAULT_E1_CONFIG_DIR,
    dry_run: bool = False,
) -> dict:
    """读 E1 的 val 段结论 → 落 10 份 E2 config 与路由记录（``dry_run`` 只算不写）。

    config 的 ``out_dir`` 指向 ``<results-dir>/e2_<variant>_n<N>_s<idx>``（run 目录与 E1 同级，
    供 `40` [S4] 的产出物脚本按 ``*/test_eval*.json`` 收集）；``out_dir`` 参数只决定 10 份
    config 文件的落盘目录（默认 ``<results-dir>/e2_configs``）。
    """
    results_dir = Path(results_dir)
    configs_dir = Path(out_dir) if out_dir is not None else results_dir / CONFIG_DIRNAME
    routing_path = results_dir / ROUTING_FILENAME

    runs = load_e1_runs(results_dir)
    decision = route_winner(runs)
    variant = decision["winner_variant"]
    template, template_path = load_variant_template(runs[variant], variant, e1_config_dir)

    run_root = _repo_relative(results_dir)
    configs: list[dict] = []
    for n_low in E2_N_LOW:
        for idx, seed in enumerate(E1_SEEDS):
            run_id = f"e2_{variant}_n{n_low}_s{idx}"
            out_value = f"{run_root}/{run_id}"
            cfg = build_e2_config(template, variant, seed, n_low, out_value)
            unfair = sorted(k for k in E2_FAIRNESS_KEYS if cfg.get(k) != template.get(k))
            if unfair:
                raise RuntimeError(
                    f"{run_id}：公平性键 {unfair} 与 E1 模板不一致（GC4）"
                    "——E2 与 E1 的差异只允许落在 N_low 档位"
                )
            path = configs_dir / f"{run_id}.json"
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            configs.append(
                {
                    "path": str(path),
                    "run_id": run_id,
                    "variant": variant,
                    "seed": seed,
                    "n_low": n_low,
                    "out_dir": out_value,
                }
            )

    payload = {
        "results_dir": str(results_dir),
        "e1_config_dir": str(e1_config_dir),
        "template": {
            "variant": variant,
            "config": template_path,
            "diff_keys_allowed": sorted(E1_TEMPLATE_DIFF_KEYS),
        },
        "fairness_keys": list(E2_FAIRNESS_KEYS),
        **decision,
        "n_low_tiers": list(E2_N_LOW),
        "config_dir": str(configs_dir),
        "routing_path": str(routing_path),
        "configs": configs,
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        routing_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _cli_summary(payload: Mapping) -> dict:
    return {
        "winner_variant": payload["winner_variant"],
        "winner_branch": payload["winner_branch"],
        "branch_rule": payload["branch_rule"],
        "h3_val_verdict": payload["h3_val_verdict"],
        "joint_wins": payload["joint_wins"],
        "n_pairs": payload["n_pairs"],
        "val_medians": payload["val_medians"],
        "n_configs": len(payload["configs"]),
        "config_dir": payload["config_dir"],
        "routing_path": payload["routing_path"],
        "dry_run": payload["dry_run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="line2 E2 config 生成（spec 40 [S1] C4 + [S3] 变体路由；只读 E1 产物）"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="E1 run 与 E2 run 共同的 results 目录（路由记录亦落此处）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"10 份 config 的落盘目录（默认 <results-dir>/{CONFIG_DIRNAME}）")
    parser.add_argument("--e1-config-dir", type=Path, default=DEFAULT_E1_CONFIG_DIR,
                        help="E1 config 模板目录")
    parser.add_argument("--dry-run", action="store_true", help="只算路由与 config 内容，不写文件")
    args = parser.parse_args(argv)
    try:
        payload = generate(
            args.results_dir,
            out_dir=args.out_dir,
            e1_config_dir=args.e1_config_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[make_e2_configs] 无法生成 E2 config：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(_cli_summary(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
