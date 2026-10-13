"""`50` [S5]：Phase 1.5 矩阵编排（E0' 时间戳门禁 + 双 GPU 队列 + 60 评估 + 路由 + 对齐 + 门禁）。

子命令（每一步只读上一步的落盘产物，任一子命令失败即停）：

- ``configs``：生成 10 份 config（过 `50` [S4] C3 矩阵断言，见
  :mod:`src.line2.experiments.phase15_configs`）；
- ``train``：双卡队列跑 run。**每个 run 启动前**调
  :func:`check_e0_prime_precedes`——v2 E0' 三件产物（曲线 / 容差线 / B-const 两站）的
  落盘时间戳必须早于该 run 的启动时间戳，违例拒绝启动（`50` [S5] C1 / Global Constraints
  「尺子先行」）；队列按臂整分（joint 一卡、independent 一卡），每 run 落 ``run_meta.json``；
- ``eval``：每 run × {3×10³,1×10⁴,1×10⁵} × 站 {0,1} = 6 份评估（`50` [S5] C8），
  产物 ``eval_tier*_station*.json``；站 1 = 纯迁移评估（训练未见）；已存在则默认跳过
  （`00` GC3「test 只评一次」），每次派发（含跳过）追加 ``eval_history.jsonl``；
- ``route``：读 val 段 val_nll → ``e15_routing.json``（`50` [S5] 路由）；
- ``align``：对**获胜臂** 5 个 run 跑个体信息仪器 → ``e15_align_<run_id>.json``（G2 输入）；
- ``gates``：G1–G5 → ``e15_gates.json``（:mod:`src.line2.experiments.phase15_gates`）；
- ``map``：v2 产出物渲染 → ``phase15_gates_report.md``（`50` [S5] C9）。

用法：
    python -m src.line2.experiments.phase15_orchestrator configs
    python -m src.line2.experiments.phase15_orchestrator train --gpus 0 1
    python -m src.line2.experiments.phase15_orchestrator eval --device cuda:0
    python -m src.line2.experiments.phase15_orchestrator route
    python -m src.line2.experiments.phase15_orchestrator align --device cuda:0
    python -m src.line2.experiments.phase15_orchestrator gates
    python -m src.line2.experiments.phase15_orchestrator map
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from . import map_report, phase15_align, phase15_configs, phase15_gates
from .run_e1 import EVAL_HISTORY_FILENAME, gpu_free_mib, tree_dirty
from .run_e1 import _train_env as train_env

__all__ = [
    "REPO",
    "DEFAULT_RESULTS",
    "DEFAULT_DATA",
    "MIN_FREE_MIB",
    "CURVE_V2_FILENAME",
    "TOLERANCE_V2_FILENAME",
    "BCONST_FILENAME",
    "BCONST_GROUPS_FILENAME",
    "TRAIN_OUTCOMES_FILENAME",
    "now_iso",
    "check_e0_prime_precedes",
    "plan_arm_queues",
    "run_one_train",
    "cmd_configs",
    "cmd_train",
    "cmd_eval",
    "cmd_route",
    "cmd_align",
    "cmd_gates",
    "cmd_map",
    "main",
]

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = REPO / "studies/line2_particle_sr/results"
DEFAULT_DATA = REPO / "studies/line2_particle_sr/data/line2_ds_v2"
MIN_FREE_MIB = 20_000
TRAIN_OUTCOMES_FILENAME = "e15_train_outcomes.json"

CURVE_V2_FILENAME = phase15_gates.CURVE_V2_FILENAME
TOLERANCE_V2_FILENAME = phase15_gates.TOLERANCE_V2_FILENAME
BCONST_FILENAME = phase15_gates.BCONST_FILENAME
BCONST_GROUPS_FILENAME = phase15_gates.BCONST_GROUPS_FILENAME
#: E0' 的第三件（裸误差–N 曲线）由 `e0.py --data-dir <v2>` 产出，元数据在同名 .meta.json
CURVE_V2_META_FILENAME = "e0_curve_v2.meta.json"
#: 容差线必须覆盖的训练档（`50` [S3] C3：13 项 × 3 档 × 站 0）
E0_PRIME_TOLERANCE_TIERS = ("3000", "10000", "100000")


def now_iso() -> str:
    """当前 UTC 时间戳（ISO 8601）；编排器的默认 clock，测试可注入可控 clock。"""
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺产物：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_e0_prime_precedes(
    results_dir: str | Path,
    *,
    started_at: str | None = None,
    clock: Callable[[], str] = now_iso,
    data_dir: str | Path | None = None,
) -> dict:
    """v2 E0' 三件产物的落盘时间戳必须先于 run 启动（`50` [S5] C1「尺子先行」）。

    三件 = 裸误差–N 曲线（``e0_curve_v2.json`` + ``.meta.json`` 的 ``completed_at``）、
    容差线表（``tolerance_v2.json``，13 项 × 3 档）、B-const 两站
    （``bconst_v2.json`` 的 ``station["0"]``/``["1"]`` + ``bconst_v2_groups.npz``）。
    任一缺失、时间戳不晚于启动时间、或（给定 ``data_dir`` 时）产物记录的 data_dir 与本次
    运行的数据集不同 → ``ValueError``（违例拒绝启动，不静默降级）。

    Args:
        results_dir: results 目录（三件产物的落盘位置）。
        started_at: run 的启动时间戳；缺省取 ``clock()``。
        clock: 可控时钟（测试注入；缺省 :func:`now_iso`）。
        data_dir: 本次运行的数据集目录（提供时校验产物同源；缺省不校验）。

    Returns:
        ``{"started_at", "products": {名: {path, completed_at, data_version, ...}},
        "data_dir": ...}``。
    """
    results_dir = Path(results_dir)
    started = str(started_at) if started_at is not None else str(clock())
    started_dt = datetime.fromisoformat(started)
    problems: list[str] = []
    products: dict[str, dict] = {}

    curve_dump = results_dir / CURVE_V2_FILENAME
    curve_meta = results_dir / CURVE_V2_META_FILENAME
    if not curve_dump.is_file():
        problems.append(f"缺 curve 产物：{curve_dump}")
    if not curve_meta.is_file():
        problems.append(f"缺 curve 元数据：{curve_meta}（completed_at 无来源）")
    else:
        meta = _read_json(curve_meta)
        products["curve"] = {
            "path": str(curve_dump), "meta": str(curve_meta),
            "completed_at": meta.get("completed_at"), "data_version": meta.get("data_version"),
            "data_dir": meta.get("data_dir"),
            "tiers": meta.get("tiers"), "stations": meta.get("stations"),
        }

    tolerance_path = results_dir / TOLERANCE_V2_FILENAME
    if not tolerance_path.is_file():
        problems.append(f"缺 tolerance 产物：{tolerance_path}")
    else:
        payload = _read_json(tolerance_path)
        products["tolerance"] = {
            "path": str(tolerance_path), "completed_at": payload.get("completed_at"),
            "data_version": payload.get("data_version"), "data_dir": payload.get("data_dir"),
            "tiers": sorted((payload.get("dev_table") or {}).keys()),
        }
        missing_tiers = [
            tier for tier in E0_PRIME_TOLERANCE_TIERS
            if tier not in (payload.get("dev_table") or {})
        ]
        if missing_tiers:
            problems.append(f"{tolerance_path}：容差线缺档 {missing_tiers}（`50` [S3] C3 三档）")

    bconst_path = results_dir / BCONST_FILENAME
    groups_path = results_dir / BCONST_GROUPS_FILENAME
    if not bconst_path.is_file():
        problems.append(f"缺 bconst 产物：{bconst_path}")
    if not groups_path.is_file():
        problems.append(f"缺 bconst 组图：{groups_path}")
    if bconst_path.is_file():
        payload = _read_json(bconst_path)
        stations = sorted(int(s) for s in (payload.get("station") or {}))
        products["bconst"] = {
            "path": str(bconst_path), "groups": str(groups_path),
            "completed_at": payload.get("completed_at"), "data_version": payload.get("data_version"),
            "data_dir": payload.get("data_dir"), "stations": stations,
        }
        if stations != [0, 1]:
            problems.append(
                f"{bconst_path}：B-const 站集 {stations} ≠ [0, 1]（`50` [S3] C4 两站口径）"
            )

    for name, record in products.items():
        completed = record.get("completed_at")
        if completed is None:
            problems.append(f"{name} 缺 completed_at（落盘时间戳无来源，`50` [S5] C1）")
            continue
        if datetime.fromisoformat(str(completed)) >= started_dt:
            problems.append(
                f"违反 `50` [S5] C1：{name} 完成于 {completed}，不早于 run 启动 {started}"
            )
        if data_dir is not None and record.get("data_dir"):
            expected = Path(data_dir).resolve()
            actual = Path(str(record["data_dir"])).resolve()
            if actual != expected:
                problems.append(
                    f"{name} 的 data_dir={record['data_dir']} 与本次运行的数据集 {data_dir} 不同源"
                    "（尺子必须先于训练、且必须量同一份数据）"
                )

    if problems:
        raise ValueError("E0' 尺子先行违例（`50` [S5] C1）：" + "；".join(problems))
    return {
        "started_at": started,
        "products": products,
        "data_dir": None if data_dir is None else str(data_dir),
        "rule": "v2 E0'（curve/tolerance/bconst 两站）落盘时间戳 SHALL 早于任一训练启动",
    }


def plan_arm_queues(runs: Sequence[str], gpus: Sequence[int]) -> dict[int, list[str]]:
    """按**臂**整分队列：joint → 第一张卡、independent → 第二张卡（单卡时同卡串行）。

    两臂逐 run 成本同量级（同数据/同步数/同批大小，拓扑参数差 ≤30%），臂整分使失败域
    清晰（某臂全挂 = 该臂的问题），也避免轮转把同臂 run 分散到两张卡上。
    """
    if not runs:
        raise ValueError("run 列表为空")
    if not gpus:
        raise ValueError("GPU 列表为空")
    groups: dict[str, list[str]] = {"joint": [], "independent": []}
    seen: set[str] = set()
    for run_id in runs:
        if run_id in seen:
            raise ValueError(f"run 重复：{run_id!r}")
        seen.add(run_id)
        if "_joint_" in run_id:
            groups["joint"].append(run_id)
        elif "_indep_" in run_id:
            groups["independent"].append(run_id)
        else:
            raise ValueError(
                f"run_id 不含臂标记：{run_id!r}（Phase 1.5 的命名 = e15_joint_s*/e15_indep_s*）"
            )
    queue: dict[int, list[str]] = {int(g): [] for g in gpus}
    arms = [arm for arm in ("joint", "independent") if groups[arm]]
    for index, arm in enumerate(arms):
        queue[int(gpus[index % len(gpus)])].extend(groups[arm])
    return queue


def run_one_train(
    configs_dir: str | Path,
    results_dir: str | Path,
    data_dir: str | Path,
    run_id: str,
    gpu: int,
    force: bool = False,
    *,
    clock: Callable[[], str] = now_iso,
    free_mib: Mapping[int, int] | None = None,
) -> dict:
    """跑一个 run：E0' 时间戳门禁 → GPU 纪律 → 落 run_meta → 训练子进程。

    ``clock``/``free_mib`` 为注入点（测试用可控 clock 与假显存读数）；E0' 门禁在任何
    GPU 相关动作之前执行，违例时连 run_meta.json 都不落。
    """
    configs_dir = Path(configs_dir)
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)
    cfg_path = configs_dir / f"{run_id}.json"
    cfg = _read_json(cfg_path)
    from src.line2.training.config_check import validate_run_config

    validate_run_config(cfg)

    started_at = str(clock())
    e0_prime = check_e0_prime_precedes(results_dir, started_at=started_at, data_dir=data_dir)

    free = dict(gpu_free_mib() if free_mib is None else free_mib)
    if int(gpu) not in free:
        raise RuntimeError(f"GPU {gpu} 不存在（可用：{sorted(free)}）")
    if free[int(gpu)] < MIN_FREE_MIB and not force:
        raise RuntimeError(
            f"GPU {gpu} 剩余显存 {free[int(gpu)]} MiB < {MIN_FREE_MIB} MiB（05 [S2] GPU 纪律）；"
            "--force 可越过"
        )

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(data_dir / "manifest.json")
    meta = {
        "run_id": run_id,
        "phase": cfg.get("phase"),
        "config": str(cfg_path),
        "started_at": started_at,
        "completed_at": None,
        "returncode": None,
        "gpu": int(gpu),
        "variant": cfg.get("variant"),
        "seed": cfg.get("seed"),
        "kappa": cfg.get("kappa"),
        "tiers": cfg.get("tiers"),
        "stations": cfg.get("stations"),
        "k_train": cfg.get("k_train"),
        "max_steps": cfg.get("max_steps"),
        "data_dir": str(data_dir),
        "data_version": manifest.get("data_version"),
        "spec_version": manifest.get("spec_version"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "tree_dirty": tree_dirty(),
        "e0_prime": {name: rec["completed_at"] for name, rec in e0_prime["products"].items()},
        "cmd": [sys.executable, "-u", "-m", "src.line2.training.train", "--config", str(cfg_path)],
    }
    meta_path = results_dir / run_id / "run_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = results_dir / f"{run_id}_train.log"
    t0 = time.time()
    with log_path.open("ab") as log:
        proc = subprocess.run(meta["cmd"], cwd=REPO, env=train_env(int(gpu)), stdout=log, stderr=log)
    meta["returncode"] = proc.returncode
    meta["completed_at"] = str(clock())
    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _worker(
    gpu: int,
    queue: Sequence[str],
    configs_dir: Path,
    results_dir: Path,
    data_dir: Path,
    force: bool,
    outcomes: dict,
) -> None:
    for run_id in queue:
        print(f"[e15] GPU{gpu} 启动 {run_id} @ {now_iso()}", flush=True)
        try:
            meta = run_one_train(configs_dir, results_dir, data_dir, run_id, gpu, force)
        except Exception as exc:  # 单 run 失败不拖垮同卡后续 run，但必须留痕
            print(f"[e15] GPU{gpu} {run_id} 异常：{type(exc).__name__}: {exc}", flush=True)
            outcomes[run_id] = {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}
            continue
        print(
            f"[e15] GPU{gpu} 结束 {run_id} rc={meta['returncode']} elapsed={meta.get('elapsed_s')}s",
            flush=True,
        )
        outcomes[run_id] = meta


def _run_dirs(results_dir: Path) -> list[str]:
    """results 目录下的 Phase 1.5 run 目录（``e15_*``，含 ckpt 或 summary 才算）。"""
    return sorted(
        p.name for p in Path(results_dir).glob("e15_*")
        if p.is_dir() and ((p / "summary.json").is_file() or (p / "best_val.ckpt").is_file())
    )


def _runs_by_arm(results_dir: Path, runs: Sequence[str] | None = None) -> dict[str, list[str]]:
    """run_id 列表 → {臂: run_ids}（缺臂即报错：门禁与路由都需要两臂齐全）。"""
    ids = list(runs) if runs else _run_dirs(results_dir)
    out: dict[str, list[str]] = {"joint": [], "independent": []}
    for run_id in ids:
        if "_joint_" in run_id:
            out["joint"].append(run_id)
        elif "_indep_" in run_id:
            out["independent"].append(run_id)
        else:
            raise ValueError(f"run_id 不含臂标记：{run_id!r}")
    missing = [arm for arm, runs_ in out.items() if not runs_]
    if missing:
        raise ValueError(f"缺臂的 run 目录：{missing}（`50` [S5] 矩阵 = 两臂各 5 run）")
    return {arm: sorted(runs_) for arm, runs_ in out.items()}


def _append_history(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def cmd_configs(args: argparse.Namespace) -> int:
    payload = phase15_configs.generate(
        results_dir=args.results_dir, data_dir=args.data_dir,
        out_dir=args.out_dir, dry_run=args.dry_run,
    )
    print(json.dumps(
        {"configs": len(payload["configs"]), "config_dir": payload["config_dir"],
         "matrix_path": payload["matrix_path"],
         "params_total": payload["validation"]["params_total"],
         "station_check": payload["validation"]["station_check"], "dry_run": payload["dry_run"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if tree_dirty():
        print(
            "[e15] 警告：工作树有未提交改动——run_meta.json 的 code_version 记的是 HEAD，"
            "与训练进程实际导入的磁盘代码可能不一致（tree_dirty=true 已留痕）",
            flush=True,
        )
    runs = args.runs or sorted(
        p.stem for p in Path(args.configs_dir).glob("e15_*.json")
    )
    queue = plan_arm_queues(runs, args.gpus)
    print(f"[e15] 队列（按臂整分）：{queue}", flush=True)
    outcomes: dict = {}
    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu, q, Path(args.configs_dir), Path(args.results_dir), Path(args.data_dir),
                  args.force, outcomes),
            daemon=False,
        )
        for gpu, q in queue.items() if q
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [run for run, meta in outcomes.items() if meta.get("returncode") != 0]
    out_path = Path(args.results_dir) / TRAIN_OUTCOMES_FILENAME
    out_path.write_text(
        json.dumps({"queue": {str(g): q for g, q in queue.items()}, "outcomes": outcomes},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[e15] 完成：{len(outcomes)} runs，失败 {len(failed)} 个 {failed}", flush=True)
    return 1 if failed else 0


def cmd_eval(args: argparse.Namespace) -> int:
    """每 run × 3 档 × 2 站 = 6 份评估（`50` [S5] C8）。"""
    from . import evaluate

    results_dir = Path(args.results_dir)
    runs = args.runs or _run_dirs(results_dir)
    curve = results_dir / CURVE_V2_FILENAME
    if not curve.is_file():
        raise FileNotFoundError(
            f"{curve} 不存在：评估的 N_eff 插值基底 = v2 裸误差–N 曲线（`50` [S5] C1 尺子先行）"
        )
    for run_id in runs:
        ckpt = results_dir / run_id / "best_val.ckpt"
        if not ckpt.is_file():
            print(f"[e15] 跳过 {run_id}：无 best_val.ckpt", flush=True)
            continue
        for tier in args.tiers:
            for station in args.stations:
                out = phase15_gates.eval_path(results_dir, run_id, int(tier), int(station))
                history = results_dir / run_id / EVAL_HISTORY_FILENAME
                if out.is_file() and not args.force:
                    reason = f"{out.name} 已存在（`00` GC3：test 只评一次；--force 才覆盖）"
                    print(f"[e15] 跳过 {run_id} {out.name}：{reason}", flush=True)
                    _append_history(history, {
                        "at": now_iso(), "run_id": run_id, "tier": int(tier),
                        "station": int(station), "subs": args.subs, "out": out.name,
                        "skipped": True, "reason": reason, "overwrote_existing": False,
                    })
                    continue
                overwrote = out.is_file()
                argv = [
                    "--ckpt", str(ckpt), "--data-dir", str(args.data_dir), "--split", "test",
                    "--station", str(int(station)), "--n-low", str(int(tier)),
                    "--subs", args.subs, "--curve", str(curve), "--out", str(out),
                    "--device", args.device,
                ]
                print(f"[e15] eval {run_id} tier={tier} station={station} → {out.name}", flush=True)
                rc = evaluate.main(argv)
                if rc != 0:
                    print(f"[e15] eval {run_id} tier={tier} station={station} 返回 {rc}", flush=True)
                    return rc
                _append_history(history, {
                    "at": now_iso(), "run_id": run_id, "tier": int(tier), "station": int(station),
                    "subs": args.subs, "out": out.name, "skipped": False, "force": bool(args.force),
                    "overwrote_existing": bool(overwrote),
                })
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    runs_by_arm = _runs_by_arm(args.results_dir, args.runs)
    payload = phase15_gates.write_routing(args.results_dir, runs_by_arm)
    print(json.dumps(
        {"winner_arm": payload["winner_arm"], "arm_reading": payload["arm_reading"],
         "winner_by_tier_1e4": payload["winner_by_tier_1e4"],
         "reading_disagreement": payload["reading_disagreement"],
         "routing_path": payload["routing_path"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    """对获胜臂的 run 跑个体信息仪器（G2 的输入，`50` [S5] C3）。"""
    results_dir = Path(args.results_dir)
    routing = phase15_gates.load_routing(results_dir)
    winner = str(routing["winner_arm"])
    runs = args.runs or [
        str(r) for r in (routing.get("runs_by_arm", {}).get(winner) or [])
    ]
    if not runs:
        raise ValueError(f"获胜臂 {winner} 无 run 可对齐（e15_routing.json 的 runs_by_arm 为空）")
    groups = results_dir / BCONST_GROUPS_FILENAME
    for run_id in runs:
        out = results_dir / phase15_gates.align_filename(run_id)
        if out.is_file() and not args.force:
            print(f"[e15] 跳过 align {run_id}：{out.name} 已存在（--force 才覆盖）", flush=True)
            continue
        ckpt = results_dir / run_id / "best_val.ckpt"
        print(f"[e15] align {run_id} → {out.name}", flush=True)
        phase15_align.run_alignment(
            ckpt, args.data_dir, groups, split="test", station=phase15_gates.GATE_STATION,
            n_low=phase15_gates.GATE_TIER, subs=args.subs, device=args.device, out=out,
        )
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    payload = phase15_gates.evaluate_e15_gates_from_disk(
        args.results_dir, tolerance=args.tolerance, out=args.out
    )
    print(json.dumps(
        {"winner_arm": payload["arms"]["winner"], **payload["verdicts"],
         "routing_reading_disagreement": payload["routing"]["reading_disagreement"],
         "out": payload["gates_path"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    out = map_report.generate_v2(args.results_dir, reports_dir=args.reports_dir)
    print(json.dumps(
        {"report": str(out["report"]), "winner_arm": out["winner_arm"],
         "verdicts": out["verdicts"], "n_evals": out["n_evals"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="line2 Phase 1.5 矩阵编排（spec 50 [S5]；E0' 尺子先行 + 双卡队列）"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cfg = sub.add_parser("configs", help="生成 10 份 Phase 1.5 config（过矩阵断言）")
    p_cfg.add_argument("--out-dir", type=Path, default=None)
    p_cfg.add_argument("--dry-run", action="store_true")
    p_cfg.set_defaults(func=cmd_configs)

    p_train = sub.add_parser("train", help="双卡队列跑 run（每 run 先过 E0' 时间戳门禁）")
    p_train.add_argument("--runs", nargs="*", default=None)
    p_train.add_argument("--configs-dir", type=Path,
                         default=DEFAULT_RESULTS / phase15_configs.CONFIG_DIRNAME)
    p_train.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p_train.add_argument("--force", action="store_true", help="越过显存空闲检查")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="每 run × 3 档 × 2 站评估")
    p_eval.add_argument("--runs", nargs="*", default=None)
    p_eval.add_argument("--tiers", type=int, nargs="+",
                        default=list(phase15_configs.PHASE15_TIERS))
    p_eval.add_argument("--stations", type=int, nargs="+", default=[0, 1])
    p_eval.add_argument("--subs", default="0-19")
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--force", action="store_true",
                        help="覆盖已存在的 eval_*.json（缺省跳过；`00` GC3 的派发端强制）")
    p_eval.set_defaults(func=cmd_eval)

    p_route = sub.add_parser("route", help="val 段 val_nll → e15_routing.json")
    p_route.add_argument("--runs", nargs="*", default=None)
    p_route.set_defaults(func=cmd_route)

    p_align = sub.add_parser("align", help="获胜臂的个体信息仪器读数（G2 输入）")
    p_align.add_argument("--runs", nargs="*", default=None)
    p_align.add_argument("--subs", default="0-19")
    p_align.add_argument("--device", default="cpu")
    p_align.add_argument("--force", action="store_true")
    p_align.set_defaults(func=cmd_align)

    p_gates = sub.add_parser("gates", help="G1–G5 → e15_gates.json")
    p_gates.add_argument("--tolerance", type=Path, default=None)
    p_gates.add_argument("--out", type=Path, default=None)
    p_gates.set_defaults(func=cmd_gates)

    p_map = sub.add_parser("map", help="v2 产出物渲染（phase15_gates_report.md）")
    p_map.add_argument("--reports-dir", type=Path, default=None)
    p_map.set_defaults(func=cmd_map)

    # `--results-dir`/`--data-dir` 在主命令与子命令两侧都可用：子命令侧用 SUPPRESS 缺省，
    # 不给就沿用主命令的值（两种写法等价，链脚本与手工调用都不会因位置不同而走错目录）。
    for sub_parser in (p_cfg, p_train, p_eval, p_route, p_align, p_gates, p_map):
        sub_parser.add_argument("--results-dir", type=Path, default=argparse.SUPPRESS)
        sub_parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
