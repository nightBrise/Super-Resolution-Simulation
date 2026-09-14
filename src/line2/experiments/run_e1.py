"""L4 [S1] E1/E2 矩阵编排：按 GPU 队列串行跑 run、逐 run 落 `run_meta.json`。

纪律与契约：
- **E0 先行**（`40` [S1] C5）：启动前读 `results/e0_curve.meta.json` 的 `completed_at`，
  与本次 run 的 `started_at` 比较，E0 未完成或时间戳晚于 run → 直接报错不启动。
- **GPU 纪律**（05 [S2]）：启动前查 `nvidia-smi`，目标卡空闲显存 < `--min-free-mib`
  则拒绝（`--force` 可越过）；每 run 只钉一张卡（`CUDA_VISIBLE_DEVICES`），
  config 的 `device` 保持 `cuda:0`（映射到被钉的物理卡）。
- **确定性**：`CUBLAS_WORKSPACE_CONFIG=:4096:8` + `PYTHONHASHSEED=0`，与手工启动一致。
- **run 目录契约**（`40` [S1] C2 + GC7）：`results/<run_id>/` 内含
  `run_meta.json`（启动/结束时间戳 + code/data/spec 三元组 + 变体/种子/档位/GPU
  + `tree_dirty` 工作树脏标记）与训练产出的 `train_log.jsonl`/`best_val.ckpt`/`summary.json`。

子命令：
    python -m src.line2.experiments.run_e1 train [--runs ...] [--gpus 0 1]
    python -m src.line2.experiments.run_e1 eval  [--runs ...] [--station 0] [--n-low <档位>]
    python -m src.line2.experiments.run_e1 gates [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIGS = REPO / "docs/specs/line2/e1_configs"
DEFAULT_RESULTS = REPO / "studies/line2_particle_sr/results"
DEFAULT_DATA = REPO / "studies/line2_particle_sr/data/line2_ds_v1"

__all__ = [
    "main", "run_meta_path", "check_e0_precedes", "gpu_free_mib", "plan_queue", "tree_dirty",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tree_dirty() -> bool:
    """工作树是否有未提交改动（``git status --porcelain`` 非空）。

    ``run_meta.json`` 的 ``code_version`` 记的是 HEAD，而训练子进程导入的是**启动时刻的
    磁盘代码**：树脏时两者可能不一致（`00` GC7 三元组的语义是"这次 run 用的就是这份代码"）。
    因此脏树不阻断启动，但必须留痕——``run_meta.json`` 记 ``tree_dirty``，``train`` 子命令
    启动时打印告警。
    """
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return bool(out.stdout.strip())


def e0_completed_at(results_dir: Path) -> str:
    """E0 曲线完成时间戳（`40` [S1] C5 的左侧证据）。"""
    meta = results_dir / "e0_curve.meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"E0 元数据缺失：{meta}（E0 SHALL 先于任何 E1 训练，`40` [S1] C5）")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if not payload.get("completed_at"):
        raise ValueError(f"{meta} 缺 completed_at 字段")
    return str(payload["completed_at"])


def check_e0_precedes(results_dir: Path, started_at: str) -> None:
    """断言 E0 完成时间戳早于给定启动时间戳，否则抛错（顺序约束的守门）。"""
    e0_at = datetime.fromisoformat(e0_completed_at(results_dir))
    run_at = datetime.fromisoformat(started_at)
    if e0_at > run_at:
        raise RuntimeError(
            f"违反 `40` [S1] C5：E0 完成于 {e0_at.isoformat()}，晚于 run 启动 {run_at.isoformat()}"
        )


def gpu_free_mib() -> dict[int, int]:
    """各卡的剩余显存（MiB）——GPU 纪律检查用。"""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, total, used = (int(part.strip()) for part in line.split(","))
        free[idx] = total - used
    return free


def plan_queue(runs: list[str], gpus: list[int]) -> dict[int, list[str]]:
    """把 run 列表轮转分配到各卡（同名 run 只出现一次，顺序保持）。"""
    if not runs:
        raise ValueError("run 列表为空")
    if not gpus:
        raise ValueError("GPU 列表为空")
    queue: dict[int, list[str]] = {g: [] for g in gpus}
    for i, run_id in enumerate(runs):
        queue[gpus[i % len(gpus)]].append(run_id)
    return queue


def run_meta_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / run_id / "run_meta.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_env(gpu: int) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"
    return env


def run_one_train(
    configs_dir: Path, results_dir: Path, data_dir: Path, run_id: str, gpu: int, force: bool
) -> dict:
    """跑一个训练 run，逐阶段落 `run_meta.json`；返回其内容。"""
    cfg_path = configs_dir / f"{run_id}.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config 不存在：{cfg_path}")
    cfg = _load_json(cfg_path)
    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(data_dir / "manifest.json")

    started_at = now_iso()
    check_e0_precedes(results_dir, started_at)  # C5 守门：E0 未完成则不启动
    free = gpu_free_mib()
    if gpu not in free:
        raise RuntimeError(f"GPU {gpu} 不存在（可用：{sorted(free)}）")
    if free[gpu] < 20000 and not force:
        raise RuntimeError(
            f"GPU {gpu} 剩余显存 {free[gpu]} MiB < 20000 MiB（05 [S2] GPU 纪律）；--force 可越过"
        )

    meta = {
        "run_id": run_id,
        "config": str(cfg_path.relative_to(REPO)),
        "started_at": started_at,
        "completed_at": None,
        "returncode": None,
        "gpu": gpu,
        "variant": cfg.get("variant", "joint"),
        "seed": cfg.get("seed"),
        "n_low": cfg.get("n_low"),
        "n_high": cfg.get("n_high"),
        "max_steps": cfg.get("max_steps"),
        "data_dir": str(data_dir.relative_to(REPO)),
        "data_version": manifest.get("data_version"),
        "spec_version": manifest.get("spec_version"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "tree_dirty": tree_dirty(),
        "cmd": [sys.executable, "-u", "-m", "src.line2.training.train", "--config", str(cfg_path)],
    }
    meta_path = run_meta_path(results_dir, run_id)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = results_dir / f"{run_id}_train.log"
    t0 = time.time()
    with log_path.open("ab") as log:
        proc = subprocess.run(meta["cmd"], cwd=REPO, env=_train_env(gpu), stdout=log, stderr=log)
    meta["returncode"] = proc.returncode
    meta["completed_at"] = now_iso()
    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _worker(
    gpu: int,
    queue: list[str],
    configs_dir: Path,
    results_dir: Path,
    data_dir: Path,
    force: bool,
    outcomes: dict,
) -> None:
    for run_id in queue:
        print(f"[run_e1] GPU{gpu} 启动 {run_id} @ {now_iso()}", flush=True)
        try:
            meta = run_one_train(configs_dir, results_dir, data_dir, run_id, gpu, force)
        except Exception as exc:  # 单 run 失败不拖垮同卡后续 run，但必须留痕
            print(f"[run_e1] GPU{gpu} {run_id} 异常：{type(exc).__name__}: {exc}", flush=True)
            outcomes[run_id] = {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}
            continue
        print(
            f"[run_e1] GPU{gpu} 结束 {run_id} rc={meta['returncode']} "
            f"elapsed={meta.get('elapsed_s')}s",
            flush=True,
        )
        outcomes[run_id] = meta


def cmd_train(args: argparse.Namespace) -> int:
    if tree_dirty():
        print(
            "[run_e1] 警告：工作树有未提交改动（git status --porcelain 非空）——"
            "run_meta.json 的 code_version 记的是 HEAD，与训练进程实际导入的磁盘代码可能"
            "不一致；本次各 run 的 run_meta.json 将记 tree_dirty=true（00 GC7 留痕）",
            flush=True,
        )
    runs = args.runs or sorted(p.stem for p in args.configs_dir.glob("e1_*.json"))
    queue = plan_queue(runs, args.gpus)
    print(f"[run_e1] 计划：{queue}", flush=True)
    outcomes: dict = {}
    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu, q, args.configs_dir, args.results_dir, args.data_dir, args.force, outcomes),
            daemon=False,
        )
        for gpu, q in queue.items()
        if q
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [r for r, m in outcomes.items() if m.get("returncode") != 0]
    tag = "-".join(str(g) for g in args.gpus)
    (args.results_dir / f"e1_train_outcomes_g{tag}.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[run_e1] 完成：{len(outcomes)} runs，失败 {len(failed)} 个 {failed}", flush=True)
    return 1 if failed else 0


def cmd_eval(args: argparse.Namespace) -> int:
    """对每个 run 的 best_val.ckpt 产出 `results/<run_id>/test_eval.json`。"""
    from src.line2.experiments import evaluate

    runs = args.runs or sorted(
        p.name for p in args.results_dir.glob("*") if (p / "best_val.ckpt").is_file()
    )
    curve = args.results_dir / "e0_curve.json"
    for run_id in runs:
        ckpt = args.results_dir / run_id / "best_val.ckpt"
        if not ckpt.is_file():
            print(f"[run_e1] 跳过 {run_id}：无 best_val.ckpt", flush=True)
            continue
        # `--n-low` 缺省 = 按 run 自己的档位（E2 的 3e3/1e5 run 与 E1 的 1e4 同表，
        # `40` [S1] C1 要求覆盖五档；档位是 config 的既有字段，读 run_meta 而非人工传参）
        n_low = args.n_low
        if n_low is None:
            meta = run_meta_path(args.results_dir, run_id)
            if not meta.is_file():
                print(f"[run_e1] 跳过 {run_id}：缺 run_meta.json，无法确定 n_low", flush=True)
                continue
            n_low = int(_load_json(meta)["n_low"])
        # 站 0 用规范名 `test_eval.json`；站 ≠ 0（E3 跨站）另起名，避免与站 0 产物同名相覆
        # ——门禁按 `evaluate.EVAL_FILENAME` + `test_eval*.json` 的 glob 消费（`40` [S2]）。
        name = evaluate.EVAL_FILENAME if args.station == 0 else f"test_eval_station{args.station}.json"
        out = args.results_dir / run_id / name
        argv = [
            "--ckpt", str(ckpt), "--data-dir", str(args.data_dir), "--split", args.split,
            "--station", str(args.station), "--n-low", str(n_low), "--subs", args.subs,
            "--curve", str(curve), "--out", str(out), "--device", args.device,
        ]
        print(f"[run_e1] eval {run_id} (n_low={n_low}) → {out.name}", flush=True)
        rc = evaluate.main(argv)
        if rc != 0:
            print(f"[run_e1] eval {run_id} 返回 {rc}", flush=True)
            return rc
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    from src.line2.experiments import gates

    payload = gates.evaluate_gates_from_disk(args.results_dir)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 E1/E2 矩阵编排（spec 40 [S1]）")
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="按 GPU 队列跑训练 run")
    p_train.add_argument("--runs", nargs="*", default=None)
    p_train.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p_train.add_argument("--force", action="store_true", help="越过显存空闲检查（GPU 纪律）")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="对 run 的 best_val.ckpt 产出 test_eval.json")
    p_eval.add_argument("--runs", nargs="*", default=None)
    p_eval.add_argument("--split", default="test")
    p_eval.add_argument("--station", type=int, default=0)
    p_eval.add_argument(
        "--n-low", type=int, default=None,
        help="缺省 = 按各 run 的 run_meta.json 的 n_low（E2 的 3e3/1e5 run 需按 run 档位）",
    )
    p_eval.add_argument("--subs", default="0-19")
    p_eval.add_argument("--device", default="cpu")
    p_eval.set_defaults(func=cmd_eval)

    p_gates = sub.add_parser("gates", help="由落盘产物汇总出口门禁")
    p_gates.add_argument("--out", type=Path, default=DEFAULT_RESULTS / "e1_gates.json")
    p_gates.set_defaults(func=cmd_gates)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
