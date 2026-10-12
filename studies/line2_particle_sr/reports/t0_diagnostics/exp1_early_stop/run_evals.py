"""T0-实验1 评估驱动：对 20 个 Phase 1 run 各评 last.ckpt 与 best_val.ckpt（对照）。

调用形态逐参数复刻 `run_e1.cmd_eval` 的内部 argv（即 phase1_chain.sh 里
`run_e1 eval --station 0 --subs 0-19 --device cuda:0` 展开后的
`evaluate.main` 参数表），只改 `--ckpt` 与 `--out`：
- ckpt: last.ckpt（实验组）/ best_val.ckpt（对照组）
- out: 本目录
环境变量与 run_e1._train_env 一致（CUDA_VISIBLE_DEVICES / CUBLAS_WORKSPACE_CONFIG /
PYTHONHASHSEED），device=cuda:0。

对 results/ 与 data/ 只读：所有 --out 落在本目录。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/home/zhangny/Super-Resolution-Simulation")
RESULTS = REPO / "studies/line2_particle_sr/results"
DATA = REPO / "studies/line2_particle_sr/data/line2_ds_v1"
CURVE = RESULTS / "e0_curve.json"
OUTDIR = REPO / "studies/line2_particle_sr/reports/t0_diagnostics/exp1_early_stop"

RUNS = [
    f"e1_joint_s{i}" for i in range(5)
] + [
    f"e1_indep_s{i}" for i in range(5)
] + [
    f"e2_joint_n3000_s{i}" for i in range(5)
] + [
    f"e2_joint_n100000_s{i}" for i in range(5)
]

#: (kind → ckpt 文件名后缀, 输出文件名后缀) —— 实验组与对照组各评一次
KINDS = {"last": ("last.ckpt", "last_eval.json"), "bestval": ("best_val.ckpt", "bestval_eval.json")}


def eval_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"
    return env


def one(run_id: str, kind: str) -> dict:
    ckpt_name, suffix = KINDS[kind]
    meta = json.loads((RESULTS / run_id / "run_meta.json").read_text(encoding="utf-8"))
    n_low = int(meta["n_low"])
    ckpt = RESULTS / run_id / ckpt_name
    out = OUTDIR / f"{run_id}_{suffix}"
    log = OUTDIR / f"{run_id}_{kind}.log"
    argv = [
        sys.executable, "-u", "-m", "src.line2.experiments.evaluate",
        "--ckpt", str(ckpt), "--data-dir", str(DATA), "--split", "test",
        "--station", "0", "--n-low", str(n_low), "--subs", "0-19",
        "--curve", str(CURVE), "--out", str(out), "--device", "cuda:0",
    ]
    t0 = time.time()
    with log.open("wb") as fh:
        rc = subprocess.run(argv, cwd=REPO, env=eval_env(), stdout=fh, stderr=subprocess.STDOUT).returncode
    return {
        "run_id": run_id, "kind": kind, "n_low": n_low, "returncode": rc,
        "elapsed_s": round(time.time() - t0, 1), "argv": argv, "out": str(out),
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(r, k) for r in RUNS for k in KINDS]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("EVAL_WORKERS", "4"))) as pool:
        for res in pool.map(lambda j: one(*j), jobs):
            results.append(res)
            print(
                f"[{res['run_id']:22s} {res['kind']:8s}] rc={res['returncode']} "
                f"{res['elapsed_s']:7.1f}s n_low={res['n_low']}",
                flush=True,
            )
    (OUTDIR / "eval_dispatch.json").write_text(
        json.dumps({"jobs": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [r for r in results if r["returncode"] != 0]
    print(f"完成 {len(results)} 次评估，失败 {len(failed)}: {[r['run_id'] + '/' + r['kind'] for r in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
