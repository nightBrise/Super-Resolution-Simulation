"""T0-实验1 步骤 2a：train_log.jsonl 的 val_nll 轨迹 vs 冻结选型点（只读）。

对每个 run 报告：val_nll 的 argmin step、summary.json 的 best_step、末步 step 三者关系；
val_nll 轨迹是否近似单调下降；选型点与末步的 val_nll / train_nll 差。

产出：``trajectory.csv``（20 行）+ 终端摘要。只读 results/，不写任何 run 目录。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/zhangny/Super-Resolution-Simulation")
RESULTS = REPO / "studies/line2_particle_sr/results"
HERE = Path(__file__).resolve().parent

RUNS = (
    [f"e1_joint_s{i}" for i in range(5)]
    + [f"e1_indep_s{i}" for i in range(5)]
    + [f"e2_joint_n3000_s{i}" for i in range(5)]
    + [f"e2_joint_n100000_s{i}" for i in range(5)]
)

FIELDS = [
    "run_id", "arm", "n_low", "steps", "best_step", "val_nll_argmin_step",
    "argmin_is_last", "n_evals", "monotone_frac",
    "val_nll_at_best", "val_nll_at_last", "val_nll_drop_abs", "val_nll_drop_frac",
    "train_nll_at_best", "train_nll_at_last", "train_nll_drop_frac",
    "n_eff_saturated_from_step",
]


def arm_of(run_id: str) -> str:
    if run_id.startswith("e1_joint"):
        return "e1_joint"
    if run_id.startswith("e1_indep"):
        return "e1_indep"
    if run_id.startswith("e2_joint_n3000"):
        return "e2_n3000"
    return "e2_n100000"


def main() -> int:
    rows = []
    for run_id in RUNS:
        run_dir = RESULTS / run_id
        log = [json.loads(line) for line in
               (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        steps = np.array([e["step"] for e in log], dtype=int)
        val_nll = np.array([e["val_nll"] for e in log], dtype=float)
        train_nll = np.array([e["train_nll"] for e in log], dtype=float)
        val_neff = np.array([e["val_n_eff"] for e in log], dtype=float)

        best_step = int(summary["best_step"])
        steps_last = int(summary["steps"])
        argmin_step = int(steps[int(np.argmin(val_nll))])
        i_best = int(np.argmax(steps == best_step))
        monotone = float(np.mean(np.diff(val_nll) < 0)) if len(val_nll) > 1 else float("nan")
        sat = steps[val_neff >= 100000.0]
        rows.append({
            "run_id": run_id,
            "arm": arm_of(run_id),
            "n_low": int(json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["n_low"]),
            "steps": steps_last,
            "best_step": best_step,
            "val_nll_argmin_step": argmin_step,
            "argmin_is_last": bool(argmin_step == steps_last),
            "n_evals": int(len(log)),
            "monotone_frac": round(monotone, 3),
            "val_nll_at_best": round(float(val_nll[i_best]), 6),
            "val_nll_at_last": round(float(val_nll[-1]), 6),
            "val_nll_drop_abs": round(float(val_nll[-1] - val_nll[i_best]), 6),
            "val_nll_drop_frac": round(float((val_nll[-1] - val_nll[i_best]) / val_nll[i_best]), 4),
            "train_nll_at_best": round(float(train_nll[i_best]), 6),
            "train_nll_at_last": round(float(train_nll[-1]), 6),
            "train_nll_drop_frac": round(float((train_nll[-1] - train_nll[i_best]) / train_nll[i_best]), 4),
            "n_eff_saturated_from_step": int(sat[0]) if len(sat) else -1,
        })

    with (HERE / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'run':22s} {'best':>6s} {'steps':>6s} {'nllarg':>7s} {'argmin==last':>13s} "
          f"{'mono':>5s} {'val_best':>9s} {'val_last':>9s} {'drop%':>7s} {'neff_sat@':>9s}")
    for r in rows:
        print(f"{r['run_id']:22s} {r['best_step']:6d} {r['steps']:6d} {r['val_nll_argmin_step']:7d} "
              f"{str(r['argmin_is_last']):>13s} {r['monotone_frac']:5.2f} {r['val_nll_at_best']:9.4f} "
              f"{r['val_nll_at_last']:9.4f} {100 * r['val_nll_drop_frac']:6.1f}% "
              f"{r['n_eff_saturated_from_step']:9d}")
    n_argmin_last = sum(r["argmin_is_last"] for r in rows)
    drops = np.array([r["val_nll_drop_frac"] for r in rows])
    print(f"\nargmin(val_nll)==末步 的 run 数：{n_argmin_last}/{len(rows)}；"
          f"val_nll 相对降幅 中位 {100 * np.median(drops):.1f}% —— "
          f"范围 [{100 * drops.min():.1f}%, {100 * drops.max():.1f}%]")
    print(f"末步/预算 步数：{sorted({r['steps'] for r in rows})}（预算 50000）")
    print(f"trajectory.csv → {HERE / 'trajectory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
