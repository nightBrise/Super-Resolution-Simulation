"""T0-实验1 步骤 3：合并 trajectory.csv + pairing.csv → summary_table.csv（20 行汇总表）。

列：err_median(best_val 冻结选型点) / err_median(last) / Δ / 95% CI / p / val_nll@best_step /
val_nll@last / steps（+ 臂、档位、束数、改进束比例、N_eff、对照一致性）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = Path("/home/zhangny/Super-Resolution-Simulation")

COLS = [
    "run_id", "arm", "n_low", "n_beams", "steps", "best_step",
    "err_median_best", "err_median_last", "delta_median", "ci_lo", "ci_hi",
    "ci_excludes_0", "p_wilcoxon", "frac_beams_improved",
    "val_nll_at_best", "val_nll_at_last", "steps_saved_frac_of_budget",
    "n_eff_best", "n_eff_last", "ckpt_step_matches_summary", "ctrl_bit_identical",
]


def main() -> int:
    traj = {r["run_id"]: r for r in csv.DictReader((HERE / "trajectory.csv").open(encoding="utf-8"))}
    pair = list(csv.DictReader((HERE / "pairing.csv").open(encoding="utf-8")))
    rows = []
    for p in pair:
        t = traj[p["run_id"]]
        ckpt = torch.load(
            REPO / "studies/line2_particle_sr/results" / p["run_id"] / "last.ckpt",
            map_location="cpu", weights_only=False,
        )
        rows.append({
            "run_id": p["run_id"], "arm": p["arm"], "n_low": p["n_low"], "n_beams": p["n_beams"],
            "steps": p["steps"], "best_step": p["best_step"],
            "err_median_best": p["err_median_best_existing"], "err_median_last": p["err_median_last"],
            "delta_median": p["delta_median"], "ci_lo": p["ci_lo"], "ci_hi": p["ci_hi"],
            "ci_excludes_0": p["ci_excludes_0"], "p_wilcoxon": p["p_wilcoxon"],
            "frac_beams_improved": p["frac_beams_improved"],
            "val_nll_at_best": p["val_nll_at_best"], "val_nll_at_last": p["val_nll_at_last"],
            "steps_saved_frac_of_budget": p["steps_saved_frac_of_budget"],
            "n_eff_best": p["n_eff_best"], "n_eff_last": p["n_eff_last"],
            "ckpt_step_matches_summary": int(ckpt["step"]) == int(p["steps"]),
            "ctrl_bit_identical": p["ctrl_bit_identical"],
        })
    with (HERE / "summary_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(r["ckpt_step_matches_summary"] for r in rows)
    print(f"summary_table.csv → {HERE / 'summary_table.csv'}（{len(rows)} 行）")
    print(f"last.ckpt 内嵌 step == summary.steps：{n_ok}/{len(rows)}；"
          f"对照逐位一致：{sum(r['ctrl_bit_identical'] == 'True' for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
