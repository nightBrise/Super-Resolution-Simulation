"""T0-实验1 步骤 2c/3：last.ckpt vs best_val.ckpt 的逐束配对比较（只读）。

配对：同一 run、同一 beam_id、同一 20 个子样本（两边都是 --subs 0-19、station 0、split test），
逐束 err 差 Δ = err(last) − err(best)。Δ < 0 表示末步（训练更充分）更好。

统计：配对 bootstrap（10,000 次，束级重采样，种子经 SeedSequence(20260825) 派生，无裸随机）
给出中位 Δ 的 95% 百分位 CI；Wilcoxon 符号秩检验（scipy，双尾）给 p 值。

对照：best_val.ckpt 由本目录的复算（``*_bestval_eval.json``）与既有 ``results/<run>/test_eval.json``
逐束比较，用于量化本复算与原始评估的确定性一致性（Δ 的基线偏移量）。

产出：``pairing.csv``（20 行）+ 终端摘要。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/home/zhangny/Super-Resolution-Simulation")
RESULTS = REPO / "studies/line2_particle_sr/results"
HERE = Path(__file__).resolve().parent

BOOT_N = 10_000
BOOT_SEED = 20260825

RUNS = (
    [f"e1_joint_s{i}" for i in range(5)]
    + [f"e1_indep_s{i}" for i in range(5)]
    + [f"e2_joint_n3000_s{i}" for i in range(5)]
    + [f"e2_joint_n100000_s{i}" for i in range(5)]
)

FIELDS = [
    "run_id", "arm", "n_low", "n_beams", "best_step", "steps",
    "err_median_best_existing", "err_median_best_rerun", "err_median_last",
    "delta_median", "ci_lo", "ci_hi", "ci_excludes_0", "p_wilcoxon",
    "frac_beams_improved", "n_eff_best", "n_eff_last",
    "ctrl_max_abs_diff", "ctrl_bit_identical",
    "val_nll_at_best", "val_nll_at_last", "steps_saved_frac_of_budget",
]


def arm_of(run_id: str) -> str:
    if run_id.startswith("e1_joint"):
        return "e1_joint"
    if run_id.startswith("e1_indep"):
        return "e1_indep"
    if run_id.startswith("e2_joint_n3000"):
        return "e2_n3000"
    return "e2_n100000"


def beam_errs(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(b["beam_id"]): float(b["err"]) for b in payload["per_beam"]}


def paired_bootstrap_ci(delta: np.ndarray) -> tuple[float, float, float]:
    """束级重采样 bootstrap：返回（点估计中位 Δ, 2.5% 分位, 97.5% 分位）。"""
    rng = np.random.default_rng(np.random.SeedSequence(BOOT_SEED))
    n = delta.size
    idx = rng.integers(0, n, size=(BOOT_N, n))
    medians = np.median(delta[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(np.median(delta)), float(lo), float(hi)


def main() -> int:
    rows = []
    for run_id in RUNS:
        run_dir = RESULTS / run_id
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        existing = json.loads((run_dir / "test_eval.json").read_text(encoding="utf-8"))
        last = json.loads((HERE / f"{run_id}_last_eval.json").read_text(encoding="utf-8"))
        rerun = json.loads((HERE / f"{run_id}_bestval_eval.json").read_text(encoding="utf-8"))

        ids = sorted(beam_errs(run_dir / "test_eval.json"))
        e_existing = beam_errs(run_dir / "test_eval.json")
        e_rerun = beam_errs(HERE / f"{run_id}_bestval_eval.json")
        e_last = beam_errs(HERE / f"{run_id}_last_eval.json")
        assert sorted(e_last) == ids == sorted(e_rerun), f"{run_id}: 束集合不一致"

        base = np.array([e_existing[b] for b in ids])
        base_rerun = np.array([e_rerun[b] for b in ids])
        last_v = np.array([e_last[b] for b in ids])
        delta = last_v - base
        med, lo, hi = paired_bootstrap_ci(delta)
        try:
            p = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
        except ValueError:  # 全零差
            p = float("nan")

        log = [json.loads(line) for line in
               (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = [e["step"] for e in log]
        val_nll = [e["val_nll"] for e in log]
        i_best = steps.index(int(summary["best_step"]))
        ctrl_diff = np.abs(base_rerun - base)

        rows.append({
            "run_id": run_id,
            "arm": arm_of(run_id),
            "n_low": int(meta["n_low"]),
            "n_beams": len(ids),
            "best_step": int(summary["best_step"]),
            "steps": int(summary["steps"]),
            "err_median_best_existing": round(float(existing["err_median"]), 6),
            "err_median_best_rerun": round(float(rerun["err_median"]), 6),
            "err_median_last": round(float(last["err_median"]), 6),
            "delta_median": round(med, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "ci_excludes_0": bool(lo > 0.0 or hi < 0.0),
            "p_wilcoxon": p,
            "frac_beams_improved": round(float(np.mean(delta < 0)), 3),
            "n_eff_best": float(existing["n_eff"]),
            "n_eff_last": float(last["n_eff"]),
            "ctrl_max_abs_diff": float(ctrl_diff.max()),
            "ctrl_bit_identical": bool(np.array_equal(base_rerun, base)),
            "val_nll_at_best": round(float(val_nll[i_best]), 6),
            "val_nll_at_last": round(float(val_nll[-1]), 6),
            "steps_saved_frac_of_budget": round(1.0 - int(summary["steps"]) / 50_000, 4),
        })

    with (HERE / "pairing.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'run':22s} {'best_med':>9s} {'last_med':>9s} {'Δmed':>9s} "
          f"{'CI_lo':>9s} {'CI_hi':>9s} {'sig':>4s} {'p':>9s} {'impr':>5s} {'ctrlΔ':>9s}")
    for r in rows:
        print(f"{r['run_id']:22s} {r['err_median_best_existing']:9.4f} {r['err_median_last']:9.4f} "
              f"{r['delta_median']:+9.4f} {r['ci_lo']:+9.4f} {r['ci_hi']:+9.4f} "
              f"{('Y' if r['ci_excludes_0'] else 'n'):>4s} {r['p_wilcoxon']:9.2e} "
              f"{r['frac_beams_improved']:5.2f} {r['ctrl_max_abs_diff']:9.1e}")

    print()
    for arm in ("e1_joint", "e1_indep", "e2_n3000", "e2_n100000"):
        sub = [r for r in rows if r["arm"] == arm]
        n_sig_neg = sum(r["ci_excludes_0"] and r["delta_median"] < 0 for r in sub)
        n_sig_pos = sum(r["ci_excludes_0"] and r["delta_median"] > 0 for r in sub)
        d = np.array([r["delta_median"] for r in sub])
        print(f"{arm:12s} n={len(sub)} 显著负(last更好) {n_sig_neg}/{len(sub)}  显著正 {n_sig_pos}/{len(sub)}  "
              f"Δmed 中位 {np.median(d):+.4f}  范围 [{d.min():+.4f}, {d.max():+.4f}]")
    all_d = np.array([r["delta_median"] for r in rows])
    print(f"{'ALL':12s} n={len(rows)} Δmed 中位 {np.median(all_d):+.4f}  "
          f"范围 [{all_d.min():+.4f}, {all_d.max():+.4f}]  "
          f"CI 不含 0 全为负的 run 数 {sum(r['ci_excludes_0'] and r['delta_median'] < 0 for r in rows)}/{len(rows)}")
    n_ctrl = sum(r["ctrl_bit_identical"] for r in rows)
    print(f"对照组（best_val 复算 vs 既有 test_eval.json）逐位一致 {n_ctrl}/{len(rows)}；"
          f"最大束级 |Δerr| {max(r['ctrl_max_abs_diff'] for r in rows):.3e}")
    print(f"pairing.csv → {HERE / 'pairing.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
