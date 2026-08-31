"""M4 G2 判定综合分析（临时脚本，2026-08-28 生成，EXP-02 六 run 评估后）。

计算：
1. G1(b) 零学习基线：A 方案 d_i = e_high_mask_lup − e_high_mask（L_up 基线误差 − A 模型误差），
   正值 = A 优于 L_up；逐 seed + 合并 bootstrap CI + Wilcoxon。
2. G2 跨种子合并（concatenated）CI：对 A−B / A−C / B−C 三对对比，拼接两 seed 配对差
   重算 bootstrap 95% CI 与三分类（70 [S7] 297 行：报告同时给出逐种子与合并 CI）。
3. 汇总判定表输出（test_id 主判定）。

注意：本脚本结果不修改任何研究产物，只输出诊断信息；判定由 70/80 协议决定。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import bootstrap_ci, paired_wilcoxon, three_class

ROOT = Path("/home/zhangny/Super-Resolution-Simulation")
SEEDS = [0, 1]


def load_test_id_rows(seed: int) -> dict[str, dict[str, float]]:
    """seedN/metrics.csv（test_id 聚合）→ {scheme: {sample_id: row}}。"""
    path = ROOT / f"results/EXP-02_summary/seed{seed}/metrics.csv"
    rows: dict[str, dict[str, dict]] = {s: {} for s in "ABC"}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["split"] != "test_id":
                continue
            row = {}
            for k, v in r.items():
                if k in ("sample_id", "split", "scheme", "peak_positions", "R_E_class", "F_i"):
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            rows[r["scheme"]][r["sample_id"]] = row
    return rows


def gain_ci(seed_rows: dict[str, dict[str, dict]], metric: str, y: str, x: str, n_boot: int = 10000):
    """配对差 d_i = M_y(i) − M_x(i) 的统计。y 与 x 为方案字母（A/B/C）。"""
    common = sorted(set(seed_rows[y]) & set(seed_rows[x]))
    d = np.array([seed_rows[y][sid][metric] - seed_rows[x][sid][metric] for sid in common])
    lo, hi = bootstrap_ci(d, n_boot=n_boot)
    return {
        "n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
        "ci95": [lo, hi], "verdict": three_class((lo, hi)),
        "wilcoxon_p": paired_wilcoxon(d),
    }


def main() -> None:
    out = {}
    # ---------- 1. G1(b)：A vs L_up 零学习基线 ----------
    g1b = {}
    for seed in SEEDS:
        rows = load_test_id_rows(seed)
        common = sorted(set(rows["A"]) )
        d = np.array([rows["A"][sid]["e_high_mask_lup"] - rows["A"][sid]["e_high_mask"] for sid in common])
        lo, hi = bootstrap_ci(d, n_boot=10000)
        g1b[f"seed{seed}"] = {
            "n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [lo, hi], "verdict": three_class((lo, hi)),
            "wilcoxon_p": paired_wilcoxon(d),
        }
    # 合并（两 seed 拼接）
    all_d = []
    for seed in SEEDS:
        rows = load_test_id_rows(seed)
        all_d += [rows["A"][sid]["e_high_mask_lup"] - rows["A"][sid]["e_high_mask"] for sid in rows["A"]]
    all_d = np.array(all_d)
    lo, hi = bootstrap_ci(all_d, n_boot=10000)
    g1b["merged"] = {
        "n": len(all_d), "mean": float(all_d.mean()), "median": float(np.median(all_d)),
        "ci95": [lo, hi], "verdict": three_class((lo, hi)),
        "wilcoxon_p": paired_wilcoxon(all_d),
    }
    out["G1b_A_vs_Lup"] = g1b

    # ---------- 2. G2 三对对比：逐 seed + 合并 ----------
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for y, x in pairs:
        key = f"M_{y}_minus_M_{x}"
        per_seed = {}
        dlist = []
        for seed in SEEDS:
            rows = load_test_id_rows(seed)
            res = gain_ci(rows, "e_high_mask", y, x)
            per_seed[f"seed{seed}"] = res
            common = sorted(set(rows[y]) & set(rows[x]))
            dlist += [rows[y][sid]["e_high_mask"] - rows[x][sid]["e_high_mask"] for sid in common]
        d = np.array(dlist)
        lo, hi = bootstrap_ci(d, n_boot=10000)
        merged = {
            "n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [lo, hi], "verdict": three_class((lo, hi)),
            "wilcoxon_p": paired_wilcoxon(d),
        }
        out[key] = {"per_seed": per_seed, "merged": merged,
                    "consistent": all(v["verdict"] == per_seed["seed0"]["verdict"] for v in per_seed.values())}

    print(json.dumps(out, indent=2, ensure_ascii=False))

    # ---------- 3. 可读汇总 ----------
    print("\n===== 汇总（test_id 主判定） =====")
    print("--- G1(b)：A vs L_up（正值 = A 优于零学习基线） ---")
    for k, v in g1b.items():
        print(f"  {k}: mean={v['mean']:.6f} ci95=[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] {v['verdict']} p={v['wilcoxon_p']:.3g}")
    print("--- G2 主指标三分类（significant_positive = y 误差>x，即 x 更优） ---")
    for y, x in pairs:
        key = f"M_{y}_minus_M_{x}"
        v = out[key]
        s0, s1 = v["per_seed"]["seed0"], v["per_seed"]["seed1"]
        m = v["merged"]
        print(f"  {y}-{x}: seed0 {s0['verdict']:20s} seed1 {s1['verdict']:20s} | "
              f"merged {m['verdict']:20s} mean={m['mean']:.6f} ci=[{m['ci95'][0]:.6f},{m['ci95'][1]:.6f}] "
              f"consistent={v['consistent']}")


if __name__ == "__main__":
    main()
