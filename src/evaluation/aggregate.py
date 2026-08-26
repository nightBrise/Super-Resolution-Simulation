"""跨方案评估聚合（G2/G3 判定输入，70 [S7]，60 [S15] 15.7 配套）。

`evaluate.py` 每次评估一个 config（一个方案），per-run metrics.csv 只有单一方案
行；而 `build_summary` 的先验增益要求同一 metrics.csv 内有跨方案配对行
（70 [S7.2] 配对 Wilcoxon + bootstrap CI）。本工具把同一种子/同划分下三个方案的
run 级 metrics.csv 合并（按 (sample_id, scheme)），重建合并 metrics.csv +
summary.json（含 prior_gain / three_class / one_veto），作为 G2/G3 判定输入。

用法：:

    python -m src.evaluation.aggregate --runs <A_dir>,<B_dir>,<C_dir> \
        --split test_id --out results/EXP-02_summary/seed0

说明：bootstrap 随机源由 config.master_seed 派生（与 evaluate 同分支），
三个 run 的 master_seed 相同 → 聚合结果可复现；seed<N> 聚合目录的
summary.json 即为该种子该划分的判定依据（80 [S8] ci95_per_seed 语义）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.evaluate import (
    _merge_rows,
    _read_metrics,
    _write_metrics,
    build_summary,
)
from src.utils.config_utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def aggregate_runs(
    run_dirs: list[Path],
    split: str,
    out_dir: Path,
) -> dict:
    """合并 run 级 metrics.csv 行并重建 summary.json；返回 summary 字典。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"{run_dir} 无 metrics.csv（先跑 evaluate）")
        merged = _merge_rows(merged, _read_metrics(metrics_path))

    # 过滤出目标 split 的行（run 级 metrics.csv 可能含多 split，70 [S7.1] C7 分列）
    split_rows = [r for r in merged if r.get("split") == split]
    if not split_rows:
        raise ValueError(f"{split} 无行（run 级 metrics.csv 需含 split={split} 的行）")

    _write_metrics(out_dir / "metrics.csv", split_rows)

    config = load_config(run_dirs[0] / "config.yaml")
    summary = build_summary(split_rows, config, split, baseline=None)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.aggregate",
        description="跨方案评估聚合（G2/G3 判定输入）",
    )
    parser.add_argument("--runs", required=True, help="run 目录逗号分隔（A,B,C 三个方案）")
    parser.add_argument("--split", required=True, choices=["test_id", "test_pb", "test_ood", "exp03", "exp04"])
    parser.add_argument("--out", required=True, help="聚合输出目录（如 results/EXP-02_summary/seed0）")
    args = parser.parse_args(argv)

    run_dirs = [Path(p.strip()) for p in args.runs.split(",") if p.strip()]
    if len(run_dirs) < 2:
        print("--runs 至少两个方案目录（配对增益需要跨方案行）", file=sys.stderr)
        return 1

    summary = aggregate_runs(run_dirs, args.split, Path(args.out))
    gains = {
        k: v.get("M_A_minus_M_B", {}).get("verdict", "n/a")
        for k, v in summary.get("prior_gain", {}).items()
    }
    print(f"[aggregate] split={args.split} -> {args.out}")
    print(f"  schemes={sorted({r.get('scheme') for r in _read_metrics(Path(args.out) / 'metrics.csv')})}")
    print(f"  prior_gain verdicts: {gains}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
