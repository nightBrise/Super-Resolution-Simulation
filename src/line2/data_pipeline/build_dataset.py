"""CLI：L1 全链数据集构建与 v1 重传对账（spec line2/10、line2/50 [S2]）。

用法：

- v1 构建：``python -m src.line2.data_pipeline.build_dataset --out studies/line2_particle_sr/data/line2_ds_v1``
- v2 构建（公共域定标）：``python -m src.line2.data_pipeline.build_dataset
  --data-version line2_ds_v2 --out studies/line2_particle_sr/data/line2_ds_v2``
- C1 重传对账（不构建，落盘 ``reconciliation_v1.json``）：同上加 ``--reconcile-only``；
  存在计数差或 bin_edges 不一致时以非零码退出。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor import read_anchor_particles
from .dataset import MAX_CLAMP_FRAC_V2, V2_DATA_VERSION, build_dataset, reconcile_particles_with_v1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1/1.5 数据集构建（spec 10 [S1]-[S5]、50 [S2]）")
    parser.add_argument("--anchor-dir", type=Path,
                        default=Path("studies/line2_particle_sr/data/anchor"))
    parser.add_argument("--out", type=Path, default=Path("studies/line2_particle_sr/data/line2_ds_v1"))
    parser.add_argument("--n-beams", type=int, default=1000)
    parser.add_argument("--master-seed", type=int, default=20260913)
    parser.add_argument("--data-version", default="line2_ds_v1")
    parser.add_argument("--v1-dir", type=Path,
                        default=Path("studies/line2_particle_sr/data/line2_ds_v1"))
    parser.add_argument("--max-clamp-frac", type=float, default=None,
                        help=f"v2 夹取占比上限（默认 {MAX_CLAMP_FRAC_V2}，`50` [S2] C6 硬线）；生产口径应保持默认，覆盖仅供测试/诊断"
    parser.add_argument("--reconcile-only", action="store_true",
                        help="只做 C1 重传对账（不构建数据集）")
    args = parser.parse_args(argv)

    anchor_table = read_anchor_particles(args.anchor_dir / "injector.sdds")
    if args.reconcile_only:
        report = reconcile_particles_with_v1(anchor_table, args.v1_dir)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "reconciliation_v1.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[reconcile] {report['n_beams']} 束 × 2 站重传对账："
            f"{json.dumps(report['per_station'], ensure_ascii=False)}"
            f"，不一致 {len(report['mismatches'])} 条 → {args.out / 'reconciliation_v1.json'}",
            flush=True,
        )
        return 1 if report["mismatches"] else 0

    out = build_dataset(
        args.out, anchor_table, n_beams=args.n_beams,
        master_seed=args.master_seed, data_version=args.data_version,
        max_clamp_frac=args.max_clamp_frac,
    )
    version = args.data_version
    if version != V2_DATA_VERSION:
        print(f"数据集落盘：{out}（{args.n_beams} 束，逐束自定标口径）")
    else:
        print(f"数据集落盘：{out}（{args.n_beams} 束，公共域口径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
