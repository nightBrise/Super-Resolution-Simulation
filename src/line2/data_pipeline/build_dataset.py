"""CLI：L1 全链数据集构建（spec line2/10）。

用法：``python -m src.line2.data_pipeline.build_dataset --n-beams 1000 --out data/line2_ds_v1``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .anchor import read_anchor_particles
from .dataset import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1 数据集构建（spec 10 [S1]-[S5]）")
    parser.add_argument("--anchor-dir", type=Path,
                        default=Path("studies/line2_particle_sr/data/anchor"))
    parser.add_argument("--out", type=Path, default=Path("studies/line2_particle_sr/data/line2_ds_v1"))
    parser.add_argument("--n-beams", type=int, default=1000)
    parser.add_argument("--master-seed", type=int, default=20260913)
    parser.add_argument("--data-version", default="line2_ds_v1")
    args = parser.parse_args(argv)

    from .anchor import read_anchor_particles
    from .dataset import build_dataset

    anchor_table = read_anchor_particles(args.anchor_dir / "injector.sdds")
    out = build_dataset(
        args.out, anchor_table, n_beams=args.n_beams,
        master_seed=args.master_seed, data_version=args.data_version,
    )
    print(f"数据集落盘：{out}（{args.n_beams} 束）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
