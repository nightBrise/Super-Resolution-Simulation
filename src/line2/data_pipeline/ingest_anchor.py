"""CLI：L1 [S1] 锚点入库（复制 → MD5 校验 → twiss.json/manifest.json）。

用法：``python -m src.line2.data_pipeline.ingest_anchor [--source <sdds>] [--anchor-dir <dir>]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor import ingest_anchor

_DEFAULT_SOURCE = Path.home() / "pyaml-main/examples/SXFEL/elegant_files/injector.sdds"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 锚点入库与元数据提取（spec 10 [S1]）")
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="锚点源 SDDS 路径")
    parser.add_argument(
        "--anchor-dir", type=Path, default=Path("studies/line2_particle_sr/data/anchor")
    )
    parser.add_argument("--master-seed", type=int, default=20260913)
    args = parser.parse_args(argv)

    result = ingest_anchor(args.anchor_dir, args.source, args.master_seed)
    twiss = result["twiss"]
    print(
        json.dumps(
            {
                "row_count": result["row_count"],
                "charge_c": twiss["charge_c"],
                "beta_gamma": twiss["beta_gamma"],
                "emit_n_x": twiss["emit_n"]["x"],
                "emit_n_y": twiss["emit_n"]["y"],
                "sigma_t_s": twiss["sigma_t_s"],
                "written": [
                    str(Path(args.anchor_dir) / "twiss.json"),
                    str(Path(args.anchor_dir) / "manifest.json"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
