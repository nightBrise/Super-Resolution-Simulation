"""L4 [S1] E0 基线 CLI：裸误差–N 曲线、互差地板、参照噪声地板（spec line2/20 [S2]）。

门禁口径 = **跨 val+test 束**（`20` [S2]："跨 val/test 束中位数与 IQR"），两站、
五档 × k = 20；`--beams val` 供训练期诊断口径（`train.py` 的 val N_eff 插值基底）。

产物：
- `--out`（默认 `studies/line2_particle_sr/results/e0_curve.json`）——曲线完整 dump
  （含原始逐样本值，`20` [S2] C1 要求可复核）；
- `--out` 同目录同名 `.meta.json`——`started_at`/`completed_at` 与溯源三元组，
  是 `40` [S1] C5"E0 完成时间戳早于 E1 任一训练启动"的落盘证据。

用法：
    python -m src.line2.experiments.e0 --data-dir <数据集目录> [--out <json>] [--beams valtest|val|test]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src.line2.metrics.n_eff import error_n_curve

DEFAULT_DATA_DIR = Path("studies/line2_particle_sr/data/line2_ds_v1")
DEFAULT_OUT = Path("studies/line2_particle_sr/results/e0_curve.json")

__all__ = ["main", "DEFAULT_DATA_DIR", "DEFAULT_OUT"]


def code_version() -> str:
    """生成时 git HEAD 完整 40 位 hash（`00` GC7）。"""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 E0 基线（spec 40 [S1]）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--beams", choices=("valtest", "val", "test"), default="valtest")
    parser.add_argument("--stations", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tiers", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000])
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args(argv)

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.beams == "valtest":
        beam_ids = [int(b) for s in ("val", "test") for b in manifest["splits"][s]]
    else:
        beam_ids = [int(b) for b in manifest["splits"][args.beams]]

    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    curve = error_n_curve(
        args.data_dir,
        beam_ids=beam_ids,
        stations=tuple(args.stations),
        tiers=tuple(args.tiers),
        k=args.k,
        dump=args.out,
        progress_every=25,
    )
    elapsed = round(time.time() - t0, 1)
    meta = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "beams_scope": args.beams,
        "n_beams": len(beam_ids),
        "stations": list(args.stations),
        "tiers": list(args.tiers),
        "k": args.k,
        "data_dir": str(args.data_dir),
        "data_version": manifest.get("data_version"),
        "code_version": code_version(),
        "spec_version": manifest.get("spec_version"),
        "out": str(args.out),
        "reference_noise_floor": curve.reference_noise_floor,
        "points": [
            {"n_low": p.n_low, "median": p.median, "iqr_lo": p.iqr_lo, "iqr_hi": p.iqr_hi}
            for p in curve.points
        ],
        "mutual_floor": {str(t): v for t, v in curve.mutual_floor.items()},
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
