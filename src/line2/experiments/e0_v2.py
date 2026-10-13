"""v2 E0' 尺子产物（容差线表 + B-const 两站）的生成入口（`50` [S3] C3/C4）。

两件产物与 v2 裸误差–N 曲线同属"纯数据尺子"（不含任何模型输出），按 `50` [S5] C1
须在训练启动前落盘。本模块只负责生成与落盘；"完成时间戳早于任一训练"的顺序约束
由编排器（`50` [S5] C1）强制。

用法：
    python -m src.line2.experiments.e0_v2 tolerance [--data-dir <dir>] [--out <json>]
    python -m src.line2.experiments.e0_v2 bconst    [--data-dir <dir>] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.line2.experiments.tolerance_table import build_payload
from src.line2.metrics.baseline import bconst_records
from src.line2.metrics.tolerance_v2 import V2_TIERS, deviation_table_v2

DEFAULT_DATA_DIR = Path("studies/line2_particle_sr/data/line2_ds_v2")
DEFAULT_RESULTS_DIR = Path("studies/line2_particle_sr/results")
TOLERANCE_FILENAME = "tolerance_v2.json"
M6_READINGS_FILENAME = "tolerance_v2_m6_train.npz"
BCONST_FILENAME = "bconst_v2.json"
BCONST_GROUPS_FILENAME = "bconst_v2_groups.npz"
STATION = 0
K = 20

__all__ = [
    "build_tolerance_product",
    "build_bconst_product",
    "main",
    "DEFAULT_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "TOLERANCE_FILENAME",
    "M6_READINGS_FILENAME",
    "BCONST_FILENAME",
    "BCONST_GROUPS_FILENAME",
]


def _code_version() -> str:
    from src.line2.experiments.e0 import code_version

    return code_version()


def build_tolerance_product(
    data_dir: str | Path, *, out: Path, readings_out: Path | None = None,
    progress_every: int | None = None,
) -> dict:
    """v2 容差线产物（13 项 × 3 训练档 × 站 0，M6 去偏）→ JSON + M6 原始读数 npz。

    ``dev_table`` 沿用 `20` [S4] 的落盘格式（字符串档位 → 13 维向量），可被
    `gates._load_tolerance` 直接读回；``m6_debias`` 段落盘去偏量与验证判据读数。
    """
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    table = deviation_table_v2(
        data_dir, tiers=V2_TIERS, station=STATION, k=K, progress_every=progress_every
    )
    payload = build_payload(
        table["dev_table"],
        beam_ids=_scope_beams(manifest, table["scope"]),
        tiers=table["tiers"],
        k=table["k"],
        data_dir=data_dir,
        beams_scope="valtest" if table["scope"] == "valtest_pooled" else table["scope"],
        code_version=_code_version(),
        data_version=manifest.get("data_version"),
        spec_version=manifest.get("spec_version"),
        quantile=table["quantile"],
        scope=table["scope"],
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_s=time.time() - t0,
    )
    payload["stations"] = [int(table["station"])]
    payload["m6_debias"] = {
        "method": "split_half_train_subsamples_median_remove",
        "seed": int(table["seed"]),
        "per_tier": {str(t): rec for t, rec in sorted(table["m6_debias"].items())},
        "raw_readings": str(readings_out) if readings_out is not None else None,
        "note": "`50` [S3] C3：M6 行取去偏口径（train 估计半区中位数扣除，另一半验证）",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if readings_out is not None:
        readings_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            readings_out,
            **{
                f"tier_{int(t)}": np.asarray(v, dtype=np.float64)
                for t, v in table["m6_signed_train"].items()
            },
        )
    return payload


def _scope_beams(manifest: dict, scope: str) -> list[int]:
    """scope 串 → 束号列表（`valtest_pooled` → val+test）。"""
    if scope.startswith("valtest"):
        return [int(b) for s in ("val", "test") for b in manifest["splits"][s]]
    split = scope.split("_")[0]
    return [int(b) for b in manifest["splits"][split]]


def build_bconst_product(
    data_dir: str | Path, *, out: Path, groups_out: Path | None = None
) -> dict:
    """B-const 两站产物（均值图 npz + 逐束误差 JSON）（`50` [S3] C4）。"""
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    built = bconst_records(data_dir, stations=(0, 1), eval_split="test")
    payload = dict(built["records"])
    payload["started_at"] = started_at
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["code_version"] = _code_version()
    payload["groups_file"] = str(groups_out) if groups_out is not None else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if groups_out is not None:
        groups_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            groups_out,
            **{f"station{int(s)}": np.asarray(g, dtype=np.float64)
               for s, g in built["groups"].items()},
        )
    return payload


def _run_tolerance(args: argparse.Namespace) -> int:
    out = args.out or args.results_dir / TOLERANCE_FILENAME
    readings = args.readings_out or args.results_dir / M6_READINGS_FILENAME
    payload = build_tolerance_product(
        args.data_dir, out=out, readings_out=readings, progress_every=args.progress_every
    )
    print(json.dumps(
        {
            "out": str(out),
            "readings_out": str(readings),
            "n_beams": payload["n_beams"],
            "elapsed_s": payload["elapsed_s"],
            "m6_debias": {
                tier: {
                    "bias": rec["bias"],
                    "raw_abs_median_over_sigma": rec["raw_abs_median_over_sigma"],
                    "val_abs_median_over_sigma": rec["val_abs_median_over_sigma"],
                    "val_p95_over_1p96sigma": rec["val_p95_over_1p96sigma"],
                    "is_noise_line": rec["is_noise_line"],
                }
                for tier, rec in payload["m6_debias"]["per_tier"].items()
            },
        },
        ensure_ascii=False, indent=2,
    ))
    return 0


def _run_bconst(args: argparse.Namespace) -> int:
    out = args.out or args.results_dir / BCONST_FILENAME
    groups = args.groups_out or args.results_dir / BCONST_GROUPS_FILENAME
    payload = build_bconst_product(args.data_dir, out=out, groups_out=groups)
    print(json.dumps(
        {
            "out": str(out),
            "groups_out": str(groups),
            "err_median": {s: r["err_median"] for s, r in payload["station"].items()},
            "elapsed_s": payload["elapsed_s"],
        },
        ensure_ascii=False, indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 E0' 尺子产物（`50` [S3] C3/C4）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tol = sub.add_parser("tolerance", help="v2 容差线表（M6 分半去偏）")
    p_tol.add_argument("--out", type=Path, default=None)
    p_tol.add_argument("--readings-out", type=Path, default=None)
    p_tol.add_argument("--progress-every", type=int, default=50)

    p_bc = sub.add_parser("bconst", help="B-const 两站均值图与逐束误差")
    p_bc.add_argument("--out", type=Path, default=None)
    p_bc.add_argument("--groups-out", type=Path, default=None)

    args = parser.parse_args(argv)
    return _run_tolerance(args) if args.cmd == "tolerance" else _run_bconst(args)


if __name__ == "__main__":
    raise SystemExit(main())
