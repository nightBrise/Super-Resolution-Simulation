"""`50` [S3] C5 / [S5] C3：个体信息仪器的运行端——ckpt + 数据集 + B-const → 逐 run 读数。

G2 的判定输入是**对齐仪器读数**（``cos_true`` / 其 95% CI / ``drop_frac``），仪器本身
（:func:`src.line2.metrics.alignment.alignment_instrument`）只消费预测图组与束表；本模块
负责把两件事做齐：

- **真输入侧**：该束该子样本的输入图组 → 预测图组；
- **打乱侧**：输入换成 π(束) 的图组后前向（π 由 :func:`scramble_pairs` 给出，固定
  derangement）。打乱后的输入对含 π(束) 的 κ——前向走 :func:`evaluate.predict_group`
  的既有分支，κ 由**输入计数**派生，故这一点自动成立（`50` [S1] 的 κ 口径）。

读数按 `50` [S1] 的聚合口径（逐 (束, 子样本) → 逐束 → 束级统计）由仪器给出，落盘
``e15_align_<run_id>.json``（G2 的取数口，见 :mod:`src.line2.experiments.phase15_gates`）。

用法：
    python -m src.line2.experiments.phase15_align --ckpt <run>/best_val.ckpt \
        --data-dir <line2_ds_v2> --bconst-groups <results>/bconst_v2_groups.npz \
        --n-low 10000 --station 0 --subs 0-19 [--out <json>] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.line2.data_pipeline.dataset import git_head
from src.line2.metrics import ANALYSIS_SEED
from src.line2.metrics.alignment import alignment_instrument, scramble_pairs

from .evaluate import load_model, parse_subs, predict_group

__all__ = [
    "ALIGN_SPLIT",
    "run_alignment",
    "main",
]

#: 对齐仪器的评估段（G2 定义在 test 束上；站 0）
ALIGN_SPLIT = "test"


def run_alignment(
    ckpt: str | Path,
    data_dir: str | Path,
    bconst_groups: str | Path,
    *,
    split: str = ALIGN_SPLIT,
    station: int = 0,
    n_low: int = 10_000,
    subs: str | int | list[int] | None = "0-19",
    device: str = "cpu",
    seed: int = ANALYSIS_SEED,
    out: str | Path | None = None,
) -> dict:
    """单 run 的对齐仪器读数（真输入 + π(束) 打乱侧各前向一次）。

    Returns:
        ``{run_id, variant, seed, split, station, n_low, subs, n_beams, ckpt,
        bconst_groups, code_version, ...仪器读数}``，并落盘（``out`` 缺省
        ``<ckpt 父目录>/e15_align_<run_id>.json``）。
    """
    ckpt = Path(ckpt)
    data_dir = Path(data_dir)
    bconst_groups = Path(bconst_groups)
    if not bconst_groups.is_file():
        raise FileNotFoundError(f"B-const 组图缺失：{bconst_groups}（`50` [S3] C4 的产物）")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if split not in manifest["splits"]:
        raise ValueError(f"{split!r} 不在 manifest 切分中：{sorted(manifest['splits'])}")
    beam_ids = [int(b) for b in manifest["splits"][split]]
    sub_list = parse_subs(subs)
    station = int(station)
    n_low = int(n_low)

    model, cfg = load_model(ckpt)
    if "seed" not in cfg:
        raise ValueError(f"{ckpt}：config 缺 seed，落盘读数的种子字段无来源")
    variant = str(cfg.get("variant", "joint"))
    prior = np.asarray(np.load(bconst_groups)[f"station{station}"], dtype=np.float64)
    if prior.ndim != 3 or prior.shape[0] != 6:
        raise ValueError(
            f"{bconst_groups} 的 station{station} 形状 {prior.shape} ≠ (6, H, W)"
            "（B-const 组图，`50` [S3] C4；仪器要求与数据集图组同形状）"
        )

    pairs = scramble_pairs(beam_ids, seed=seed)
    targets: dict[int, np.ndarray] = {}
    predictions: dict[int, np.ndarray] = {}
    scrambled: dict[int, np.ndarray] = {}
    with h5py.File(data_dir / f"{split}.h5", "r") as h5:
        for beam_id in beam_ids:
            stg = h5[f"beam_{beam_id:04d}"][f"station{station}"]
            targets[beam_id] = stg["target_full"][...].astype(np.float64)
            true_rows, scr_rows = [], []
            for sub_idx in sub_list:
                name = f"tier_{n_low:06d}_sub_{sub_idx:02d}"
                if name not in stg:
                    raise KeyError(f"{split}.h5 beam_{beam_id:04d}/station{station} 缺 {name}")
                dset = stg[name]
                edges = np.asarray(dset.attrs["bin_edges"], dtype=np.float64)
                true_rows.append(
                    predict_group(model, cfg, dset[...], edges, n_low, device=device)
                )
                other = h5[f"beam_{pairs[beam_id]:04d}"][f"station{station}"][name]
                scr_rows.append(
                    predict_group(
                        model, cfg, other[...],
                        np.asarray(other.attrs["bin_edges"], dtype=np.float64), n_low,
                        device=device,
                    )
                )
            predictions[beam_id] = np.stack(true_rows)
            scrambled[beam_id] = np.stack(scr_rows)

    readings = alignment_instrument(
        predictions, scrambled, targets, prior, seed=seed
    )
    payload = {
        "run_id": ckpt.parent.name,
        "variant": variant,
        "seed": int(cfg["seed"]),
        "split": split,
        "station": station,
        "n_low": n_low,
        "subs": [int(s) for s in sub_list],
        "n_beams": int(readings["n_beams"]),
        "n_subs": int(readings["n_subs"]),
        "ckpt": str(ckpt),
        "bconst_groups": str(bconst_groups),
        "data_dir": str(data_dir),
        "code_version": git_head(),
        "scramble": {
            "rule": "fixed_derangement(SeedSequence(20260825))",
            "pairing": {str(b): int(p) for b, p in sorted(readings["pairing"].items())},
        },
        **{k: readings[k] for k in (
            "cos_true", "cos_scr", "drop_frac", "disp_ratio",
            "cos_true_ci95", "cos_scr_ci95", "bootstrap",
        )},
        "cos_true_per_beam": {str(b): float(v) for b, v in readings["cos_true_per_beam"].items()},
        "cos_scr_per_beam": {str(b): float(v) for b, v in readings["cos_scr_per_beam"].items()},
    }
    out_path = Path(out) if out is not None else ckpt.parent / f"e15_align_{ckpt.parent.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["align_path"] = str(out_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1.5 个体信息仪器运行（spec 50 [S5] C3）")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bconst-groups", type=Path, required=True)
    parser.add_argument("--split", default=ALIGN_SPLIT)
    parser.add_argument("--station", type=int, default=0)
    parser.add_argument("--n-low", type=int, default=10_000)
    parser.add_argument("--subs", default="0-19")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_alignment(
        args.ckpt, args.data_dir, args.bconst_groups, split=args.split, station=args.station,
        n_low=args.n_low, subs=args.subs, device=args.device, out=args.out,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"], "seed": payload["seed"],
                "n_beams": payload["n_beams"], "cos_true": payload["cos_true"],
                "cos_true_ci95": payload["cos_true_ci95"], "drop_frac": payload["drop_frac"],
                "disp_ratio": payload["disp_ratio"], "out": payload["align_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
