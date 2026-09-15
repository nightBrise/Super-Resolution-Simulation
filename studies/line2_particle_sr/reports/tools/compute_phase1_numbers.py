"""复算 Phase 1 报告所需全部派生量，落盘为 JSON 快照（供图件与数字自检消费）。

运行：python studies/line2_particle_sr/reports/tools/compute_phase1_numbers.py
只读 studies/line2_particle_sr/{data,results} 与 docs/specs/line2；只写本目录 JSON。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.line2.metrics.metrics import group_rel_rmse  # noqa: E402

LINE2 = ROOT / "studies" / "line2_particle_sr"
RESULTS = LINE2 / "results"
DATA = LINE2 / "data" / "line2_ds_v1"
OUT = Path(__file__).resolve().parent / "phase1_numbers.json"

JOINT_RUNS = [f"e1_joint_s{i}" for i in range(5)]
INDEP_RUNS = [f"e1_indep_s{i}" for i in range(5)]
E2_RUNS = [f"e2_joint_n{t}_s{i}" for t in (3000, 100000) for i in range(5)]
ALL_RUNS = JOINT_RUNS + INDEP_RUNS + E2_RUNS
SEEDS = [100, 101, 102, 103, 104]


def load_n_eff_table() -> list[dict]:
    with (RESULTS / "n_eff_table.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["n_low"] = int(r["n_low"])
        r["station"] = int(r["station"])
        r["n_eff"] = float(r["n_eff"])
        r["err_median"] = float(r["err_median"])
        r["n_beams"] = int(r["n_beams"])
    return rows


def zero_information_solution() -> dict:
    """零信息常数解：700 个 train 束 station0/target_full 的逐投影归一化均值图。"""
    acc = None
    n_beams = 0
    with h5py.File(DATA / "train.h5", "r") as fh:
        for beam in sorted(fh.keys()):
            grp = fh[beam]["station0"]["target_full"][:].astype(np.float64)
            sums = grp.sum(axis=(1, 2), keepdims=True)
            acc = grp / sums if acc is None else acc + grp / sums
            n_beams += 1
    mean_beam = acc / n_beams

    with h5py.File(DATA / "test.h5", "r") as fh:
        beam_ids = sorted(fh.keys())
        errs = []
        refs = {}
        for beam in beam_ids:
            ref = fh[beam]["station0"]["target_full"][:]
            refs[int(beam.split("_")[1])] = ref
            errs.append(group_rel_rmse(mean_beam, ref))
    errs = np.asarray(errs, dtype=np.float64)

    # 前 200 个 train 束的敏感性对照（口径同 stage report §3）
    acc200 = None
    with h5py.File(DATA / "train.h5", "r") as fh:
        for beam in sorted(fh.keys())[:200]:
            grp = fh[beam]["station0"]["target_full"][:].astype(np.float64)
            sums = grp.sum(axis=(1, 2), keepdims=True)
            acc200 = grp / sums if acc200 is None else acc200 + grp / sums
    mean200 = acc200 / 200
    errs200 = np.asarray(
        [group_rel_rmse(mean200, refs[b]) for b in sorted(refs)], dtype=np.float64
    )

    per_beam = {}
    for run in ALL_RUNS:
        payload = json.loads((RESULTS / run / "test_eval.json").read_text(encoding="utf-8"))
        pb = sorted(payload["per_beam"], key=lambda d: d["beam_id"])
        run_errs = np.asarray([d["err"] for d in pb], dtype=np.float64)
        order = [int(b.split("_")[1]) for b in beam_ids]
        const_errs = np.asarray([errs[i] for i in range(len(errs))], dtype=np.float64)
        wins = int(np.sum(run_errs < const_errs))
        per_beam[run] = {
            "n_wins_vs_const": wins,
            "n_beams": len(run_errs),
            "median_ratio_model_over_const": float(np.median(run_errs / const_errs)),
            "err_median_model": float(np.median(run_errs)),
            "err_median_const": float(np.median(const_errs)),
            "model_errs": [float(v) for v in run_errs],
            "const_errs": [float(v) for v in const_errs],
            "beam_ids": order,
        }
    return {
        "n_train_beams": n_beams,
        "const_median": float(np.median(errs)),
        "const_std": float(errs.std(ddof=0)),
        "const_min": float(errs.min()),
        "const_max": float(errs.max()),
        "const_errs": [float(v) for v in errs],
        "const_first200_median": float(np.median(errs200)),
        "per_run": per_beam,
    }


def dark_buckets() -> dict:
    """B0–B3 的 M2 裸输入 vs 模型（5 个 joint run × 150 束 × 站 0 @1e4）。"""
    names = ["B0", "B1", "B2", "B3"]
    bare: dict[str, list[list[float]]] = {n: [] for n in names}
    model: dict[str, list[list[float]]] = {n: [] for n in names}
    per_beam_rows: list[dict] = []
    for run in JOINT_RUNS:
        payload = json.loads((RESULTS / run / "test_eval.json").read_text(encoding="utf-8"))
        pb = sorted(payload["per_beam"], key=lambda d: d["beam_id"])
        for name in names:
            bare[name].append([d["m2_buckets"][name]["bare_median"] for d in pb])
            model[name].append([d["m2_buckets"][name]["model_median"] for d in pb])
        for d in pb:
            per_beam_rows.append(
                {
                    "beam_id": d["beam_id"],
                    "n_cells": {n: d["m2_buckets"][n]["n_cells"] for n in names},
                }
            )
    out = {}
    for name in names:
        b = np.median(np.asarray(bare[name], dtype=np.float64), axis=0)
        m = np.median(np.asarray(model[name], dtype=np.float64), axis=0)
        out[name] = {
            "bare_median": float(np.median(b)),
            "model_median": float(np.median(m)),
            "relative_improvement": (
                None if np.median(b) == 0 else float((np.median(b) - np.median(m)) / np.median(b))
            ),
        }
    n_cells = np.median(
        np.asarray([[r["n_cells"][n] for n in names] for r in per_beam_rows]), axis=0
    ).tolist()
    out["_n_cells_median"] = {n: float(v) for n, v in zip(names, n_cells)}
    combined = np.median(
        np.asarray(
            [
                [
                    (np.median(np.asarray(bare["B0"], dtype=np.float64)[:, i])
                     + np.median(np.asarray(bare["B1"], dtype=np.float64)[:, i])) / 2,
                    (np.median(np.asarray(model["B0"], dtype=np.float64)[:, i])
                     + np.median(np.asarray(model["B1"], dtype=np.float64)[:, i])) / 2,
                ]
                for i in range(150)
            ]
        ),
        axis=0,
    )
    out["B0B1_combined"] = {
        "bare_median": float(combined[0]),
        "model_median": float(combined[1]),
        "relative_improvement": float((combined[0] - combined[1]) / combined[0]),
    }
    return out


def training_dynamics() -> dict:
    out = {}
    for run in ALL_RUNS:
        rows = [
            json.loads(line)
            for line in (RESULTS / run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads((RESULTS / run / "summary.json").read_text(encoding="utf-8"))
        out[run] = {
            "steps": [r["step"] for r in rows],
            "val_n_eff": [r["val_n_eff"] for r in rows],
            "train_nll": [r["train_nll"] for r in rows],
            "val_nll": [r["val_nll"] for r in rows],
            "best_step": summary["best_step"],
            "steps_total": summary["steps"],
            "params_total": summary["params_total"],
            "elapsed_s": summary["elapsed_s"],
        }
    return out


def derived() -> dict:
    """报告渲染直接消费的派生量（时间戳、体积、几何统计、曲线聚合）。"""
    meta = {}
    for run in ALL_RUNS:
        meta[run] = json.loads((RESULTS / run / "run_meta.json").read_text(encoding="utf-8"))
    e1 = JOINT_RUNS + INDEP_RUNS
    e1_start = min(meta[r]["started_at"] for r in e1)
    e1_end = max(meta[r]["completed_at"] for r in e1)
    e2_start = min(meta[r]["started_at"] for r in E2_RUNS)
    e2_end = max(meta[r]["completed_at"] for r in E2_RUNS)

    e0_done = json.loads((RESULTS / "e0_curve.meta.json").read_text(encoding="utf-8"))["completed_at"]
    from datetime import datetime

    def parse(t: str) -> datetime:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))

    gap_min = (parse(e1_start) - parse(e0_done)).total_seconds() / 60.0

    rows = load_n_eff_table()
    groups = {
        "3e3": [r for r in rows if r["n_low"] == 3000 and r["station"] == 0],
        "1e4_joint": [r for r in rows if r["n_low"] == 10000 and r["station"] == 0 and r["variant"] == "joint"],
        "1e4_indep": [r for r in rows if r["n_low"] == 10000 and r["station"] == 0 and r["variant"] == "independent"],
        "1e5": [r for r in rows if r["n_low"] == 100000 and r["station"] == 0],
    }
    curve_agg = {k: float(np.median([r["err_median"] for r in v])) for k, v in groups.items()}
    non_sat = [r for r in rows if r["n_eff_flag"] == ""]
    assert len(non_sat) == 1, non_sat

    sizes = {s: (DATA / f"{s}.h5").stat().st_size for s in ("train", "val", "test")}
    gb = {k: v / 1e9 for k, v in sizes.items()}

    with h5py.File(DATA / "val.h5", "r") as fh:
        grp = fh["beam_0006"]
        stats = {}
        for station in (0, 1):
            edges = np.asarray(grp[f"station{station}"]["target_full"].attrs["bin_edges"], dtype=np.float64)
            img = grp[f"station{station}"]["target_full"][:].astype(np.float64)
            total = img.sum()
            for gi, (ai, aj) in enumerate(((0, 1), (1, 2))):
                sub = img[gi]
                ui = 0.5 * (edges[ai][:-1] + edges[ai][1:])
                uj = 0.5 * (edges[aj][:-1] + edges[aj][1:])
                pi = sub.sum(axis=1) / total
                pj = sub.sum(axis=0) / total
                mi = float((pi * ui).sum()); mj = float((pj * uj).sum())
                stats[(station, ai)] = (
                    float(np.sqrt((pi * (ui - mi) ** 2).sum())),
                    float(np.sqrt((pj * (uj - mj) ** 2).sum())),
                )
    return {
        "e1_first_start": e1_start,
        "e1_last_end": e1_end,
        "e2_start": e2_start,
        "e2_train_end": e2_end,
        "chain_end": "2026-09-14T23:53:42Z",
        "e0_pre_e1_minutes": gap_min,
        "curve_agg": curve_agg,
        "single_non_saturated": non_sat[0],
        "dataset_gb": gb,
        "fig02_stats": {
            "sigma_xp_s0": stats[(0, 0)][1],
            "sigma_xp_s1": stats[(1, 0)][1],
            "sigma_x_s0": stats[(0, 0)][0],
            "sigma_x_s1": stats[(1, 0)][0],
        },
    }


def main() -> None:
    snapshot = {
        "sources": {
            "n_eff_table": "studies/line2_particle_sr/results/n_eff_table.csv",
            "e0_curve": "studies/line2_particle_sr/results/e0_curve.json",
            "e0_meta": "studies/line2_particle_sr/results/e0_curve.meta.json",
            "tolerance": "studies/line2_particle_sr/results/tolerance.json",
            "e1_gates": "studies/line2_particle_sr/results/e1_gates.json",
            "e2_routing": "studies/line2_particle_sr/results/e2_routing.json",
        },
        "n_eff_rows": load_n_eff_table(),
        "e0": json.loads((RESULTS / "e0_curve.json").read_text(encoding="utf-8")),
        "e0_meta": json.loads((RESULTS / "e0_curve.meta.json").read_text(encoding="utf-8")),
        "tolerance": json.loads((RESULTS / "tolerance.json").read_text(encoding="utf-8")),
        "gates": json.loads((RESULTS / "e1_gates.json").read_text(encoding="utf-8")),
        "routing": json.loads((RESULTS / "e2_routing.json").read_text(encoding="utf-8")),
        "routing_summary": {
            k: v
            for k, v in json.loads((RESULTS / "e2_routing.json").read_text(encoding="utf-8")).items()
            if k in ("winner_variant", "basis", "joint_wins", "n_pairs", "val_medians", "branch_rule")
        },
    }
    print("[1/4] 训练动力学", flush=True)
    snapshot["training"] = training_dynamics()
    print("[2/4] 暗桶 M2", flush=True)
    snapshot["dark_buckets"] = dark_buckets()
    print("[3/4] 零信息常数解（读 train.h5 700 束）", flush=True)
    snapshot["zero_information"] = zero_information_solution()
    print("[4/5] 派生量（时间戳/体积/几何/聚合）", flush=True)
    snapshot["derived"] = derived()
    print("[5/5] 落盘", flush=True)
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
