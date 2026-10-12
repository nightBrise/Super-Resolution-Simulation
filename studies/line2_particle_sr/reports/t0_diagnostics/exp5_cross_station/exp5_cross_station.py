#!/usr/bin/env python
"""T0 实验5：用不饱和度量（逐束 M1 = group_rel_rmse）重测跨站（站 0 → 站 1）。

背景：Phase 1 门禁 H7 消费的 N_eff 尺在站 0 / 站 1 两侧全部溢出触顶 1×10⁵，差异不可测。
本脚本换用不顶格的连续尺子——`test_eval*.json` 里已算好的逐束 M1（`per_beam[].err`）——
度量同一 ckpt 在站 0 与站 1 的落差。

口径：
- 数据源只读：`<results-dir>/<run>/test_eval.json`（站 0）与 `test_eval_station1.json`（站 1），
  两者由 `run_e1 eval --station {0,1}` 用**同一个 best_val.ckpt** 产出；
- 按 `beam_id` 逐束配对；两文件的 station / n_low / subs / beam_id 集合任一不一致即报错退出；
- Δ(b) = err_s1(b) − err_s0(b)；相对落差 rel(b) = Δ(b) / err_s0(b)；
- run 级统计量 = 对 150 束取中位数；配对 bootstrap 10k 次重采样束（同一束的站 0/1 同进同出，
  保持配对结构），中位数分布的 2.5/97.5 分位为 95% CI；
- Wilcoxon 符号秩检验双尾，对象为 Δ 与 rel 各一次；
- 随机性全部经 `SeedSequence(20260825).spawn(8)[run_index]` 派生，无裸随机；
- 分母语境三条：① E0 曲线 dump 的裸输入 M1（`raw_samples` 按「束主导 → 站 → 子样本」解码，
  解码次序经逐点复算校验）；② 两站模型 err 的绝对水平；③ 零信息常数解基线（复算口径与
  `t0_diagnostics/exp3_constant_control` 一致，仅作对照，不参与判定）。

用法：
    python exp5_cross_station.py [--results-dir <path>] [--out-dir <path>] [--no-constant-control]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = ROOT / "studies/line2_particle_sr/results"
DEFAULT_DATA = ROOT / "studies/line2_particle_sr/data/line2_ds_v1"
DEFAULT_OUT = ROOT / "studies/line2_particle_sr/reports/t0_diagnostics/exp5_cross_station"
EXP3_SNAPSHOT = ROOT / "studies/line2_particle_sr/reports/tools/phase1_numbers.json"

RUNS = [f"e1_joint_s{i}" for i in range(5)]
STATION0_FILENAME = "test_eval.json"
STATION1_FILENAME = "test_eval_station1.json"

MASTER_SEED = 20260825
N_BOOT = 10000
REL_THRESHOLD = 0.05


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _beam_errors(payload: dict) -> dict[int, dict]:
    return {int(b["beam_id"]): b for b in payload["per_beam"]}


def _check_pair(payload0: dict, payload1: dict, run_id: str) -> None:
    """配对前置校验：站号、档位、子样本清单、束集合、逐束中位数自洽，五项全过才可用。"""
    problems = []
    if int(payload0["station"]) != 0:
        problems.append(f"站 0 文件的 station={payload0['station']!r}")
    if int(payload1["station"]) != 1:
        problems.append(f"站 1 文件的 station={payload1['station']!r}")
    if str(payload0["run_id"]) != str(payload1["run_id"]) or str(payload0["run_id"]) != run_id:
        problems.append(f"run_id 不一致：{payload0['run_id']!r} / {payload1['run_id']!r}")
    if int(payload0["seed"]) != int(payload1["seed"]):
        problems.append(f"seed 不一致：{payload0['seed']} vs {payload1['seed']}")
    if int(payload0["n_low"]) != int(payload1["n_low"]):
        problems.append(f"n_low 不等：{payload0['n_low']} vs {payload1['n_low']}")
    if list(payload0["subs"]) != list(payload1["subs"]):
        problems.append(
            f"subs 不等：{len(payload0['subs'])} 项 vs {len(payload1['subs'])} 项"
        )
    beams0, beams1 = set(_beam_errors(payload0)), set(_beam_errors(payload1))
    if beams0 != beams1:
        problems.append(
            f"beam_id 集合不等：站 0 独有 {sorted(beams0 - beams1)[:5]}，"
            f"站 1 独有 {sorted(beams1 - beams0)[:5]}"
        )
    for tag, payload in (("站 0", payload0), ("站 1", payload1)):
        if abs(float(np.median([b["err"] for b in payload["per_beam"]]))
               - float(payload["err_median"])) > 1e-12:
            problems.append(f"{tag} 的 err_median 与逐束 err 中位数不符")
    if problems:
        raise ValueError(f"{run_id}：配对校验失败 —— " + "；".join(problems))


def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


def _station_n_eff(payload: dict) -> dict:
    flags = [str(b["n_eff_flag"]) for b in payload["per_beam"]]
    return {
        "n_eff": float(payload["n_eff"]),
        "flag": str(payload["n_eff_flag"]),
        "overflow_frac": sum(f == "overflow" for f in flags) / len(flags),
        "flagged_frac": sum(f not in ("ok", "None", "nan") for f in flags) / len(flags),
        "n_eff_beam_median": float(np.median([float(b["n_eff"]) for b in payload["per_beam"]])),
    }


def bare_by_station(results_dir: Path, test_beam_ids: set[int]) -> dict:
    """E0 曲线 dump 的裸输入 M1（分母语境）。

    `error_n_curve` 的累加次序为「束 → 站 → 档 → 子样本」，故 `raw_samples[tier]` 可按
    (束, 站, k) 展开；`beam_ids` 给出束序，据此切出逐站与 test 子集。
    """
    curve = _load(results_dir / "e0_curve.json")
    beam_ids = [int(b) for b in curve["beam_ids"]]
    k, n_st = int(curve["k"]), 2
    is_test = np.asarray([b in test_beam_ids for b in beam_ids])
    out = {"k": k, "n_beams": len(beam_ids), "n_stations": n_st, "tiers": {}}
    for tier in curve["tiers"]:
        raw = np.asarray(curve["raw_samples"][str(tier)], dtype=np.float64)
        raw = raw.reshape(len(beam_ids), n_st, k)
        out["tiers"][str(int(tier))] = {
            "bare_median_pooled": float(np.median(raw)),
            "bare_median_station0": float(np.median(raw[:, 0, :])),
            "bare_median_station1": float(np.median(raw[:, 1, :])),
            "bare_median_testonly_station0": float(np.median(raw[is_test, 0, :])),
            "bare_median_testonly_station1": float(np.median(raw[is_test, 1, :])),
        }
    out["reference_noise_floor"] = float(curve["reference_noise_floor"])
    out["mutual_floor"] = {str(t): float(v) for t, v in curve["mutual_floor"].items()}
    return out


def constant_solution(data_dir: Path, station: int) -> tuple[list[int], np.ndarray]:
    """零信息常数解基线：700 个 train 束 target_full 逐投影归一后的均值图 → 150 test 束 M1。

    口径照抄 `t0_diagnostics/exp3_constant_control/run_exp3_analysis.py`（该脚本只做了站 0）。
    """
    import h5py

    from src.line2.metrics.metrics import group_rel_rmse

    acc, n_beams = None, 0
    with h5py.File(data_dir / "train.h5", "r") as fh:
        for beam in sorted(fh.keys()):
            grp = fh[beam][f"station{station}"]["target_full"][:].astype(np.float64)
            sums = grp.sum(axis=(1, 2), keepdims=True)
            acc = grp / sums if acc is None else acc + grp / sums
            n_beams += 1
    mean_beam = acc / n_beams

    with h5py.File(data_dir / "test.h5", "r") as fh:
        keys = sorted(fh.keys())
        beam_ids = [int(b.split("_")[1]) for b in keys]
        errs = np.asarray(
            [group_rel_rmse(mean_beam, fh[b][f"station{station}"]["target_full"][:]) for b in keys],
            dtype=np.float64,
        )
    return beam_ids, errs


def verify_bare_decode(results_dir: Path, data_dir: Path, tier: int = 10000) -> dict:
    """抽点复算裸输入 M1，与 `raw_samples` 的解码值逐位比对。

    解码前提是 `error_n_curve` 的累加次序（束 → 站 → 档 → 子样本）；本函数用数据集
    重算 4 束 × 2 站 × 3 个子样本的值来证伪该前提。
    """
    import h5py

    from src.line2.metrics.metrics import group_rel_rmse

    curve = _load(results_dir / "e0_curve.json")
    beam_ids = [int(b) for b in curve["beam_ids"]]
    k = int(curve["k"])
    raw = np.asarray(curve["raw_samples"][str(tier)], dtype=np.float64)
    raw = raw.reshape(len(beam_ids), 2, k)

    manifest = _load(data_dir / "manifest.json")
    split_of = {int(b): s for s, ids in manifest["splits"].items() for b in ids}
    idx_list = [0, 1, len(beam_ids) // 2, len(beam_ids) - 1]
    subs = (0, 7, 19)
    max_diff, checks = 0.0, 0
    files: dict[str, h5py.File] = {}
    try:
        for idx in idx_list:
            beam = beam_ids[idx]
            split = split_of[beam]
            fh = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            for station in (0, 1):
                grp = fh[f"beam_{beam:04d}"][f"station{station}"]
                ref = grp["target_full"][...].astype(np.float64)
                for sub in subs:
                    img = grp[f"tier_{tier:06d}_sub_{sub:02d}"][...].astype(np.float64)
                    diff = abs(group_rel_rmse(img, ref) - raw[idx, station, sub])
                    max_diff = max(max_diff, diff)
                    checks += 1
    finally:
        for f in files.values():
            f.close()
    return {
        "tier": tier,
        "n_points": checks,
        "max_abs_diff": float(max_diff),
        "all_exact": bool(max_diff == 0.0),
        "sampled_beam_indices": idx_list,
        "sampled_subs": list(subs),
    }


def station_difficulty(data_dir: Path, n_beams: int = 30) -> dict:
    """两站的跨束差异刻度：test 束目标图两两 M1 中位数。

    与模型 err 同一 M1 口径，故直接同尺度——它给出"换成另一张束的图"的基线误差，
    是判断模型 err 在同一站内是否可信的最直接分母。
    """
    import h5py

    from src.line2.metrics.metrics import group_rel_rmse

    manifest = _load(data_dir / "manifest.json")
    keys = [f"beam_{b:04d}" for b in sorted(manifest["splits"]["test"])[:n_beams]]
    out: dict = {"n_beams_sampled": n_beams, "pair_scheme": "test 切分前 n 束的全部两两组合"}
    with h5py.File(data_dir / "test.h5", "r") as fh:
        for station in (0, 1):
            imgs = [
                fh[k][f"station{station}"]["target_full"][...].astype(np.float64) for k in keys
            ]
            vals = [
                group_rel_rmse(imgs[a], imgs[b])
                for a in range(len(imgs))
                for b in range(a + 1, len(imgs))
            ]
            out[f"station{station}"] = {
                "inter_beam_m1_median": float(np.median(vals)),
                "inter_beam_m1_p25": float(np.percentile(vals, 25)),
                "inter_beam_m1_p75": float(np.percentile(vals, 75)),
                "n_pairs": len(vals),
            }
    return out


def scan_scope(results_dir: Path) -> dict:
    """扫出全部带 test_eval.json 的 run 及其站 1 产物有无（范围可机读核对）。"""
    found: dict[str, dict] = {}
    for path0 in sorted(results_dir.glob(f"*/{STATION0_FILENAME}")):
        run_id = path0.parent.name
        found[run_id] = {
            "station0": True,
            "station1": (path0.parent / STATION1_FILENAME).is_file(),
        }
    for path1 in sorted(results_dir.glob(f"*/{STATION1_FILENAME}")):
        found.setdefault(path1.parent.name, {"station0": False, "station1": True})
    return {
        "runs_with_station0": sorted(k for k, v in found.items() if v["station0"]),
        "runs_with_station1": sorted(k for k, v in found.items() if v["station1"]),
        "station0_only": sorted(
            k for k, v in found.items() if v["station0"] and not v["station1"]
        ),
        "paired_available": sorted(
            k for k, v in found.items() if v["station0"] and v["station1"]
        ),
    }


def _verdict(per_run: list[dict], summary: dict) -> tuple[str, str, str]:
    """预注册判定规则的两分支落表（返回 verdict_code / verdict_text / branch）。"""
    n = len(per_run)
    if n == 0:
        return "indeterminate", "不可判定：无可用 run", "no_data"
    ci_excluded = summary["n_rel_ci_includes_zero"] == 0
    amp_over = abs(summary["median_rel_across_runs"]) > REL_THRESHOLD
    negative = all(r["median_rel"] < 0 for r in per_run)
    positive = all(r["median_rel"] > 0 for r in per_run)
    if ci_excluded and amp_over and (negative or positive):
        direction = "站 1 误差低于站 0（负向，非退化）" if negative else "站 1 误差高于站 0（退化）"
        text = (
            f"存在可量化的跨站差异：{n}/{n} 个种子的中位相对落差 CI 排除 0，"
            f"跨种子中位 = {summary['median_rel_across_runs'] * 100:+.2f}%（阈值 ±5%）；"
            f"方向 = {direction}"
        )
        return "quantified_cross_station_difference", text, "rule2_ci_excludes_zero_and_over_threshold"
    if summary["n_rel_ci_includes_zero"] >= n - 1 or not amp_over:
        text = (
            f"图像层跨站迁移成立（次级准绳口径）：{summary['n_rel_ci_includes_zero']}/{n} 个种子 CI 含 0，"
            f"跨种子中位 = {summary['median_rel_across_runs'] * 100:+.2f}%"
        )
        return "transfer_holds", text, "rule1_ci_contains_zero_or_within_threshold"
    return (
        "indeterminate",
        f"不可判定：{summary['n_rel_ci_includes_zero']}/{n} 个种子 CI 含 0 而跨种子中位超阈值",
        "mixed",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T0-exp5：跨站（站 0 → 站 1）不饱和度量复测")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", nargs="*", default=RUNS)
    parser.add_argument("--no-constant-control", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    children = np.random.SeedSequence(MASTER_SEED).spawn(8)
    per_run: list[dict] = []
    per_beam_rows: list[dict] = []
    missing: list[str] = []

    for run_idx, run_id in enumerate(args.runs):
        run_dir = args.results_dir / run_id
        path0, path1 = run_dir / STATION0_FILENAME, run_dir / STATION1_FILENAME
        if not path0.exists() or not path1.exists():
            absent = [p.name for p in (path0, path1) if not p.exists()]
            missing.append(f"{run_id}（缺 {'/'.join(absent)}）")
            continue
        payload0, payload1 = _load(path0), _load(path1)
        _check_pair(payload0, payload1, run_id)

        beam0, beam1 = _beam_errors(payload0), _beam_errors(payload1)
        beam_ids = sorted(beam0)
        err0 = np.asarray([beam0[b]["err"] for b in beam_ids], dtype=np.float64)
        err1 = np.asarray([beam1[b]["err"] for b in beam_ids], dtype=np.float64)
        delta = err1 - err0
        rel = delta / err0

        rng = np.random.default_rng(children[run_idx])
        rel_lo, rel_hi = _bootstrap_median_ci(rel, rng)
        delta_lo, delta_hi = _bootstrap_median_ci(delta, rng)

        st0, st1 = _station_n_eff(payload0), _station_n_eff(payload1)
        record = {
            "run_id": run_id,
            "seed": int(payload0["seed"]),
            "n_low": int(payload0["n_low"]),
            "n_subs": len(payload0["subs"]),
            "n_beams": len(beam_ids),
            "err_station0_median": float(np.median(err0)),
            "err_station1_median": float(np.median(err1)),
            "err_station0_run_field": float(payload0["err_median"]),
            "err_station1_run_field": float(payload1["err_median"]),
            "median_delta": float(np.median(delta)),
            "median_delta_ci": [delta_lo, delta_hi],
            "median_rel": float(np.median(rel)),
            "median_rel_ci": [rel_lo, rel_hi],
            "rel_ci_includes_zero": bool(rel_lo <= 0.0 <= rel_hi),
            "delta_ci_includes_zero": bool(delta_lo <= 0.0 <= delta_hi),
            "wilcoxon_delta_p": float(wilcoxon(delta).pvalue),
            "wilcoxon_rel_p": float(wilcoxon(rel).pvalue),
            "frac_beams_station1_worse": float(np.mean(delta > 0)),
            "station0": st0,
            "station1": st1,
            "code_version_station0": payload0["code_version"],
            "code_version_station1": payload1["code_version"],
        }
        per_run.append(record)
        for i, b in enumerate(beam_ids):
            per_beam_rows.append({
                "run_id": run_id,
                "seed": record["seed"],
                "beam_id": b,
                "err_station0": float(err0[i]),
                "err_station1": float(err1[i]),
                "delta": float(delta[i]),
                "rel": float(rel[i]),
                "n_eff_station0": float(beam0[b]["n_eff"]),
                "n_eff_station1": float(beam1[b]["n_eff"]),
                "flag_station0": str(beam0[b]["n_eff_flag"]),
                "flag_station1": str(beam1[b]["n_eff_flag"]),
            })

    med_rel = np.asarray([r["median_rel"] for r in per_run], dtype=np.float64)
    med_delta = np.asarray([r["median_delta"] for r in per_run], dtype=np.float64)
    summary = {
        "n_runs": len(per_run),
        "median_rel_across_runs": float(np.median(med_rel)),
        "median_rel_range": [float(med_rel.min()), float(med_rel.max())],
        "median_delta_across_runs": float(np.median(med_delta)),
        "median_delta_range": [float(med_delta.min()), float(med_delta.max())],
        "err_station0_median_across_runs": float(
            np.median([r["err_station0_median"] for r in per_run])
        ),
        "err_station1_median_across_runs": float(
            np.median([r["err_station1_median"] for r in per_run])
        ),
        "n_rel_ci_includes_zero": sum(r["rel_ci_includes_zero"] for r in per_run),
        "n_abs_rel_le_threshold": sum(abs(r["median_rel"]) <= REL_THRESHOLD for r in per_run),
        "n_seeds_station1_better": sum(r["median_rel"] < 0 for r in per_run),
        "rel_threshold": REL_THRESHOLD,
        "bootstrap": {"n_resamples": N_BOOT, "seed_sequence": MASTER_SEED},
    }
    verdict, verdict_text, branch = _verdict(per_run, summary)
    summary.update(verdict=verdict, verdict_text=verdict_text, verdict_branch=branch)

    bare = bare_by_station(args.results_dir, {int(r["beam_id"]) for r in per_beam_rows
                                             if r["run_id"] == per_run[0]["run_id"]})
    bare["decode_verification"] = verify_bare_decode(args.results_dir, args.data_dir)

    control: dict = {"enabled": not args.no_constant_control, "definition": (
        "700 个 train 束 target_full 逐投影归一后的均值图，对 150 个 test 束的 M1 中位数"
        "（口径同 exp3_constant_control，仅作分母语境，不参与判定）"
    )}
    if not args.no_constant_control:
        for station in (0, 1):
            ids, errs = constant_solution(args.data_dir, station)
            entry = {
                "n_train_beams": 700,
                "n_test_beams": len(ids),
                "constant_err_median": float(np.median(errs)),
                "constant_err_percentile25": float(np.percentile(errs, 25)),
                "constant_err_percentile75": float(np.percentile(errs, 75)),
            }
            if station == 0:
                if EXP3_SNAPSHOT.exists():
                    snap = json.loads(EXP3_SNAPSHOT.read_text(encoding="utf-8"))
                    ref = np.asarray(snap["zero_information"]["const_errs"], dtype=np.float64)
                    entry["exp3_snapshot_median"] = float(np.median(ref))
                    entry["max_abs_diff_vs_exp3_snapshot"] = float(
                        np.max(np.abs(errs - ref))
                    )
                entry["rerun_match_reported_value"] = abs(float(np.median(errs)) - 0.194398) < 1e-6
            control[f"station{station}"] = entry
        for station, tag in ((0, "station0"), (1, "station1")):
            model_err = np.asarray(
                [r[f"err_station{station}"] for r in per_beam_rows
                 if r["run_id"] == per_run[0]["run_id"]]
            )
            control[tag]["model_err_median_run0"] = float(np.median(model_err))

    difficulty = station_difficulty(args.data_dir)
    n_low = per_run[0]["n_low"]
    for tag in ("station0", "station1"):
        bare_test = bare["tiers"][str(n_low)][f"bare_median_testonly_{tag}"]
        model_m1 = summary[f"err_{tag}_median_across_runs"]
        floor = control.get(tag, {}).get("constant_err_median")
        entry = difficulty[tag]
        entry["bare_input_m1"] = bare_test
        entry["model_err_median"] = model_m1
        entry["constant_floor_m1"] = floor
        entry["model_over_inter_beam"] = model_m1 / entry["inter_beam_m1_median"]
        if floor is not None:
            entry["model_minus_constant"] = model_m1 - floor
            entry["model_over_constant"] = model_m1 / floor
            entry["bare_to_floor_gap"] = bare_test - floor
            entry["gap_closure_fraction"] = (bare_test - model_m1) / (bare_test - floor)

    payload_out = {
        "analysis": "T0-exp5：跨站（站 0 → 站 1）不饱和度量复测",
        "metric": "M1 = group_rel_rmse，取 test_eval per_beam[].err（逐束对 20 子样本取中位数）",
        "n_low": per_run[0]["n_low"] if per_run else None,
        "master_seed": MASTER_SEED,
        "n_bootstrap": N_BOOT,
        "scope": {
            "included": [r["run_id"] for r in per_run],
            "missing_station1": missing,
            "scan": scan_scope(args.results_dir),
        },
        "interpretation_caveats": [
            "E1/E2 训练消费两站：train.py 构造 Line2Pairs 未传 stations，取 data.py 默认 "
            "(0, 1)（src/line2/training/train.py:104、src/line2/training/data.py:47），故站 1 在"
            "训练分布内——本对比度量的是站间性能一致性，不是把模型外推到未见站点的迁移检验",
            "E0 曲线 dump 的裸输入 M1 为两站池化；逐站值由 raw_samples 的（束, 站, 子样本）"
            "展开次序解出，次序经逐点复算校验",
            "站 0 与站 1 的 test_eval 由不同 commit 产生（文件名内记录 code_version），但 "
            "src/line2/metrics/ 与模型推理路径在两 commit 间逐字节相同（差异只新增 M2 分桶字段）",
        ],
        "per_run": per_run,
        "summary": summary,
        "bare_reference": bare,
        "constant_control": control,
        "difficulty_normalized": difficulty,
    }
    (args.out_dir / "exp5_summary.json").write_text(
        json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (args.out_dir / "exp5_per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "run_id", "seed", "n_low", "n_beams", "err_station0_median", "err_station1_median",
            "median_delta", "median_delta_ci_lo", "median_delta_ci_hi", "median_rel",
            "median_rel_ci_lo", "median_rel_ci_hi", "rel_ci_includes_zero",
            "wilcoxon_delta_p", "wilcoxon_rel_p", "frac_beams_station1_worse",
            "n_eff_station0", "flag_station0", "overflow_frac_station0",
            "n_eff_station1", "flag_station1", "overflow_frac_station1",
        ])
        for r in per_run:
            writer.writerow([
                r["run_id"], r["seed"], r["n_low"], r["n_beams"],
                f"{r['err_station0_median']:.10f}", f"{r['err_station1_median']:.10f}",
                f"{r['median_delta']:.10f}", f"{r['median_delta_ci'][0]:.10f}",
                f"{r['median_delta_ci'][1]:.10f}", f"{r['median_rel']:.10f}",
                f"{r['median_rel_ci'][0]:.10f}", f"{r['median_rel_ci'][1]:.10f}",
                r["rel_ci_includes_zero"], f"{r['wilcoxon_delta_p']:.6e}",
                f"{r['wilcoxon_rel_p']:.6e}", f"{r['frac_beams_station1_worse']:.6f}",
                f"{r['station0']['n_eff']:.1f}", r["station0"]["flag"],
                f"{r['station0']['overflow_frac']:.6f}", f"{r['station1']['n_eff']:.1f}",
                r["station1"]["flag"], f"{r['station1']['overflow_frac']:.6f}",
            ])

    with (args.out_dir / "exp5_per_beam.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_beam_rows[0]))
        writer.writeheader()
        writer.writerows(per_beam_rows)

    print(json.dumps(
        {**summary, "scope": payload_out["scope"], "constant_control": control,
         "difficulty_normalized": difficulty},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
