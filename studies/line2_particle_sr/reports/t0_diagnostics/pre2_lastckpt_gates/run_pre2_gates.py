"""前置②：last.ckpt 口径的门禁复算、三点主曲线与常数解对照（纯 CPU，只读）。

输入（全部只读）：
- ``results/e1_{joint,indep}_s{0..4}/test_eval.json``——**best_val 口径**的官方站 0 @1e4 test 评估；
- ``results/e2_joint_n{3000,100000}_s{0..4}/test_eval.json``——best_val 口径的另两档曲线点；
- ``reports/t0_diagnostics/exp1_early_stop/<run>_last_eval.json``——last.ckpt 同口径评估（exp1 产物）；
- ``results/e0_curve.json``、``results/tolerance.json``——裸误差曲线与容差线（官方尺子）；
- ``reports/t0_diagnostics/exp3_constant_control/exp3_results.json``（与 ``reports/tools/phase1_numbers.json``
  的 ``zero_information.const_errs`` 逐位一致）——150 束常数解 err。

判定一律走官方代码路径 ``src/line2/experiments/gates.py``（``evaluate_h1`` / ``evaluate_h3`` /
``h3_disclosure`` / ``_pair_tally`` / ``_h8_from_disk``）+ ``src/line2/metrics/tolerance.py`` 的
``tolerance_for``；本脚本只自写 loader（把 eval JSON 变成 gates 期望的 ``per_beam`` 结构），
不改判定函数、不调用 ``evaluate_gates_from_disk``（那会向 ``results/`` 回写 ``e1_gates.json``
——``results/`` 本任务只读）。

H3 的**次级准绳口径**：官方 H3 判据吃 N_eff，@1e4 两臂全部上溢 ⇒ 5 对全为域顶平局，无区分力。
本脚本把同一判定函数作用在**连续 err** 上（取负号以保持 gates 的"越大越好"取向），
标注为"次级准绳口径"，**不是**预注册判据。

随机性：``np.random.SeedSequence(20260825)`` 两个命名子流（H3 err 配对 bootstrap / 常数解配对
bootstrap），每种子再 ``spawn(5)[i]``、pooled 用 ``spawn(5)[4]``，无裸随机调用。

产物（全部写本目录）：``pre2_gates_compare.json``、``pre2_compare.csv``、``pre2_compare_table.md``、
``pre2_curve_points.csv``、``pre2_h3_err_pairs.csv``、``pre2_const_pairs.csv``。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/home/zhangny/Super-Resolution-Simulation")
sys.path.insert(0, str(REPO))

from src.line2.experiments import gates  # noqa: E402
from src.line2.metrics.tolerance import tolerance_for  # noqa: E402

LINE2 = REPO / "studies/line2_particle_sr"
RESULTS = LINE2 / "results"
EXP1 = LINE2 / "reports/t0_diagnostics/exp1_early_stop"
EXP3 = LINE2 / "reports/t0_diagnostics/exp3_constant_control"
PHASE1_NUMBERS = LINE2 / "reports/tools/phase1_numbers.json"
HERE = Path(__file__).resolve().parent

SEEDS = (100, 101, 102, 103, 104)
JOINT_RUNS = tuple(f"e1_joint_s{i}" for i in range(5))
INDEP_RUNS = tuple(f"e1_indep_s{i}" for i in range(5))
CURVE_RUNS = {
    3000: tuple(f"e2_joint_n3000_s{i}" for i in range(5)),
    10000: JOINT_RUNS,
    100000: tuple(f"e2_joint_n100000_s{i}" for i in range(5)),
}
BARE_1E5 = 0.28563574695791116              # 1e5 档裸子样本误差中位数（e0_curve point）
BOOT_N = 10_000

_ROOT = np.random.SeedSequence(20260825)
_STREAM = dict(
    zip(("h3_err", "const"), _ROOT.spawn(2), strict=True)
)


# --------------------------------------------------------------------------- loader
def load_eval(path: Path) -> dict:
    """读一份 eval JSON 并执行与 ``gates._load_test_evals`` 相同的 split 断言。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test":
        raise ValueError(f"{path}：split={payload.get('split')!r}——出口门禁只消费 test 段")
    payload["eval_path"] = str(path)
    return payload


def load_best_val(runs) -> dict[int, dict]:
    return {
        int(json.loads((RESULTS / r / "test_eval.json").read_text(encoding="utf-8"))["seed"]):
        load_eval(RESULTS / r / "test_eval.json")
        for r in runs
    }


def load_last(runs) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in runs:
        payload = load_eval(EXP1 / f"{r}_last_eval.json")
        if payload["run_id"] != r:
            raise ValueError(f"{EXP1}/{r}_last_eval.json：run_id={payload['run_id']!r} ≠ {r}")
        out[int(payload["seed"])] = payload
    return out


def beam_records(payload: dict) -> dict[int, dict]:
    return {int(b["beam_id"]): b for b in payload["per_beam"]}


def beam_values(runs_by_seed: dict[int, dict], field: str) -> tuple[list[int], dict[int, np.ndarray]]:
    """束 id 交集（升序）× 每种子该字段向量。"""
    records = {s: beam_records(r) for s, r in sorted(runs_by_seed.items())}
    ids = sorted(set.intersection(*[set(rec) for rec in records.values()]))
    per_seed = {
        s: np.asarray([records[s][b][field] for b in ids], dtype=np.float64)
        for s in sorted(records)
    }
    return ids, per_seed


# --------------------------------------------------------------------------- stats
def paired_bootstrap_median_ci(delta: np.ndarray, stream: np.random.SeedSequence) -> tuple[float, float, float]:
    rng = np.random.default_rng(stream)
    n = delta.size
    idx = rng.integers(0, n, size=(BOOT_N, n))
    medians = np.median(delta[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(np.median(delta)), float(lo), float(hi)


def wilcoxon_p(delta: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
    except ValueError:  # 全零差
        return float("nan")


# --------------------------------------------------------------------------- 门禁复算
def h1_block(runs_by_seed: dict[int, dict]) -> dict:
    values = [float(runs_by_seed[s]["n_eff"]) for s in sorted(runs_by_seed)]
    payload = {
        "verdict": gates.evaluate_h1(values),
        "n_seeds": len(values),
        "per_seed_n_eff": {str(s): float(runs_by_seed[s]["n_eff"]) for s in sorted(runs_by_seed)},
        "per_seed_flag": {str(s): runs_by_seed[s]["n_eff_flag"] for s in sorted(runs_by_seed)},
        "median_n_eff": float(np.median(values)),
        "n_above_n_low": int(sum(1 for v in values if v > gates.GATE_N_LOW)),
        "thresholds": {"n_low": float(gates.GATE_N_LOW), "median_min": 2.0 * gates.GATE_N_LOW},
    }
    return payload


def h3_block(joint: dict[int, dict], indep: dict[int, dict], criterion: str) -> dict:
    """criterion="n_eff"（官方）或 "neg_err"（次级准绳，取负号保持"越大越好"取向）。"""
    if criterion == "n_eff":
        pick = lambda p: float(p["n_eff"])  # noqa: E731
    elif criterion == "neg_err":
        pick = lambda p: -float(p["err_median"])  # noqa: E731
    else:
        raise ValueError(criterion)
    pairs = gates.pair_by_seed({s: pick(joint[s]) for s in joint}, {s: pick(indep[s]) for s in indep})
    jv = [pick(joint[s]) for s, _ in pairs]
    iv = [pick(indep[s]) for _, s in pairs]
    flags = [joint[s]["n_eff_flag"] for s, _ in pairs] + [indep[s]["n_eff_flag"] for _, s in pairs]
    return {
        "criterion": criterion,
        "verdict": gates.evaluate_h3(jv, iv),
        **_pair_tally_public(jv, iv),
        # 域顶平局标志只对 N_eff 口径有语义；err 口径不携带 N_eff 标记
        **gates.h3_disclosure(jv, iv, flags=flags if criterion == "n_eff" else None),
        "pairing": {"basis": "seed_value", "pairs": [[a, b] for a, b in pairs]},
        "n_eff_flags": {
            "joint": {str(s): joint[s]["n_eff_flag"] for s, _ in pairs},
            "independent": {str(s): indep[s]["n_eff_flag"] for _, s in pairs},
        },
        "inputs": {
            "joint": {str(s): pick(joint[s]) for s, _ in pairs},
            "independent": {str(s): pick(indep[s]) for _, s in pairs},
        },
    }


def _pair_tally_public(joint, indep) -> dict:
    """官方 ``gates._pair_tally``（同一计票实现，单一实现点）。"""
    return gates._pair_tally(joint, indep)


def h8_block(runs_by_seed: dict[int, dict], curve_nodes, tolerance_path: Path) -> dict:
    """官方 H8 聚合路径：``gates._h8_from_disk``（逐观测量先对 5 种子取中位数、再判达标）。"""
    return gates._h8_from_disk(runs_by_seed, curve_nodes, tolerance_path)


def h8_per_seed_disclosure(runs_by_seed: dict[int, dict], curve_nodes, tolerance_path: Path) -> dict:
    """披露（**非**官方判定）：容差线仍用束级 N_eff 中位数的档位映射，但改为逐种子单独判达标。

    用于检查官方"先对种子取中位数"这一步是否掩盖了种子间差异；结论不进 ``evaluate_h8`` 口径。
    """
    dev_table = gates._load_tolerance(Path(tolerance_path))
    ids, per_seed = beam_values(runs_by_seed, "obs")
    records = [beam_records(r) for _, r in sorted(runs_by_seed.items())]
    tol = np.stack([
        tolerance_for(dev_table, curve_nodes,
                      float(np.median([rec[b]["n_eff"] for rec in records])))
        for b in ids
    ])
    per_seed_rate = {
        str(s): gates.evaluate_h8(per_seed[s], tol)["pass_rate"] for s in sorted(per_seed)
    }
    return {
        "per_seed_pass_rate": per_seed_rate,
        "median_of_per_seed": float(np.median(list(per_seed_rate.values()))),
        "min_of_per_seed": float(min(per_seed_rate.values())),
        "max_of_per_seed": float(max(per_seed_rate.values())),
    }


def _bare_curve_extrapolation(err: float, bare_at: dict[int, float]) -> dict:
    """披露（**非**测量）：用裸曲线 3e4→1e5 段的 log-log 局部斜率外推 err 对应的 N。

    报告域顶 = 1e5（`20` [S3]），外推值**超出尺子定义域**，只说明"当前尺子量不到"，
    不得当作 N_eff 的测量结果引用。
    """
    slope = math.log10(bare_at[100000] / bare_at[30000]) / math.log10(30000.0 / 100000.0)
    n_ext = 100000.0 * (bare_at[100000] / err) ** (1.0 / slope)
    return {"local_loglog_slope_30k_100k": -slope, "n_extrapolated": float(n_ext),
            "n_extrapolated_over_1e4": float(n_ext / 1e4)}


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    curve = json.loads((RESULTS / "e0_curve.json").read_text(encoding="utf-8"))
    curve_nodes = tuple(int(p["n_low"]) for p in curve["points"])
    bare_at = {int(p["n_low"]): float(p["median"]) for p in curve["points"]}
    tolerance_path = RESULTS / "tolerance.json"
    official = json.loads((RESULTS / "e1_gates.json").read_text(encoding="utf-8"))
    out: dict = {
        "task": "前置②：last.ckpt 口径门禁复算 + 三点主曲线 + 常数解对照",
        "note": "本产物为 Phase 1.5 预注册提案的附件证据；官方判定输出（results/e1_gates.json）未改动。",
        "bare_curve_medians": {str(k): v for k, v in bare_at.items()},
        "curve_nodes": [int(n) for n in curve_nodes],
    }

    # ---- 输入装载（best_val = 官方 test_eval.json；last = exp1 产物）
    joint_best = load_best_val(JOINT_RUNS)
    indep_best = load_best_val(INDEP_RUNS)
    joint_last = load_last(JOINT_RUNS)
    indep_last = load_last(INDEP_RUNS)

    # ---- 口径一致性核对：exp1 的 bestval 复算文件 vs 官方 test_eval.json（束级 err/obs/n_eff 逐位）
    ctrl = []
    for run in (*JOINT_RUNS, *INDEP_RUNS):
        official_eval = json.loads((RESULTS / run / "test_eval.json").read_text(encoding="utf-8"))
        rerun = json.loads((EXP1 / f"{run}_bestval_eval.json").read_text(encoding="utf-8"))
        a = beam_records(official_eval)
        b = beam_records(rerun)
        err_diff = max(abs(a[k]["err"] - b[k]["err"]) for k in a)
        obs_diff = max(
            float(np.max(np.abs(np.asarray(a[k]["obs"]) - np.asarray(b[k]["obs"])))) for k in a
        )
        ctrl.append({"run_id": run, "max_abs_err_diff": err_diff, "max_abs_obs_diff": obs_diff,
                     "n_eff_equal": float(official_eval["n_eff"]) == float(rerun["n_eff"])})
    out["control_rerun_vs_official"] = {
        "rows": ctrl,
        "max_abs_err_diff_all": max(r["max_abs_err_diff"] for r in ctrl),
        "max_abs_obs_diff_all": max(r["max_abs_obs_diff"] for r in ctrl),
        "all_n_eff_equal": all(r["n_eff_equal"] for r in ctrl),
    }

    # ---- H1 / H3（官方 N_eff 口径）
    gates_out = {
        "best_val": {
            "H1": h1_block(joint_best),
            "H3_n_eff": h3_block(joint_best, indep_best, "n_eff"),
        },
        "last": {
            "H1": h1_block(joint_last),
            "H3_n_eff": h3_block(joint_last, indep_last, "n_eff"),
        },
    }

    # ---- H8（官方口径；joint 为门禁臂，independent 为披露对照）
    gates_out["best_val"]["H8_joint"] = h8_block(joint_best, curve_nodes, tolerance_path)
    gates_out["last"]["H8_joint"] = h8_block(joint_last, curve_nodes, tolerance_path)
    gates_out["best_val"]["H8_independent"] = h8_block(indep_best, curve_nodes, tolerance_path)
    gates_out["last"]["H8_independent"] = h8_block(indep_last, curve_nodes, tolerance_path)

    # H8 到 80% 线的对（束, 观测量）对数缺口（官方口径下的可操作数字）
    out["H8_gap_to_pass_line"] = {}
    for ckpt in ("best_val", "last"):
        for arm, key in (("joint", "H8_joint"), ("independent", "H8_independent")):
            h = gates_out[ckpt][key]
            n_total = h["n_beams"] * gates.N_OBSERVABLES
            n_pass = int(round(h["pass_rate"] * n_total))
            n_need = int(math.ceil(gates.H8_PASS_RATE * n_total))
            out["H8_gap_to_pass_line"][f"{ckpt}_{arm}"] = {
                "n_pairs_total": n_total, "n_pairs_pass": n_pass,
                "n_pairs_needed_for_pass_line": n_need, "n_pairs_short": n_need - n_pass,
            }

    # H8 逐种子披露（非官方判定）+ 曲线外推披露（超出报告域，仅说明尺子量不到）
    out["H8_per_seed_disclosure"] = {
        "best_val_joint": h8_per_seed_disclosure(joint_best, curve_nodes, tolerance_path),
        "last_joint": h8_per_seed_disclosure(joint_last, curve_nodes, tolerance_path),
        "best_val_independent": h8_per_seed_disclosure(indep_best, curve_nodes, tolerance_path),
        "last_independent": h8_per_seed_disclosure(indep_last, curve_nodes, tolerance_path),
    }

    # ---- 官方 e1_gates.json 对账（best_val 列必须逐位复现）
    val_check = {
        "H1_verdict": [official["H1"]["verdict"], gates_out["best_val"]["H1"]["verdict"]],
        "H1_median_n_eff": [official["H1"]["median_n_eff"], gates_out["best_val"]["H1"]["median_n_eff"]],
        "H3_verdict": [official["H3"]["verdict"], gates_out["best_val"]["H3_n_eff"]["verdict"]],
        "H3_n_ties": [official["H3"]["n_ties"], gates_out["best_val"]["H3_n_eff"]["n_ties"]],
        "H3_tied_at_domain_edge": [official["H3"]["tied_at_domain_edge"],
                                   gates_out["best_val"]["H3_n_eff"]["tied_at_domain_edge"]],
        "H8_pass_rate": [official["H8"]["pass_rate"], gates_out["best_val"]["H8_joint"]["pass_rate"]],
        "H8_verdict": [official["H8"]["verdict"], gates_out["best_val"]["H8_joint"]["verdict"]],
        "H8_tier_counts": [official["H8"]["tolerance_tier_counts"],
                           gates_out["best_val"]["H8_joint"]["tolerance_tier_counts"]],
    }
    val_check["all_match"] = all(a == b for k, (a, b) in val_check.items() if k != "all_match")
    out["official_gates_reproduction"] = val_check

    # ---- H3 次级准绳口径（连续 err）
    h3_err = {}
    for label, joint, indep in (("best_val", joint_best, indep_best),
                                ("last", joint_last, indep_last)):
        _, err_j = beam_values(joint, "err")
        _, err_i = beam_values(indep, "err")
        per_seed = []
        for k, seed in enumerate(SEEDS):
            d = err_j[seed] - err_i[seed]
            med, lo, hi = paired_bootstrap_median_ci(d, _STREAM["h3_err"].spawn(5)[k])
            per_seed.append({
                "seed": seed,
                "err_median_joint": float(joint[seed]["err_median"]),
                "err_median_independent": float(indep[seed]["err_median"]),
                "run_level_winner": "joint" if joint[seed]["err_median"] < indep[seed]["err_median"] else "independent",
                "delta_median": med, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0),
                "n_beams_joint_better": int(np.sum(d < 0)), "n_beams_tie": int(np.sum(d == 0)),
                "p_wilcoxon": wilcoxon_p(d),
            })
        mean_j = np.mean(np.stack([err_j[s] for s in SEEDS]), axis=0)
        mean_i = np.mean(np.stack([err_i[s] for s in SEEDS]), axis=0)
        pooled = mean_j - mean_i
        med, lo, hi = paired_bootstrap_median_ci(pooled, _STREAM["h3_err"].spawn(5)[4])
        h3_err[label] = {
            "per_seed": per_seed,
            "n_seeds_joint_better": int(sum(r["run_level_winner"] == "joint" for r in per_seed)),
            "pooled_5seed_beam_mean": {
                "delta_median": med, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0),
                "n_beams_joint_better": int(np.sum(pooled < 0)), "n_beams": int(pooled.size),
                "p_wilcoxon": wilcoxon_p(pooled),
            },
            "gates_expression_on_neg_err": h3_block(joint, indep, "neg_err"),
        }
    out["H3_err_secondary"] = h3_err

    # ---- 三点主曲线
    curve_rows = []
    for n_low, runs in CURVE_RUNS.items():
        variants = (("joint", runs),) if n_low != 10000 else (("joint", runs), ("independent", INDEP_RUNS))
        for variant, run_ids in variants:
            for ckpt in ("best_val", "last"):
                payloads = (
                    load_best_val(run_ids) if ckpt == "best_val" else load_last(run_ids)
                )
                errs = [float(payloads[s]["err_median"]) for s in sorted(payloads)]
                n_effs = [float(payloads[s]["n_eff"]) for s in sorted(payloads)]
                curve_rows.append({
                    "ckpt": ckpt, "variant": variant, "n_low": n_low, "station": 0, "split": "test",
                    "n_runs": len(errs),
                    "err_median_of_seeds": float(np.median(errs)),
                    "err_per_seed": {str(s): float(payloads[s]["err_median"]) for s in sorted(payloads)},
                    "n_eff_median": float(np.median(n_effs)),
                    "n_eff_per_seed": {str(s): float(payloads[s]["n_eff"]) for s in sorted(payloads)},
                    "n_overflow": int(sum(1 for p in payloads.values() if p["n_eff_flag"] == "overflow")),
                    "n_underflow": int(sum(1 for p in payloads.values() if p["n_eff_flag"] == "underflow")),
                    "ratio_median": float(np.median(n_effs)) / float(n_low),
                    "bare_at_tier": bare_at[n_low],
                    "err_over_bare_at_tier": float(np.median(errs)) / bare_at[n_low],
                    "err_over_bare_1e5": float(np.median(errs)) / BARE_1E5,
                })
    out["curve_points"] = curve_rows
    out["curve_extrapolation_disclosure"] = {
        f"{r['ckpt']}_{r['variant']}_{r['n_low']}": _bare_curve_extrapolation(
            r["err_median_of_seeds"], bare_at
        )
        for r in curve_rows
        if r["variant"] == "joint"
    }

    # ---- 模型 vs 常数解（@1e4 joint 5 种子）
    exp3 = json.loads((EXP3 / "exp3_results.json").read_text(encoding="utf-8"))
    phase1 = json.loads(PHASE1_NUMBERS.read_text(encoding="utf-8"))
    const_by_beam = {int(b): float(e) for b, e in zip(exp3["beam_ids"], exp3["constant_solution_errs"])}
    const_snapshot_equal = phase1["zero_information"]["const_errs"] == exp3["constant_solution_errs"]
    const_pairs = {}
    for label, joint in (("best_val", joint_best), ("last", joint_last)):
        ids, errs = beam_values(joint, "err")
        const_v = np.asarray([const_by_beam[b] for b in ids], dtype=np.float64)
        per_seed = []
        for k, seed in enumerate(SEEDS):
            adv = const_v - errs[seed]
            med, lo, hi = paired_bootstrap_median_ci(adv, _STREAM["const"].spawn(5)[k])
            per_seed.append({
                "seed": seed,
                "err_median_model": float(np.median(errs[seed])),
                "err_median_const": float(np.median(const_v)),
                "n_wins_vs_const": int(np.sum(adv > 0)),
                "n_ties": int(np.sum(adv == 0)),
                "win_rate": float(np.mean(adv > 0)),
                "median_advantage": med, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0),
                "p_wilcoxon": wilcoxon_p(adv),
            })
        model_5 = np.median(np.stack([errs[s] for s in SEEDS]), axis=0)
        adv5 = const_v - model_5
        med, lo, hi = paired_bootstrap_median_ci(adv5, _STREAM["const"].spawn(5)[4])
        const_pairs[label] = {
            "n_beams": int(len(ids)),
            "err_median_const": float(np.median(const_v)),
            "per_seed": per_seed,
            "n_seeds_median_err_below_const": int(sum(np.median(errs[s]) < np.median(const_v) for s in SEEDS)),
            "pooled_5seed_beam_median": {
                "err_median_model": float(np.median(model_5)),
                "n_wins_vs_const": int(np.sum(adv5 > 0)),
                "win_rate": float(np.mean(adv5 > 0)),
                "median_advantage": med, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0),
                "median_ratio_model_over_const": float(np.median(model_5 / const_v)),
                "p_wilcoxon": wilcoxon_p(adv5),
            },
        }
    out["const_compare"] = {
        "const_errs_snapshot_equal": const_snapshot_equal,
        "const_median": float(np.median(list(const_by_beam.values()))),
        "results": const_pairs,
    }

    # ---- 与 exp3（best_val 口径、逐种子）对账：n_wins / median_advantage 必须逐位一致
    cross = []
    for k, run in enumerate(JOINT_RUNS):
        mine = const_pairs["best_val"]["per_seed"][k]
        exp3_pairing = exp3["runs"][run]["pairing"]
        cross.append({
            "run_id": run,
            "seed": int(mine["seed"]),
            "n_wins_mine": mine["n_wins_vs_const"],
            "n_wins_exp3": int(exp3_pairing["n_wins"]),
            "median_advantage_mine": mine["median_advantage"],
            "median_advantage_exp3": float(exp3_pairing["median_advantage"]),
            "match": (
                mine["n_wins_vs_const"] == int(exp3_pairing["n_wins"])
                and abs(mine["median_advantage"] - float(exp3_pairing["median_advantage"])) < 1e-12
            ),
        })
    out["exp3_cross_check"] = {"rows": cross, "all_match": all(c["match"] for c in cross)}

    out["gates"] = gates_out
    (HERE / "pre2_gates_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 逐对明细 CSV
    with (HERE / "pre2_h3_err_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "seed", "err_joint", "err_indep", "run_level_winner",
                    "delta_median_beam", "ci_lo", "ci_hi", "ci_excludes_0",
                    "n_beams_joint_better", "p_wilcoxon"])
        for ckpt, block in h3_err.items():
            for r in block["per_seed"]:
                w.writerow([ckpt, r["seed"], f"{r['err_median_joint']:.9f}",
                            f"{r['err_median_independent']:.9f}", r["run_level_winner"],
                            f"{r['delta_median']:.9f}", f"{r['ci_lo']:.9f}", f"{r['ci_hi']:.9f}",
                            r["ci_excludes_0"], r["n_beams_joint_better"], r["p_wilcoxon"]])

    with (HERE / "pre2_const_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "seed", "err_median_model", "err_median_const", "n_wins_vs_const",
                    "win_rate", "median_advantage", "ci_lo", "ci_hi", "ci_excludes_0", "p_wilcoxon"])
        for ckpt, block in const_pairs.items():
            for r in block["per_seed"]:
                w.writerow([ckpt, r["seed"], f"{r['err_median_model']:.9f}", f"{r['err_median_const']:.9f}",
                            r["n_wins_vs_const"], f"{r['win_rate']:.4f}", f"{r['median_advantage']:.9f}",
                            f"{r['ci_lo']:.9f}", f"{r['ci_hi']:.9f}", r["ci_excludes_0"], r["p_wilcoxon"]])

    with (HERE / "pre2_curve_points.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "variant", "n_low", "n_runs", "err_median_of_seeds", "n_eff_median",
                    "ratio_median", "n_overflow", "n_underflow", "bare_at_tier",
                    "err_over_bare_at_tier", "err_over_bare_1e5"])
        for r in curve_rows:
            w.writerow([r["ckpt"], r["variant"], r["n_low"], r["n_runs"],
                        f"{r['err_median_of_seeds']:.9f}", f"{r['n_eff_median']:.6f}",
                        f"{r['ratio_median']:.6f}", r["n_overflow"], r["n_underflow"],
                        f"{r['bare_at_tier']:.9f}", f"{r['err_over_bare_at_tier']:.6f}",
                        f"{r['err_over_bare_1e5']:.6f}"])

    _write_summary(out, curve_rows)
    print(json.dumps({
        "official_reproduction_all_match": val_check["all_match"],
        "exp3_cross_check_all_match": out["exp3_cross_check"]["all_match"],
        "H1_last": gates_out["last"]["H1"]["verdict"],
        "H3_last_n_eff": gates_out["last"]["H3_n_eff"]["verdict"],
        "H3_last_err": h3_err["last"]["gates_expression_on_neg_err"]["verdict"],
        "H8_last_joint": gates_out["last"]["H8_joint"]["pass_rate"],
        "H8_bestval_joint": gates_out["best_val"]["H8_joint"]["pass_rate"],
        "H8_last_independent": gates_out["last"]["H8_independent"]["pass_rate"],
        "out_dir": str(HERE),
    }, ensure_ascii=False, indent=2))
    return 0


def _fmt(x) -> str:
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _write_summary(out: dict, curve_rows: list[dict]) -> None:
    """两列（best_val / last）汇总表：CSV + MD（同一份行列内容，两种落盘格式）。"""
    G = out["gates"]
    h3_err = out["H3_err_secondary"]
    const = out["const_compare"]

    rows: list[dict] = []

    def add(quantity: str, best, last, note: str = "") -> None:
        rows.append({"quantity": quantity, "best_val": best, "last": last, "note": note})

    # H1 / H3（官方 N_eff 口径）
    add("H1 判定（@1e4 joint，5 种子）", G["best_val"]["H1"]["verdict"], G["last"]["H1"]["verdict"],
        "官方表达式：5/5 > 1e4 且中位数 ≥ 2e4")
    add("H1 中位数 N_eff", G["best_val"]["H1"]["median_n_eff"], G["last"]["H1"]["median_n_eff"],
        "报告域顶 = 1e5（截断值）")
    add("H1 N_eff 上溢数 / 种子数",
        f"{sum(1 for v in G['best_val']['H1']['per_seed_flag'].values() if v == 'overflow')}/5",
        f"{sum(1 for v in G['last']['H1']['per_seed_flag'].values() if v == 'overflow')}/5")
    add("H3 判定（官方 N_eff 口径）",
        G["best_val"]["H3_n_eff"]["verdict"], G["last"]["H3_n_eff"]["verdict"], "平局计入 ≥（判据冻结）")
    add("H3 平局数 / 严格联合胜出数",
        f"{G['best_val']['H3_n_eff']['n_ties']} / {G['best_val']['H3_n_eff']['n_strict_joint_wins']}",
        f"{G['last']['H3_n_eff']['n_ties']} / {G['last']['H3_n_eff']['n_strict_joint_wins']}")
    add("H3 域顶平局标志 tied_at_domain_edge",
        G["best_val"]["H3_n_eff"]["tied_at_domain_edge"], G["last"]["H3_n_eff"]["tied_at_domain_edge"])

    # H3 次级准绳（err）
    add("H3 err 口径：联合更优的种子对数",
        f"{h3_err['best_val']['n_seeds_joint_better']}/5", f"{h3_err['last']['n_seeds_joint_better']}/5",
        "次级准绳（非预注册）；比较量 = 运行级 err 中位数")
    add("H3 err 口径：判定表达式（−err 输入 gates.evaluate_h3）",
        h3_err["best_val"]["gates_expression_on_neg_err"]["verdict"],
        h3_err["last"]["gates_expression_on_neg_err"]["verdict"], "越大越好取向由取负号保持")
    add("H3 err 口径：pooled 中位 Δerr（joint − indep）",
        f"{h3_err['best_val']['pooled_5seed_beam_mean']['delta_median']:+.6f}",
        f"{h3_err['last']['pooled_5seed_beam_mean']['delta_median']:+.6f}",
        "负 = 联合更好；5 种子束均值后 150 束配对 bootstrap")
    add("H3 err 口径：pooled 95% CI",
        f"[{h3_err['best_val']['pooled_5seed_beam_mean']['ci_lo']:+.6f}, "
        f"{h3_err['best_val']['pooled_5seed_beam_mean']['ci_hi']:+.6f}]",
        f"[{h3_err['last']['pooled_5seed_beam_mean']['ci_lo']:+.6f}, "
        f"{h3_err['last']['pooled_5seed_beam_mean']['ci_hi']:+.6f}]")
    add("H3 err 口径：pooled Wilcoxon p",
        h3_err["best_val"]["pooled_5seed_beam_mean"]["p_wilcoxon"],
        h3_err["last"]["pooled_5seed_beam_mean"]["p_wilcoxon"])

    # H8（官方聚合口径）
    add("H8 达标率（joint 臂，150 束 × 13 项）",
        G["best_val"]["H8_joint"]["pass_rate"], G["last"]["H8_joint"]["pass_rate"],
        "官方聚合：逐观测量先对 5 种子取中位数再判达标")
    add("H8 判定（joint 臂）", G["best_val"]["H8_joint"]["verdict"], G["last"]["H8_joint"]["verdict"],
        "≥0.80 成立 / ≤0.50 不成立 / 之间不可判定")
    add("H8 容差档众数（joint 臂）",
        G["best_val"]["H8_joint"]["tolerance_tier_mode"], G["last"]["H8_joint"]["tolerance_tier_mode"],
        f"档位分布 best_val {G['best_val']['H8_joint']['tolerance_tier_counts']} / "
        f"last {G['last']['H8_joint']['tolerance_tier_counts']}")
    add("H8 达标率（independent 臂，披露对照）",
        G["best_val"]["H8_independent"]["pass_rate"], G["last"]["H8_independent"]["pass_rate"])
    add("H8 判定（independent 臂）",
        G["best_val"]["H8_independent"]["verdict"], G["last"]["H8_independent"]["verdict"])

    gap_bv = out["H8_gap_to_pass_line"]["best_val_joint"]
    gap_lt = out["H8_gap_to_pass_line"]["last_joint"]
    add("H8 距 80% 线的 (束, 观测量) 对数缺口（joint 臂）",
        f"{gap_bv['n_pairs_pass']}/{gap_bv['n_pairs_total']}，差 {gap_bv['n_pairs_short']} 对",
        f"{gap_lt['n_pairs_pass']}/{gap_lt['n_pairs_total']}，差 {gap_lt['n_pairs_short']} 对",
        "达到 0.80 需 1560/1950 对")
    for arm, col in (("joint", "joint"), ("independent", "independent")):
        d_bv = out["H8_per_seed_disclosure"][f"best_val_{arm}"]
        d_lt = out["H8_per_seed_disclosure"][f"last_{arm}"]
        add(f"H8 逐种子达标率（{arm} 臂，披露：非官方聚合口径）",
            f"{d_bv['median_of_per_seed']:.4f}（{d_bv['min_of_per_seed']:.4f}–{d_bv['max_of_per_seed']:.4f}）",
            f"{d_lt['median_of_per_seed']:.4f}（{d_lt['min_of_per_seed']:.4f}–{d_lt['max_of_per_seed']:.4f}）",
            "容差线仍按束级 N_eff 中位数的档位映射；仅逐种子单独判达标")

    # 三点主曲线
    for n_low in (3000, 10000, 100000):
        bv_row = next(x for x in curve_rows
                      if x["variant"] == "joint" and x["n_low"] == n_low and x["ckpt"] == "best_val")
        lt_row = next(x for x in curve_rows
                      if x["variant"] == "joint" and x["n_low"] == n_low and x["ckpt"] == "last")
        add(f"主曲线 {n_low:g}（joint）：err 中位数（5 种子中位数）",
            bv_row["err_median_of_seeds"], lt_row["err_median_of_seeds"],
            f"该档裸误差 {lt_row['bare_at_tier']:.6f}；相对 1e5 档裸误差 {BARE_1E5:.6f} 的比值 "
            f"best_val {bv_row['err_over_bare_1e5']:.4f} → last {lt_row['err_over_bare_1e5']:.4f}")
        add(f"主曲线 {n_low:g}（joint）：N_eff 中位数 / 上溢数",
            f"{bv_row['n_eff_median']:.0f} / {bv_row['n_overflow']}",
            f"{lt_row['n_eff_median']:.0f} / {lt_row['n_overflow']}",
            f"放大比 N_eff/N_low 中位数 {lt_row['ratio_median']:.4f}（两口径同值，N_eff 未变）")
    bv_i = next(x for x in curve_rows if x["variant"] == "independent" and x["ckpt"] == "best_val")
    lt_i = next(x for x in curve_rows if x["variant"] == "independent" and x["ckpt"] == "last")
    add("主曲线 1e4（independent）：err 中位数 / N_eff 中位数 / 上溢数",
        f"{bv_i['err_median_of_seeds']:.6f} / {bv_i['n_eff_median']:.0f} / {bv_i['n_overflow']}",
        f"{lt_i['err_median_of_seeds']:.6f} / {lt_i['n_eff_median']:.0f} / {lt_i['n_overflow']}")

    # 模型 vs 常数解
    add("常数解 err 中位数（对照基准）", const["const_median"], const["const_median"],
        "150 束；exp3 快照与 phase1_numbers 一致 = "
        f"{const['const_errs_snapshot_equal']}")
    add("模型 vs 常数解：逐束赢率（5 种子束中位数，150 束）",
        f"{const['results']['best_val']['pooled_5seed_beam_median']['n_wins_vs_const']}/150 = "
        f"{const['results']['best_val']['pooled_5seed_beam_median']['win_rate']:.4f}",
        f"{const['results']['last']['pooled_5seed_beam_median']['n_wins_vs_const']}/150 = "
        f"{const['results']['last']['pooled_5seed_beam_median']['win_rate']:.4f}",
        "advantage = err_const − err_model > 0 记为赢")
    add("模型 vs 常数解：err 中位数（模型 / 常数）",
        f"{const['results']['best_val']['pooled_5seed_beam_median']['err_median_model']:.6f} / "
        f"{const['results']['best_val']['err_median_const']:.6f}",
        f"{const['results']['last']['pooled_5seed_beam_median']['err_median_model']:.6f} / "
        f"{const['results']['last']['err_median_const']:.6f}")
    add("模型 vs 常数解：中位 advantage（const − model，>0 模型更好）",
        f"{const['results']['best_val']['pooled_5seed_beam_median']['median_advantage']:+.6f}",
        f"{const['results']['last']['pooled_5seed_beam_median']['median_advantage']:+.6f}")
    add("模型 vs 常数解：advantage 95% CI",
        f"[{const['results']['best_val']['pooled_5seed_beam_median']['ci_lo']:+.6f}, "
        f"{const['results']['best_val']['pooled_5seed_beam_median']['ci_hi']:+.6f}]",
        f"[{const['results']['last']['pooled_5seed_beam_median']['ci_lo']:+.6f}, "
        f"{const['results']['last']['pooled_5seed_beam_median']['ci_hi']:+.6f}]")
    add("模型 vs 常数解：Wilcoxon p（双尾）",
        const["results"]["best_val"]["pooled_5seed_beam_median"]["p_wilcoxon"],
        const["results"]["last"]["pooled_5seed_beam_median"]["p_wilcoxon"])
    add("模型 vs 常数解：逐种子 err 中位数低于常数解的种子数",
        f"{const['results']['best_val']['n_seeds_median_err_below_const']}/5",
        f"{const['results']['last']['n_seeds_median_err_below_const']}/5")

    # 官方产物对账
    add("官方 e1_gates.json 逐位复现（best_val 列自身校验）",
        out["official_gates_reproduction"]["all_match"], "—",
        "H1/H3/H8 判定与数值全部一致")
    add("exp3 常数解对照对账（best_val 逐种子 n_wins / 中位 advantage）",
        out["exp3_cross_check"]["all_match"], "—",
        "5/5 run 与 exp3 产物一致，证明常数解映射与 err 配对口径一致")

    fields = ["quantity", "best_val", "last", "note"]
    with (HERE / "pre2_compare.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (_fmt(v) if k in ("best_val", "last") else v) for k, v in r.items()})

    lines = ["| 量 | best_val 口径（官方判定） | last 口径 | 备注 |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['quantity']} | {_fmt(r['best_val'])} | {_fmt(r['last'])} | {r['note']} |")
    (HERE / "pre2_compare_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
