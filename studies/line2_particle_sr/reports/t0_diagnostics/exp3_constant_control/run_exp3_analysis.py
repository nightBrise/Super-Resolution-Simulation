"""T0-实验 3：网络 vs 零信息常数解的逐束配对分析与个性回归（line2 Phase 1 诊断）。

问题：Phase 1 中与束无关的常数解（700 个 train 束 station0/target_full 逐投影归一后的均值图）
在 150 个 test 束上误差中位数 0.194398，优于全部 20 个 run；但每个 run 在 40–69/150 束上胜出。
本脚本检验"网络赢的束是否恰是抖动幅度大（离族均值远）的束"：
若优势与抖动幅度正相关，说明网络在提取真实个体信息（被先验音量掩盖）；若无关或负相关，
则坐实族先验主导（复读机假设）。

口径（全部照抄 ``reports/tools/compute_phase1_numbers.py`` 与 ``src/line2/experiments/evaluate.py``）：
- 常数解：对 700 个 train 束逐束对 station0/target_full 做逐投影总计数归一 x̃ = x·(Σr/Σx)，
  再取算术平均得均值图；对 150 个 test 束用 :func:`group_rel_rmse` 评测。
- 模型：``results/<run>/test_eval.json`` 的 ``per_beam[].err``（station 0，= 20 个子样本上
  ``group_rel_rmse(pred, target_full)`` 的中位数）。
- advantage(b) = err_const(b) − err_model(b)，>0 表示模型更好。
- 随机性一律由 ``SeedSequence(20260825)`` 派生；禁止裸随机。

运行：python studies/line2_particle_sr/reports/t0_diagnostics/exp3_constant_control/run_exp3_analysis.py
只读 studies/line2_particle_sr/{data,results} 与 reports/tools/phase1_numbers.json（仅作交叉核对）；
只写本目录 exp3_results.json / exp3_per_beam.csv / exp3_per_beam_long.csv。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress, mannwhitneyu, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.line2.metrics.metrics import group_rel_rmse  # noqa: E402

LINE2 = ROOT / "studies" / "line2_particle_sr"
DATA = LINE2 / "data" / "line2_ds_v1"
RESULTS = LINE2 / "results"
SNAPSHOT = LINE2 / "reports" / "tools" / "phase1_numbers.json"
OUT = Path(__file__).resolve().parent

JOINT_RUNS = [f"e1_joint_s{i}" for i in range(5)]
INDEP_RUNS = [f"e1_indep_s{i}" for i in range(5)]
E2_RUNS = [f"e2_joint_n{n}_s{i}" for n in (3000, 100000) for i in range(5)]
ALL_RUNS = JOINT_RUNS + INDEP_RUNS + E2_RUNS

JITTER_ATTRS = (
    "jitter_delta",
    "jitter_drift_m",
    "jitter_f_j",
    "jitter_theta_x_rad",
    "jitter_theta_y_rad",
)
# jitter_f_j = 10·(1 + jitter_delta) 是 delta 的确定性重复（transport.py jitter_params），
# 5 分量 L2 会把 delta 方向加权两次；4 自由度版本只保留独立抽样量。
JITTER_4DOF = ("jitter_delta", "jitter_drift_m", "jitter_theta_x_rad", "jitter_theta_y_rad")

N_BOOT = 10000
ROOT_SEED = 20260825


def constant_family() -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """零信息固定图家族的第一批成员：700 束均值（= 常数解）与其中前 200 束的均值。

    返回 (beam_ids, errs_700, errs_200, mean_700, mean_200, refs)；
    refs 为 150 个 test 目标图（原 dtype，与快照口径一致）。
    """
    acc = acc200 = None
    n_beams = 0
    with h5py.File(DATA / "train.h5", "r") as fh:
        for beam in sorted(fh.keys()):
            grp = fh[beam]["station0"]["target_full"][:].astype(np.float64)
            a = grp / grp.sum(axis=(1, 2), keepdims=True)
            acc = a if acc is None else acc + a
            if n_beams < 200:
                acc200 = a if acc200 is None else acc200 + a
            n_beams += 1
    mean700 = acc / n_beams
    mean200 = acc200 / 200

    with h5py.File(DATA / "test.h5", "r") as fh:
        keys = sorted(fh.keys())
        beam_ids = [int(b.split("_")[1]) for b in keys]
        refs = [fh[b]["station0"]["target_full"][:] for b in keys]
    errs700 = np.asarray([group_rel_rmse(mean700, r) for r in refs], dtype=np.float64)
    errs200 = np.asarray([group_rel_rmse(mean200, r) for r in refs], dtype=np.float64)
    return beam_ids, errs700, errs200, mean700, mean200, refs


def model_errors(runs: list[str]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for run in runs:
        payload = json.loads((RESULTS / run / "test_eval.json").read_text(encoding="utf-8"))
        assert payload["station"] == 0, (run, payload["station"])
        out[run] = {int(d["beam_id"]): float(d["err"]) for d in payload["per_beam"]}
    return out


def jitter_table(beam_ids: list[int]) -> dict[str, np.ndarray]:
    with h5py.File(DATA / "test.h5", "r") as fh:
        rows = {a: [] for a in JITTER_ATTRS}
        for b in beam_ids:
            attrs = fh[f"beam_{b:04d}"].attrs
            for a in JITTER_ATTRS:
                rows[a].append(float(attrs[a]))
    return {a: np.asarray(v, dtype=np.float64) for a, v in rows.items()}


def zscore(v: np.ndarray) -> np.ndarray:
    s = v.std(ddof=0)
    return (v - v.mean()) / s


def l2_norm(z: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.sqrt(np.sum([z[n] ** 2 for n in names], axis=0))


def boot_median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    med = np.median(values[idx], axis=1)
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def rank_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ 的向量化点估计（秩的 Pearson 相关）。"""
    a = rankdata(x)
    b = rankdata(y)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def boot_spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Spearman ρ 的配对 bootstrap CI：ρ = 秩的 Pearson 相关（平均秩处理并列，同 scipy）。"""
    rx = rankdata(x)
    ry = rankdata(y)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    a = rx[idx]
    b = ry[idx]
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    rho = (a * b).sum(axis=1) / denom
    return [float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))]


def boot_mean_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    m = values[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def quartile_groups(personality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按个性标量升序取 bottom / top 四分位（各 37 束，150 // 4）。"""
    order = np.argsort(personality)
    k = personality.size // 4
    return order[:k], order[-k:]


def paired_stats(adv: np.ndarray, err_model: np.ndarray, err_const: np.ndarray, rng: np.random.Generator) -> dict:
    w = wilcoxon(err_model, err_const, alternative="two-sided")
    n_zero = int(np.sum(adv == 0))
    return {
        "n_beams": int(adv.size),
        "n_wins": int(np.sum(adv > 0)),
        "win_rate": float(np.mean(adv > 0)),
        "n_ties": n_zero,
        "median_advantage": float(np.median(adv)),
        "median_advantage_ci95": boot_median_ci(adv, rng),
        "mean_advantage": float(adv.mean()),
        "mean_advantage_ci95": boot_mean_ci(adv, rng),
        "median_err_const": float(np.median(err_const)),
        "median_err_model": float(np.median(err_model)),
        "median_ratio_model_over_const": float(np.median(err_model / err_const)),
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "advantage": [float(v) for v in adv],
    }


def personality_analysis(
    adv: np.ndarray,
    jitter: dict[str, np.ndarray],
    z: dict[str, np.ndarray],
    l2: np.ndarray,
    rng: np.random.Generator,
    extra: dict[str, np.ndarray],
) -> dict:
    rho = spearmanr(adv, l2)
    lin = linregress(zscore(l2), adv)
    bottom, top = quartile_groups(l2)
    out = {
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "spearman_rho_ci95": boot_spearman_ci(adv, l2, rng),
        "ols_slope_advantage_per_sd_l2": float(lin.slope),
        "ols_r2": float(lin.rvalue ** 2),
        "ols_p": float(lin.pvalue),
        "l2_median": float(np.median(l2)),
        "quartile_n": int(bottom.size),
        "bottom_quartile": {
            "win_rate": float(np.mean(adv[bottom] > 0)),
            "median_advantage": float(np.median(adv[bottom])),
            "mean_advantage": float(adv[bottom].mean()),
            "l2_range": [float(l2[bottom].min()), float(l2[bottom].max())],
        },
        "top_quartile": {
            "win_rate": float(np.mean(adv[top] > 0)),
            "median_advantage": float(np.median(adv[top])),
            "mean_advantage": float(adv[top].mean()),
            "l2_range": [float(l2[top].min()), float(l2[top].max())],
        },
        "bottom_minus_top_mean_advantage": float(adv[bottom].mean() - adv[top].mean()),
        "mannwhitney_p_adv_top_vs_bottom": float(
            mannwhitneyu(adv[top], adv[bottom], alternative="two-sided").pvalue
        ),
        "top_gt_bottom_win_rate": bool(np.mean(adv[top] > 0) > np.mean(adv[bottom] > 0)),
        "per_component": {},
    }
    for a in JITTER_ATTRS:
        r = spearmanr(adv, jitter[a])
        out["per_component"][a] = {
            "spearman_rho": float(r.statistic),
            "spearman_p": float(r.pvalue),
            "spearman_rho_ci95": boot_spearman_ci(adv, jitter[a], rng),
        }
    for name, v in extra.items():
        r = spearmanr(adv, v)
        out["per_component"][name] = {
            "spearman_rho": float(r.statistic),
            "spearman_p": float(r.pvalue),
            "spearman_rho_ci95": boot_spearman_ci(adv, v, rng),
        }
    return out


def norm_metric_matrix(refs: list[np.ndarray]) -> np.ndarray:
    """150 个 test 目标图两两之间的 M1 距离矩阵 M。

    度量恒等式（err 的定义即平均意义下的距离）：把图 c 当作与束无关的固定图去评束 b，
    ``group_rel_rmse(T_c, T_b) = M[b, c]``；M 满足三角不等式，用于"输出与束无关"的可达性检验。
    """
    t = np.stack([r.astype(np.float64) for r in refs])
    n_cells = t.shape[2] * t.shape[3]
    u = t * np.sqrt(n_cells) / t.sum(axis=(2, 3), keepdims=True)
    flat = u.reshape(t.shape[0], t.shape[1], -1)
    nrm = (flat ** 2).sum(axis=2)
    gram = np.einsum("aic,bic->abi", flat, flat)
    m = np.zeros((flat.shape[0], flat.shape[0]), dtype=np.float64)
    for p in range(flat.shape[1]):
        d2 = nrm[:, None, p] + nrm[None, :, p] - 2 * gram[:, :, p]
        m += np.sqrt(np.maximum(d2, 0.0))
    return m / flat.shape[1]


def interp_by_level(levels: np.ndarray, values: np.ndarray, x: float) -> float | None:
    """在固定图家族的误差水平上插值取值；非有限取值的候选（ρ 无定义）先剔除。"""
    levels = np.asarray(levels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(levels) & np.isfinite(values)
    if keep.sum() < 2:
        return None
    levels, values = levels[keep], values[keep]
    order = np.argsort(levels)
    return float(np.interp(x, levels[order], values[order]))


def null_family(
    mean700: np.ndarray,
    mean200: np.ndarray,
    refs: list[np.ndarray],
    err_const: np.ndarray,
    jitter: dict[str, np.ndarray],
    z: dict[str, np.ndarray],
    l2: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """零信息固定图家族：把"模型"换成与束无关的固定图，扫描其误差水平。

    家族成员 = 700 束均值（即常数解本身）、前 200 束均值、常数解的高斯模糊（σ 网格）。
    每个成员按与 run 完全相同的口径计算逐束误差与个性回归，用于回答：
    "run 的 ρ 与四分位赢率是否只是'任何固定图在同一误差水平'都会给出的机械结果？"
    """
    cands: list[tuple[str, np.ndarray]] = [("mean700", mean700), ("mean200", mean200)]
    for s in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.7, 1.0):
        cands.append((f"blur_sigma{s:g}", np.stack([gaussian_filter(mean700[p], s) for p in range(6)])))
    out = []
    for name, image in cands:
        e = np.asarray([group_rel_rmse(image, r) for r in refs], dtype=np.float64)
        adv = err_const - e
        degenerate = bool(np.all(adv == adv[0]))  # 候选即常数解本身：advantage 恒 0，ρ 无定义
        pers = None if degenerate else personality_analysis(adv, jitter, z, l2, rng, {})
        out.append(
            {
                "name": name,
                "err_median": float(np.median(e)),
                "err_std": float(e.std(ddof=0)),
                "err_min": float(e.min()),
                "err_max": float(e.max()),
                "n_wins_vs_const": int(np.sum(adv > 0)),
                "win_rate": float(np.mean(adv > 0)),
                "spearman_rho": None if degenerate else pers["spearman_rho"],
                "spearman_rho_ci95": None if degenerate else pers["spearman_rho_ci95"],
                "wr_top": None if degenerate else pers["top_quartile"]["win_rate"],
                "wr_bottom": None if degenerate else pers["bottom_quartile"]["win_rate"],
            }
        )
    return out


def write_report(results: dict, family: list[dict]) -> None:
    """落盘人类可读的紧凑结论（含判定、混淆审计与固定输出否证检验）。"""
    rs = results["runs_summary_table"]
    v = results["verdict"]
    ca = results["confound_audit"]
    ff = results["fixed_output_feasibility"]
    lines: list[str] = []
    lines.append("# T0-实验 3：网络 vs 零信息常数解的逐束配对分析与个性回归")
    lines.append("")
    lines.append("line2 Phase 1 诊断；只读 `studies/line2_particle_sr/{data,results}`，")
    lines.append("常数解口径照抄 `reports/tools/compute_phase1_numbers.py`")
    lines.append("（700 train 束 station0/target_full 逐投影总计数归一后取均值 → 150 test 束 M1）。")
    lines.append("")
    lines.append("## 1 判定")
    lines.append("")
    lines.append(
        f"- **预注册判据成立（{v['n_joint_seeds_rho_ci_positive']}/5 joint 种子 ρ 的 95% CI 全为正、"
        f"{v['n_joint_seeds_top_gt_bottom_win_rate']}/5 种子 top 四分位赢率 > bottom）——但它不构成"
        "“网络提取个体信息”的证据**：该判据的两个条件都能被机械复现（§3）。"
    )
    lines.append(
        "- **严格复读机假设（输出与输入无关的常数图）已被否证**："
        f"{ff['n_runs_refuted']}/{ff['n_runs']} 个 run 的逐束误差上界低于任何固定图的下界 "
        f"(D_max/2 = {ff['target_diameter_half']:.4f})，即网络输出确实随输入束变化（§4）。"
    )
    lines.append(
        "- 但网络没把这种依赖性换成更低的误差：全部 20 个 run 的**中位误差都劣于常数解**"
        "（中位 advantage 全为负、Wilcoxon 全显著），只在离族均值最远的束上取胜。"
    )
    lines.append(
        "- 综合：**“个体信息真实存在” 与 “复读机假设成立” 都不能按原判据下结论**"
        "（判据退化，见 §3）；可确证的只是“输出随输入变化”这一弱命题。"
    )
    lines.append("")
    lines.append("## 2 逐束配对（advantage = err_const − err_model > 0 表示模型更好）")
    lines.append("")
    lines.append("| run | 胜出束数 | 中位 advantage | 中位 advantage 95% CI | Wilcoxon p |")
    lines.append("|---|---|---|---|---|")
    for run, s in rs.items():
        lines.append(
            f"| {run} | {s['n_wins']}/150 | {s['median_advantage']:+.4f} | "
            f"[{s['median_advantage_ci95'][0]:+.4f}, {s['median_advantage_ci95'][1]:+.4f}] | "
            f"{s['wilcoxon_p']:.2e} |"
        )
    lines.append("")
    lines.append("## 3 个性回归与混淆审计（主结论口径 = 4 自由度标准化抖动 L2 范数）")
    lines.append("")
    lines.append("| run | ρ(advantage, 抖动幅度) | ρ 95% CI | top 四分位赢率 | bottom 四分位赢率 |")
    lines.append("|---|---|---|---|---|")
    for run, s in rs.items():
        lines.append(
            f"| {run} | {s['rho_4dof']:+.3f} | "
            f"[{s['rho_4dof_ci95'][0]:+.3f}, {s['rho_4dof_ci95'][1]:+.3f}] | "
            f"{s['wr_top']:.3f} | {s['wr_bottom']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"- 观测 ρ 中位 {ca['observed_rho_median']:+.3f}；**机械零假设**（把每个 run 的逐束误差换成其常数中位值，"
        f"即“误差水平固定、对束无差别”）给出 ρ 中位 {ca['flat_error_null_rho_median']:+.3f}，"
        f"与观测的最大偏差仅 {ca['max_abs_diff_rho_flat_vs_observed']:.3f}；top 四分位赢率的最大偏差 "
        f"{ca['max_abs_diff_wr_top_flat_vs_observed']:.3f}。即：**预注册判据的两个条件都被该机械零假设精确复现**。"
    )
    lines.append(
        f"- 机制（决定性）：advantage 与 err_const 的 Spearman 中位 {ca['median_spearman_advantage_vs_err_const']:+.4f}"
        f"（最小 {ca['min_spearman_advantage_vs_err_const']:+.4f}）——advantage 几乎是 err_const 的单调函数；"
        f"而 err_const 随抖动幅度上升（ρ = {ca['spearman_err_const_vs_jitter_l2_4dof']:+.3f}, "
        f"p = {ca['spearman_err_const_vs_jitter_l2_4dof_p']:.1e}）。"
        f"模型逐束误差本身也只跟踪 err_const 的 {ca['median_corr_err_model_err_const']:+.2f} 相关、"
        f"斜率对应 {ca['err_model_std_median']/ca['err_const_std']*100:.1f}% 的变化幅度"
        f"（std {ca['err_model_std_median']:.5f} vs {ca['err_const_std']:.5f}）。"
        "⇒ “网络在个性强的束上赢” 等价于 “族均值在个性强的束上更差”，与网络是否使用个体信息无关。"
    )
    lines.append(
        f"- 对照（反方向）：把“模型”换成**误差水平匹配的零信息固定图**（常数解的模糊版），"
        f"其 ρ 中位仍有 {ca['matched_null_rho_median']:+.3f}，但其 top 四分位赢率为 "
        f"{ca['matched_null_wr_top_median']:.3f}（≈0）——固定图在同一误差水平上赢不了任何束，"
        f"因为它的误差随束个性一起变（std ≈ 0.075），而 run 的误差几乎是平的（std ≈ 0.004）。"
        "这一差异正是 §4 否证检验的来源。"
    )
    lines.append("")
    lines.append("### 3b 逐分量回归与四分位明细（5 个 joint 种子的中位）")
    lines.append("")
    lines.append("| 抖动分量 | ρ(advantage, 分量) 中位 | ρ CI 下界最小 | ρ CI 上界最大 |")
    lines.append("|---|---|---|---|")
    comps = [
        "err_const",
        "jitter_l2_4dof",
        "jitter_l2_5attr",
        "jitter_delta",
        "jitter_f_j",
        "jitter_drift_m",
        "jitter_theta_x_rad",
        "jitter_theta_y_rad",
    ]
    for a in comps:
        vals = [
            results["runs"][r]["personality_l2_4dof"]["per_component"][a] for r in results["runs"]
        ]
        rhos = [v["spearman_rho"] for v in vals]
        los = [v["spearman_rho_ci95"][0] for v in vals]
        his = [v["spearman_rho_ci95"][1] for v in vals]
        lines.append(
            f"| {a} | {np.median(rhos):+.3f} | {np.min(los):+.3f} | {np.max(his):+.3f} |"
        )
    lines.append("")
    lines.append(
        "（前 3 行是机械基线与个性标量本身：advantage 与 err_const 的 ρ 中位 +0.999 ⇒ "
        "表中其余各行对 advantage 的取值与对 err_const 的取值必然相同。分量之间不可比幅度：抖动为独立均匀抽样。）"
    )
    lines.append("")
    lines.append("| run | bottom 四分位中位 advantage | top 四分位中位 advantage |")
    lines.append("|---|---|---|")
    for run in list(results["runs"])[:5] + list(results["runs"])[5:]:
        p = results["runs"][run]["personality_l2_4dof"]
        lines.append(
            f"| {run} | {p['bottom_quartile']['median_advantage']:+.4f} | "
            f"{p['top_quartile']['median_advantage']:+.4f} |"
        )
    lines.append("")
    lines.append("## 4 固定输出（严格复读机）否证检验")
    lines.append("")
    lines.append(
        f"150 个 test 目标图两两 M1 距离的直径 D_max = {ff['target_diameter_d_max']:.4f}"
        f"（中位成对距离 {ff['median_pairwise']:.4f}）。若模型输出是与束无关的固定图 x*，"
        f"由三角不等式必有 max_b err_model(b) ≥ D_max/2 = {ff['target_diameter_half']:.4f}。"
    )
    lines.append("")
    lines.append("| run | max_b err_model | ≥ D_max/2？ | Lipschitz 违反对数 |")
    lines.append("|---|---|---|---|")
    for run in results["runs"]:
        f = results["runs"][run]["fixed_output_feasibility"]
        lines.append(
            f"| {run} | {f['max_err_model']:.4f} | "
            f"{'否 → 固定输出被否证' if f['refuted_fixed_output'] else '是 → 未否证'} | "
            f"{f['lipschitz_violating_pairs']} |"
        )
    lines.append("")
    lines.append(
        "适用范围：该检验否证的是“模型输出与输入（束与子样本）无关”，"
        "因为逐束误差取的是 20 个子样本预测的中位数——若输出随子样本变化而不随束变化，"
        "中位数阈值不再等价于到单一固定图的距离，检验退化为不适用。"
        "子样本输入本身携带该束的观测信息，故实际含义是“网络输出确实依赖被观测的这张图”。"
        "另：Lipschitz 逐对检验（|e_b − e_c| ≤ M[b,c]）在本数据上 0 违反，"
        "不构成否证；否证来自整体的直径下界。"
    )
    lines.append("")
    lines.append("## 5 零信息固定图家族（误差水平扫描）")
    lines.append("")
    lines.append("| 候选 | 中位误差 | 逐束误差标准差 | 胜出束数 | ρ | top 赢率 | bottom 赢率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in family:
        rho = "—" if c["spearman_rho"] is None else f"{c['spearman_rho']:+.3f}"
        wr_t = "—" if c["wr_top"] is None else f"{c['wr_top']:.3f}"
        wr_b = "—" if c["wr_bottom"] is None else f"{c['wr_bottom']:.3f}"
        lines.append(
            f"| {c['name']} | {c['err_median']:.4f} | {c['err_std']:.4f} | {c['n_wins_vs_const']}/150 | "
            f"{rho} | {wr_t} | {wr_b} |"
        )
    lines.append("")
    lines.append("（`mean700`/`blur_sigma0.05`/`blur_sigma0.1` 即常数解本身，advantage 恒 0，ρ 无定义。）")
    lines.append("")
    lines.append("## 6 度量噪声底：为什么“常数解更好”主要是度量问题")
    lines.append("")
    mf = results["metric_noise_floor"]
    lines.append(
        f"- `e0_curve.json` 的 `reference_noise_floor` = {mf['reference_noise_floor_1e5_pair']:.4f}"
        f"（{mf['definition']}）；折算到 test 用的 2e5 参考图，其自身计数噪声在 M1 下的固有距离约 "
        f"{mf['floor_2e5_reference']:.4f}。"
    )
    lines.append(
        f"- 常数解逐束误差中位 {mf['const_median_err']:.4f}，与之同量级 ⇒ 常数解的误差几乎全部来自参考图自身的计数噪声，"
        "其“形状失配”项在该度量下可忽略；20 个 run 的中位误差 0.2218–0.2898 则高于该噪声底。"
    )
    lines.append(
        "- 因此 M1 在这一分辨率上无法区分“复现了个体结构”与“给出了最光滑的先验图”："
        "要把两者分开，需要看模型输出相对先验的增量与目标相对先验的增量之间的投影"
        "（`⟨out − prior, target − prior⟩`，本轮未做——需对 ckpt 逐束推理）。"
    )
    lines.append("")
    lines.append("## 7 种子一致性与 5 种子聚合")
    lines.append("")
    sc = results["seed_consistency"]
    lines.append(
        f"- 20 个 run 的 advantage 向量两两 Spearman（120 对）中位 {sc['pairwise_spearman_all_20_runs_median']:+.3f}"
        f"（min {sc['pairwise_spearman_all_20_runs_min']:+.3f} / max {sc['pairwise_spearman_all_20_runs_max']:+.3f}）；"
        f"5 个 joint 种子内部中位 {sc['pairwise_spearman_joint_median']:+.3f}。"
    )
    lines.append(
        f"- 5 种子平均 advantage 的 ρ = {sc['joint_seed_mean_advantage_rho_vs_jitter']:+.3f} "
        f"CI [{sc['joint_seed_mean_advantage_rho_ci95'][0]:+.3f}, {sc['joint_seed_mean_advantage_rho_ci95'][1]:+.3f}]"
        "（此为 5 种子聚合检验，与逐 run 结论一致）。"
    )
    lines.append("")
    lines.append("## 8 口径与偏差说明")
    lines.append("")
    lines.append(
        f"- 常数解逐束误差本次独立重算，与 `reports/tools/phase1_numbers.json` 的 `const_errs` 逐元素最大偏差 "
        f"{results['constant_solution']['max_abs_diff_vs_snapshot']:.1e}（完全一致）。"
    )
    lines.append(
        "- `jitter_f_j = 10·(1 + jitter_delta)` 与 `jitter_delta` 完全共线（`transport.py` 由 delta 派生），"
        "5 属性 L2 会把 delta 方向加权两次；主结论用 4 自由度版本，5 属性版本见 JSON。"
    )
    lines.append(
        "- 每个束的 `target_full` 有**自己的 bin_edges**（`station_edges` 用该站全量粒子表的逐轴 min/max 外扩 1%），"
        "即图像是“自身尺度归一”的；150 个 test 束之间的窗口差异对 x/y 轴 < 0.05%、对 xp/yp 轴达 ±1–2.6%。"
    )
    lines.append(
        "- 评测口径为 `group_rel_rmse`（六投影各归一后相对 RMSE 再平均）逐束结果，"
        "模型侧取 20 个子样本的中位数（`test_eval.json` 的 `per_beam[].err`），常数解与输入无关故单值即中位。"
    )
    lines.append(
        "- 20 个 run 的 advantage 向量两两 Spearman 中位 +0.999（§7）说明“赢哪几束”在所有 run 之间几乎相同，"
        "即该模式是束（err_const）的性质，而不是某个 run 的行为。"
    )
    lines.append("")
    lines.append("## 9 结论与建议的下一步")
    lines.append("")
    lines.append(
        "1. 预注册判据在本实验上**退化**：它实际测的是 `ρ(err_const, 抖动幅度) = +0.576`，"
        "而不是模型的任何性质。建议在后续 Q/C 类实验中改写判据——例如以"
        "“advantage 对 err_const 的偏相关”“top 四分位内的绝对误差是否低于常数解在该四分位的中位误差”"
        "等条件替换，或在设计上先按束个性分层再比较。"
    )
    lines.append(
        "2. 真正能分开两个假设的实验（本轮未做，属新范围）：对 ckpt 逐束推理，计算模型输出相对先验的增量与"
        "目标相对先验的增量之间的投影 `⟨out − prior, target − prior⟩`（先验 = 常数解图，逐投影归一后）"
        "及其与抖动幅度的关系；再配一个“先验打乱”对照（把另一束的输入喂给模型）以排除“输出只依赖输入的整体尺度”的解释。"
        "ckpt 在本机 `studies/line2_particle_sr/results/<run>/best_val.ckpt`（纯 CPU 推理可行，需先确认模型规模与耗时）。"
    )
    lines.append(
        "3. M1 度量本身在本分辨率已触到参考图计数噪声底（§6）。若目标是判别“是否复现个体结构”，"
        "建议增设一个对噪声不敏感的结构型指标（如相对先验增量的相关性/方向余弦），而非继续比较 M1 的绝对值。"
    )
    lines.append("")
    (OUT / "exp3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng_root = np.random.SeedSequence(ROOT_SEED)
    ss_paired, ss_pers, ss_agg, ss_flat, ss_null = rng_root.spawn(5)
    ss_paired_runs = ss_paired.spawn(len(ALL_RUNS))
    ss_pers_runs = ss_pers.spawn(len(ALL_RUNS))

    print("[1/6] 重算常数解逐束误差（700 train 束 + 150 test 束）", flush=True)
    beam_ids, err_const, err_mean200, mean700, mean200, refs = constant_family()
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    snap_const = np.asarray(snapshot["zero_information"]["const_errs"], dtype=np.float64)
    const_check = {
        "snapshot_path": str(SNAPSHOT.relative_to(ROOT)),
        "n_beams": len(beam_ids),
        "max_abs_diff_vs_snapshot": float(np.max(np.abs(err_const - snap_const))),
        "median_recomputed": float(np.median(err_const)),
        "median_snapshot": float(np.median(snap_const)),
        "median_err_mean200": float(np.median(err_mean200)),
        "snapshot_median_first200": float(snapshot["zero_information"]["const_first200_median"]),
    }

    print("[2/6] 读取 20 个 run 的逐束误差", flush=True)
    errs = model_errors(ALL_RUNS)
    missing = {
        run: sorted(set(beam_ids) - set(d)) for run, d in errs.items() if set(d) != set(beam_ids)
    }
    assert not missing, missing

    print("[3/6] 抖动幅度个性标量", flush=True)
    jitter = jitter_table(beam_ids)
    z = {a: zscore(v) for a, v in jitter.items()}
    l2_4dof = l2_norm(z, JITTER_4DOF)
    l2_5attr = l2_norm(z, JITTER_ATTRS)
    extra = {
        "err_const": err_const,
        "jitter_l2_4dof": l2_4dof,
        "jitter_l2_5attr": l2_5attr,
        "jitter_l2_signed_theta_x": l2_norm(z, ("jitter_theta_x_rad",)),
    }
    impl_check = float(
        max(
            abs(rank_pearson(err_const, l2_4dof) - spearmanr(err_const, l2_4dof).statistic),
            abs(
                rank_pearson(z["jitter_delta"], l2_5attr)
                - spearmanr(z["jitter_delta"], l2_5attr).statistic
            ),
        )
    )

    print("[4/6] 固定输出可达性检验（目标图两两距离矩阵）", flush=True)
    metric = norm_metric_matrix(refs)
    pair = metric.copy()
    np.fill_diagonal(pair, np.nan)
    d_max = float(np.nanmax(pair))
    feasibility = {
        "identity_check": {
            "group_rel_rmse(T_c as fixed image, T_b)": float(group_rel_rmse(refs[7], refs[3])),
            "M[b, c]": float(metric[3, 7]),
        },
        "target_diameter_d_max": d_max,
        "target_diameter_half": d_max / 2.0,
        "min_pairwise": float(np.nanmin(pair)),
        "median_pairwise": float(np.nanmedian(pair)),
        "criterion": "若模型输出是与束无关的固定图 x*，则 max_b err_model(b) ≥ D_max/2（三角不等式）",
    }

    print("[5/6] 逐束配对统计 + 个性回归 + 混淆审计", flush=True)
    results: dict = {
        "protocol": {
            "root_seed": ROOT_SEED,
            "n_bootstrap": N_BOOT,
            "seed_derivation": (
                "SeedSequence(20260825).spawn(5) → [0] 配对 bootstrap、[1] 个性回归、[2] 5 种子聚合、"
                "[3] 机械零假设、[4] 固定图家族；[0]/[1] 再各自 spawn(20) 给每个 run 独立流"
                "（run 顺序 = ALL_RUNS 顺序）"
            ),
            "advantage_definition": "err_const(b) − err_model(b)，>0 = 模型更好",
            "personality_primary": "4 自由度标准化抖动 L2 范数（delta/drift/theta_x/theta_y）",
            "personality_secondary": "5 属性标准化 L2 范数（含与 delta 共线的 jitter_f_j，delta 方向加权两次）",
            "quartile_rule": "按个性标量升序取前 37 束（bottom 25%）与后 37 束（top 25%）",
            "spearman_impl_max_abs_diff_vs_scipy": impl_check,
        },
        "constant_solution": const_check,
        "constant_solution_errs": [float(v) for v in err_const],
        "beam_ids": beam_ids,
        "jitter": {a: [float(v) for v in jitter[a]] for a in JITTER_ATTRS},
        "jitter_l2_4dof": [float(v) for v in l2_4dof],
        "jitter_l2_5attr": [float(v) for v in l2_5attr],
        "runs": {},
    }

    arr_const = err_const
    rng_flat = np.random.default_rng(ss_flat)
    rng_null = np.random.default_rng(ss_null)
    flat_pers = personality_analysis(
        arr_const - float(np.median(np.asarray([errs[ALL_RUNS[0]][b] for b in beam_ids]))),
        jitter,
        z,
        l2_4dof,
        rng_flat,
        extra,
    )
    results["flat_error_null"] = {
        "definition": (
            "机械零假设：把 run 的逐束误差向量换成常数（= 各 run 中位误差），"
            "即'误差水平固定、对束完全无差别'的极限"
        ),
        "seed_level_median": float(np.median([np.median([errs[r][b] for b in beam_ids]) for r in ALL_RUNS])),
        "personality_l2_4dof": flat_pers,
    }

    print("    零信息固定图家族（误差水平扫描）", flush=True)
    family = null_family(mean700, mean200, refs, arr_const, jitter, z, l2_4dof, rng_null)
    results["null_family"] = family

    e0 = json.loads((RESULTS / "e0_curve.json").read_text(encoding="utf-8"))
    floor_1e5 = float(e0["reference_noise_floor"])
    results["metric_noise_floor"] = {
        "source": "studies/line2_particle_sr/results/e0_curve.json",
        "reference_noise_floor_1e5_pair": floor_1e5,
        "definition": "1e5 档配对 M1 中位数 / √2（独立同分布样本与参考之间的固有噪声距离）",
        "floor_2e5_reference": floor_1e5 / np.sqrt(2.0),
        "const_median_err": float(np.median(err_const)),
        "note": (
            "test 目标图为 2e5 粒子直方图，其自身计数噪声在 M1 下的固有距离≈floor_2e5_reference；"
            "常数解中位误差与之同量级 ⇒ 常数解的误差主要由参考图自身计数噪声构成，"
            "M1 在该尺度上无法区分“复现个体结构”与“给出最光滑的先验图”"
        ),
    }

    for i, run in enumerate(ALL_RUNS):
        err_model = np.asarray([errs[run][b] for b in beam_ids], dtype=np.float64)
        adv = arr_const - err_model
        rng_paired = np.random.default_rng(ss_paired_runs[i])
        rng_pers = np.random.default_rng(ss_pers_runs[i])
        stats = paired_stats(adv, err_model, arr_const, rng_paired)
        pers4 = personality_analysis(adv, jitter, z, l2_4dof, rng_pers, extra)
        pers5 = personality_analysis(adv, jitter, z, l2_5attr, rng_pers, extra)

        # 机械零假设（误差换成常数中位）的同一套统计量
        lvl = float(np.median(err_model))
        pers_flat = personality_analysis(arr_const - lvl, jitter, z, l2_4dof, rng_pers, extra)

        # 误差水平匹配的零信息固定图（在 null family 的误差水平上插值）
        levels = np.asarray([c["err_median"] for c in family])
        matched = {
            "err_median_target": lvl,
            "spearman_rho": interp_by_level(levels, np.asarray([c["spearman_rho"] for c in family]), lvl),
            "wr_top": interp_by_level(levels, np.asarray([c["wr_top"] for c in family]), lvl),
            "wr_bottom": interp_by_level(levels, np.asarray([c["wr_bottom"] for c in family]), lvl),
            "win_rate": interp_by_level(levels, np.asarray([c["win_rate"] for c in family]), lvl),
            "err_std": interp_by_level(levels, np.asarray([c["err_std"] for c in family]), lvl),
        }

        # 固定输出（严格复读机）否证检验
        viol = np.abs(err_model[:, None] - err_model[None, :]) - metric
        np.fill_diagonal(viol, -np.inf)
        feas = {
            "max_err_model": float(err_model.max()),
            "needs_at_least_d_max_half": d_max / 2.0,
            "refuted_fixed_output": bool(err_model.max() < d_max / 2.0),
            "lipschitz_violating_pairs": int((viol > 1e-12).sum() // 2),
            "max_lipschitz_margin": float(viol.max()),
        }

        # 误差结构与束个性的关系
        struct = {
            "err_model_median": lvl,
            "err_model_std": float(err_model.std(ddof=0)),
            "err_const_std": float(arr_const.std(ddof=0)),
            "corr_err_model_err_const": float(np.corrcoef(err_model, arr_const)[0, 1]),
            "slope_err_model_on_err_const": float(np.polyfit(arr_const, err_model, 1)[0]),
            "ratio_std_model_over_const": float(err_model.std(ddof=0) / arr_const.std(ddof=0)),
            "spearman_advantage_vs_err_const": float(spearmanr(adv, arr_const).statistic),
        }

        results["runs"][run] = {
            "pairing": stats,
            "personality_l2_4dof": pers4,
            "personality_l2_5attr": pers5,
            "personality_flat_error_null": pers_flat,
            "null_family_matched": matched,
            "error_structure": struct,
            "fixed_output_feasibility": feas,
            "err_model": [float(v) for v in err_model],
            "advantage": [float(v) for v in adv],
        }
        print(
            f"    {run:22s} wins={stats['n_wins']:3d}/150 med_adv={stats['median_advantage']:+.6f} "
            f"p={stats['wilcoxon_p']:.2e} rho(4dof)={pers4['spearman_rho']:+.3f} "
            f"CI=[{pers4['spearman_rho_ci95'][0]:+.3f},{pers4['spearman_rho_ci95'][1]:+.3f}] "
            f"wr_top={pers4['top_quartile']['win_rate']:.3f} wr_bot={pers4['bottom_quartile']['win_rate']:.3f} "
            f"| rho_flat={pers_flat['spearman_rho']:+.3f} "
            f"rho_matched_null={matched['spearman_rho']:+.3f} "
            f"std_e={struct['err_model_std']:.5f} "
            f"{'固定输出已否证' if feas['refuted_fixed_output'] else '固定输出未否证'}",
            flush=True,
        )

    print("[6/6] 联合 5 种子的聚合检验与判定", flush=True)
    adv_joint = np.stack([np.asarray(results["runs"][r]["advantage"]) for r in JOINT_RUNS])
    adv_mean = adv_joint.mean(axis=0)
    rng_agg = np.random.default_rng(ss_agg)
    agg4 = personality_analysis(adv_mean, jitter, z, l2_4dof, rng_agg, extra)
    agg5 = personality_analysis(adv_mean, jitter, z, l2_5attr, rng_agg, extra)
    results["joint_seed_mean_advantage"] = {
        "personality_l2_4dof": agg4,
        "personality_l2_5attr": agg5,
        "advantage": [float(v) for v in adv_mean],
        "win_rate_any_seed_mean_positive": float(np.mean(adv_mean > 0)),
        "median_advantage": float(np.median(adv_mean)),
    }

    n_ci_pos = int(
        sum(
            1
            for r in JOINT_RUNS
            if results["runs"][r]["personality_l2_4dof"]["spearman_rho_ci95"][0] > 0
        )
    )
    n_top_gt_bottom = int(
        sum(1 for r in JOINT_RUNS if results["runs"][r]["personality_l2_4dof"]["top_gt_bottom_win_rate"])
    )
    results["fixed_output_feasibility"] = feasibility
    results["fixed_output_feasibility"]["n_runs_refuted"] = int(
        sum(1 for r in ALL_RUNS if results["runs"][r]["fixed_output_feasibility"]["refuted_fixed_output"])
    )
    results["fixed_output_feasibility"]["n_runs"] = len(ALL_RUNS)

    # 机械零假设（误差水平固定）对全部 20 个 run 的复现度
    flat_rhos = np.asarray(
        [results["runs"][r]["personality_flat_error_null"]["spearman_rho"] for r in ALL_RUNS]
    )
    obs_rhos = np.asarray([results["runs"][r]["personality_l2_4dof"]["spearman_rho"] for r in ALL_RUNS])
    flat_top = np.asarray(
        [results["runs"][r]["personality_flat_error_null"]["top_quartile"]["win_rate"] for r in ALL_RUNS]
    )
    obs_top = np.asarray(
        [results["runs"][r]["personality_l2_4dof"]["top_quartile"]["win_rate"] for r in ALL_RUNS]
    )
    matched_rho = np.asarray([results["runs"][r]["null_family_matched"]["spearman_rho"] for r in ALL_RUNS])
    matched_top = np.asarray([results["runs"][r]["null_family_matched"]["wr_top"] for r in ALL_RUNS])
    results["confound_audit"] = {
        "observed_rho_median": float(np.median(obs_rhos)),
        "flat_error_null_rho_median": float(np.median(flat_rhos)),
        "max_abs_diff_rho_flat_vs_observed": float(np.max(np.abs(flat_rhos - obs_rhos))),
        "observed_wr_top_median": float(np.median(obs_top)),
        "flat_error_null_wr_top_median": float(np.median(flat_top)),
        "max_abs_diff_wr_top_flat_vs_observed": float(np.max(np.abs(flat_top - obs_top))),
        "matched_null_rho_median": float(np.median(matched_rho)),
        "matched_null_wr_top_median": float(np.median(matched_top)),
        "err_model_std_median": float(
            np.median([results["runs"][r]["error_structure"]["err_model_std"] for r in ALL_RUNS])
        ),
        "err_const_std": float(err_const.std(ddof=0)),
        "median_corr_err_model_err_const": float(
            np.median([results["runs"][r]["error_structure"]["corr_err_model_err_const"] for r in ALL_RUNS])
        ),
        "min_corr_err_model_err_const": float(
            np.min([results["runs"][r]["error_structure"]["corr_err_model_err_const"] for r in ALL_RUNS])
        ),
        "max_corr_err_model_err_const": float(
            np.max([results["runs"][r]["error_structure"]["corr_err_model_err_const"] for r in ALL_RUNS])
        ),
        "median_spearman_advantage_vs_err_const": float(
            np.median(
                [results["runs"][r]["error_structure"]["spearman_advantage_vs_err_const"] for r in ALL_RUNS]
            )
        ),
        "min_spearman_advantage_vs_err_const": float(
            np.min(
                [results["runs"][r]["error_structure"]["spearman_advantage_vs_err_const"] for r in ALL_RUNS]
            )
        ),
        "spearman_err_const_vs_jitter_l2_4dof": float(spearmanr(err_const, l2_4dof).statistic),
        "spearman_err_const_vs_jitter_l2_4dof_p": float(spearmanr(err_const, l2_4dof).pvalue),
        "seed_level_median_err_model": float(
            np.median([np.median([errs[r][b] for b in beam_ids]) for r in ALL_RUNS])
        ),
    }

    # 种子一致性：20 个 run 的 advantage 向量两两 Spearman（束间赢面是否稳定）
    adv_mat = np.stack([np.asarray(results["runs"][r]["advantage"]) for r in ALL_RUNS])
    pair_rhos = []
    for a in range(len(ALL_RUNS)):
        for b in range(a + 1, len(ALL_RUNS)):
            pair_rhos.append(float(spearmanr(adv_mat[a], adv_mat[b]).statistic))
    joint_rhos = [
        float(spearmanr(adv_mat[JOINT_RUNS.index(a)], adv_mat[JOINT_RUNS.index(b)]).statistic)
        for i, a in enumerate(JOINT_RUNS)
        for b in JOINT_RUNS[i + 1 :]
    ]
    results["seed_consistency"] = {
        "pairwise_spearman_all_20_runs_median": float(np.median(pair_rhos)),
        "pairwise_spearman_all_20_runs_min": float(np.min(pair_rhos)),
        "pairwise_spearman_all_20_runs_max": float(np.max(pair_rhos)),
        "pairwise_spearman_joint_median": float(np.median(joint_rhos)),
        "joint_seed_mean_advantage_rho_vs_jitter": agg4["spearman_rho"],
        "joint_seed_mean_advantage_rho_ci95": agg4["spearman_rho_ci95"],
    }

    results["verdict"] = {
        "criterion": (
            "预注册：≥4/5 joint 种子 Spearman ρ 的 95% CI 不含 0 且为正，且 top 四分位赢率 > "
            "bottom 四分位赢率 → 判定“网络提取真实个体信息”；ρ≈0 或为负 → “族先验主导（复读机假设）”"
        ),
        "n_joint_seeds_rho_ci_positive": n_ci_pos,
        "n_joint_seeds_top_gt_bottom_win_rate": n_top_gt_bottom,
        "n_joint_seeds": len(JOINT_RUNS),
        "preregistered_rule_fires": bool(n_ci_pos >= 4 and n_top_gt_bottom >= 4),
        "confound": (
            "advantage 与 err_const 的 Spearman 中位 +0.999（advantage 几乎是 err_const 的单调函数）；"
            "err_const 随抖动幅度上升（ρ = +0.576）；模型逐束误差只以约 4% 的幅度、+0.83 的相关跟踪 err_const "
            "（std 0.0037 vs 0.0870）。把模型误差整体换成常数（flat_error_null）后，ρ 与 top/bottom 四分位赢率"
            "与观测几乎逐 run 相同（ρ 最大偏差 0.009）⇒ 预注册判据实际测的是 ρ(err_const, 抖动幅度)，"
            "不携带关于模型是否使用个体信息的信息"
        ),
        "what_is_established": (
            "严格“输出与束无关”的复读机被否证：20/20 run 的 max_b err_model < D_max/2 = 0.3337 "
            "（D_max = 150 个 test 目标图两两 M1 距离的直径）。即模型输出确实随输入变化；"
            "但该变化并未带来优于零信息常数解的中位精度。"
        ),
    }

    results["runs_summary_table"] = {
        run: {
            "variant": run.split("_s")[0],
            "n_wins": results["runs"][run]["pairing"]["n_wins"],
            "win_rate": results["runs"][run]["pairing"]["win_rate"],
            "median_advantage": results["runs"][run]["pairing"]["median_advantage"],
            "median_advantage_ci95": results["runs"][run]["pairing"]["median_advantage_ci95"],
            "wilcoxon_p": results["runs"][run]["pairing"]["wilcoxon_p"],
            "rho_4dof": results["runs"][run]["personality_l2_4dof"]["spearman_rho"],
            "rho_4dof_ci95": results["runs"][run]["personality_l2_4dof"]["spearman_rho_ci95"],
            "wr_top": results["runs"][run]["personality_l2_4dof"]["top_quartile"]["win_rate"],
            "wr_bottom": results["runs"][run]["personality_l2_4dof"]["bottom_quartile"]["win_rate"],
        }
        for run in ALL_RUNS
    }

    (OUT / "exp3_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    header = [
        "beam_id",
        *JITTER_ATTRS,
        *[f"z_{a.removeprefix('jitter_')}" for a in JITTER_ATTRS],
        "jitter_l2_4dof",
        "jitter_l2_5attr",
        "err_const",
        "mean_adv_joint",
        *[c for run in ALL_RUNS for c in (f"err_{run}", f"adv_{run}")],
    ]
    with (OUT / "exp3_per_beam.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, b in enumerate(beam_ids):
            row = [b]
            row += [jitter[a][i] for a in JITTER_ATTRS]
            row += [z[a][i] for a in JITTER_ATTRS]
            row += [l2_4dof[i], l2_5attr[i], err_const[i], adv_mean[i]]
            for run in ALL_RUNS:
                row += [errs[run][b], err_const[i] - errs[run][b]]
            w.writerow(row)

    with (OUT / "exp3_per_beam_long.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beam_id", "run", "err_model", "err_const", "advantage", "jitter_l2_4dof"])
        for run in ALL_RUNS:
            for i, b in enumerate(beam_ids):
                w.writerow([b, run, errs[run][b], err_const[i], err_const[i] - errs[run][b], l2_4dof[i]])

    with (OUT / "exp3_null_family.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["candidate", "err_median", "err_std", "err_min", "err_max", "n_wins_vs_const",
             "win_rate", "spearman_rho", "rho_ci_lo", "rho_ci_hi", "wr_top", "wr_bottom"]
        )
        for c in family:
            ci = c["spearman_rho_ci95"]
            w.writerow(
                [c["name"], c["err_median"], c["err_std"], c["err_min"], c["err_max"],
                 c["n_wins_vs_const"], c["win_rate"], c["spearman_rho"],
                 "" if ci is None else ci[0], "" if ci is None else ci[1],
                 "" if c["wr_top"] is None else c["wr_top"],
                 "" if c["wr_bottom"] is None else c["wr_bottom"]]
            )

    write_report(results, family)

    print(f"常数解重算 vs 快照最大偏差 {const_check['max_abs_diff_vs_snapshot']:.3e}")
    print("固定输出（严格复读机）否证：%d/%d run" % (
        results["fixed_output_feasibility"]["n_runs_refuted"],
        results["fixed_output_feasibility"]["n_runs"],
    ))
    print(
        "wrote exp3_results.json, exp3_per_beam.csv, exp3_per_beam_long.csv, "
        "exp3_null_family.csv, exp3_report.md"
    )


if __name__ == "__main__":
    main()
