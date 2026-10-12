"""T0 前置③：last.ckpt 上的逐束个体信息检验（增量投影 + 输入打乱，纯推理，零训练）。

判定问题：last.ckpt 相对零信息常数解（族均值图 = prior）的聚合优势（T0 实验1：20/20 反超）
来自 (a) 模型逐束提取了个体信息，还是 (b) 模型只是输出一张"更好的平均图"。

读数（全部在 M1 同款归一化空间：逐投影 x̃ = x·(Σr/Σx)，r = 该束 target_full）：

1. 增量对齐 cos(d_out, d_tgt)，d_out = out − prior、d_tgt = r − prior。逐 (束, 子样本) 计算，
   束级取子样本中位数，run 级取 150 束中位数 + 10k 束级 bootstrap 95% CI。
   主口径 = **六投影展平**（98 304 维）；附口径 = 逐投影余弦的六投影均值。
2. 增量解释率 (‖d_tgt‖² − ‖d_tgt − d_out‖²)/‖d_tgt‖²。
3. 输入打乱：束 b 的输入对（六图 + 由输入派生的 κ）换成束 π(b) 的，π = derangement
   （以 SeedSequence(20260825) 的固定子流拒绝采样）。读数 (a) 相对位移
   ‖out_scr − out_true‖/‖out_true − prior‖；(b) cos(d_scr, d_tgt_b)。
   cos(d_scr, d_tgt_π(b)) 与 π(b) 的 cos(d_out, d_tgt) 恒等（逐投影参照总计数相同），
   作为"对齐随输入走"的自洽核对列。

零参照：常数解就是 prior 本身 ⇒ d_out ≡ 0 ⇒ cos 为 0/0 无定义。
第二参照：输入 = 均值图（prior×n_low，用该束自己的 bin_edges）时的 cos(d_mean, d_tgt_b)。

推理：按束批量前向（20 个子样本一批），变体分支与 `src.line2.experiments.evaluate.predict_group`
逐分支一致（joint 用 κ、independent 用 edges + log₁₀N）；**批次改变会引入 ~1e-8 的 float32
核差异**，故逐束 err 与 exp1 的官方 batch=1 评估做数值对照（不要求逐位一致，见 xcheck_exp1）。
数据索引与 `src.line2.training.data.Line2Pairs` 一致（test 切分、站 0、n_low=1e4、subs 0-19），
bin_edges 按 evaluate.py 走 float64。

每个 run 跑完立即落盘（`pre3_run_<tag>.json` + 该 run 的逐束/逐子样本 CSV），再进下一个 run；
`--assemble-only` 可只从这些 per-run JSON 重组总汇总（无 GPU）。

只读 `studies/line2_particle_sr/{data,results}` 与 `src/`；只写本目录。

运行：
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
    python studies/line2_particle_sr/reports/t0_diagnostics/pre3_individual_info/run_pre3_individual_info.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.line2.experiments.evaluate import load_model  # noqa: E402
from src.line2.metrics.metrics import group_rel_rmse  # noqa: E402
from src.line2.training.data import Line2Pairs  # noqa: E402
from src.line2.training.unet import condition_vector, normalize_input_group  # noqa: E402

LINE2 = ROOT / "studies" / "line2_particle_sr"
DATA = LINE2 / "data" / "line2_ds_v1"
RESULTS = LINE2 / "results"
SNAPSHOT = LINE2 / "reports" / "tools" / "phase1_numbers.json"
EXP3_CSV = LINE2 / "reports" / "t0_diagnostics" / "exp3_constant_control" / "exp3_per_beam.csv"
EXP1_DIR = LINE2 / "reports" / "t0_diagnostics" / "exp1_early_stop"
OUT = Path(__file__).resolve().parent

SPLIT = "test"
STATION = 0
N_LOW = 10_000
SUBS = tuple(range(20))
ROOT_SEED = 20260825
N_BOOT = 10_000

#: 待测模型 = (标签, run 目录, ckpt 文件名)。前 5 个是核心（实验组 last.ckpt），第 6 个是对照。
MODELS = [
    *[(f"e1_joint_s{i}/last", f"e1_joint_s{i}", "last.ckpt") for i in range(5)],
    ("e1_joint_s0/best_val", "e1_joint_s0", "best_val.ckpt"),
]


def tag_of(label: str) -> str:
    return label.replace("/", "_")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    head = (ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        return (ROOT / ".git" / head[5:]).read_text(encoding="utf-8").strip()
    return head


def load_test_beams() -> list[int]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    return [int(b) for b in manifest["splits"][SPLIT]]


def build_prior() -> np.ndarray:
    """零信息常数解：700 个 train 束 station0/target_full 逐投影归一后的均值图 (6,128,128)。

    配方逐行照抄 `reports/tools/compute_phase1_numbers.py::zero_information_solution`；
    结果缓存为 prior_mean700.npy（存在即复用，并在 :func:`verify_prior` 中按逐束误差复核）。
    """
    cache = OUT / "prior_mean700.npy"
    if cache.is_file():
        return np.load(cache)
    acc = None
    n_beams = 0
    with h5py.File(DATA / "train.h5", "r") as fh:
        for beam in sorted(fh.keys()):
            grp = fh[beam]["station0"]["target_full"][:].astype(np.float64)
            a = grp / grp.sum(axis=(1, 2), keepdims=True)
            acc = a if acc is None else acc + a
            n_beams += 1
    prior = acc / n_beams
    np.save(cache, prior)
    return prior


def verify_prior(prior: np.ndarray, beam_ids: list[int]) -> dict:
    """prior 的逐位复核：重算 150 束 M1 常数解误差，与快照/exp3 逐元素比对。"""
    refs: dict[int, np.ndarray] = {}
    with h5py.File(DATA / f"{SPLIT}.h5", "r") as fh:
        for beam_id in beam_ids:
            refs[beam_id] = fh[f"beam_{beam_id:04d}"]["station0"]["target_full"][:]
    errs = {b: group_rel_rmse(prior, refs[b]) for b in beam_ids}

    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["zero_information"]
    snap_ids = sorted(refs)  # 快照按 HDF5 key 排序（'beam_0002' < ... 与数值序一致）
    snap_dev = float(np.max(np.abs(np.asarray([errs[b] for b in snap_ids]) - np.asarray(snap["const_errs"]))))

    exp3_ids: list[int] = []
    exp3_errs: list[float] = []
    with EXP3_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            exp3_ids.append(int(row["beam_id"]))
            exp3_errs.append(float(row["err_const"]))
    exp3_dev = float(np.max(np.abs(np.asarray([errs[b] for b in exp3_ids]) - np.asarray(exp3_errs))))

    return {
        "n_train_beams": snap["n_train_beams"],
        "const_err_median": float(np.median([errs[b] for b in beam_ids])),
        "snapshot_const_median": float(snap["const_median"]),
        "snapshot_const_errs_max_abs_dev": snap_dev,
        "exp3_err_const_max_abs_dev": exp3_dev,
        "exp3_n_rows": len(exp3_ids),
        "prior_sha256": sha256_of(OUT / "prior_mean700.npy"),
        "prior_projection_sums": [float(v) for v in prior.sum(axis=(1, 2))],
    }


def derangement(n: int, seed: int) -> np.ndarray:
    """确定性错位置换：以固定种子流拒绝采样，直至无不动点（期望 ~2.7 次抽取）。"""
    ss = np.random.SeedSequence(seed).spawn(1)[0]
    rng = np.random.default_rng(ss)
    for _ in range(1000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p.astype(np.int64)
    raise RuntimeError("derangement 构造失败（1000 次抽取仍有不动点）")


def predict_batch(
    model, cfg: dict, counts: np.ndarray, edges: np.ndarray,
    n_low: int | float, device: str,
) -> np.ndarray:
    """批量前向（counts (B,6,128,128) + 共享 edges (4,129)）→ 概率图 (B,6,128,128) float64。

    变体分支与 `evaluate.predict_group` 逐分支一致；批大小只影响 float32 卷积核选择。
    """
    variant = str(cfg.get("variant", "joint"))
    dev = torch.device(device)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.float32).to(dev)
    edges_t = (
        torch.as_tensor(np.asarray(edges), dtype=torch.float32)
        .unsqueeze(0)
        .expand(counts_t.shape[0], -1, -1)
        .to(dev)
    )
    was_training = model.training
    model.to(dev).eval()
    try:
        with torch.no_grad():
            shapes, _totals = normalize_input_group(counts_t)
            if variant == "joint":
                prob = model(shapes, condition_vector(counts_t, edges_t))
            elif variant == "independent":
                n_low_vec = counts_t.sum(dim=(1, 2, 3)).clamp_min(1.0)
                prob = model(shapes, edges_t, torch.log10(n_low_vec))
            else:
                raise ValueError(f"未知 variant：{variant!r}（应 ∈ joint/independent）")
            out = prob.detach().to(torch.float64).cpu().numpy()
    finally:
        model.train(was_training)
    if out.shape[1:] != (6, 128, 128):
        raise ValueError(f"概率图形状异常：{out.shape}（应 (B,6,128,128)）")
    return np.ascontiguousarray(out, dtype=np.float64)


def _cos_and_stats(d_out: np.ndarray, d_tgt: np.ndarray) -> tuple[float, float, float, float]:
    """返回 (展平余弦, 增量解释率, ‖d_out‖, ‖d_tgt‖)。分母为 0 时余弦/解释率为 nan。"""
    a = d_out.ravel()
    b = d_tgt.ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na > 0.0 and nb > 0.0:
        cos = float(np.dot(a, b) / (na * nb))
    else:
        cos = float("nan")
    if nb > 0.0:
        diff = b - a
        expl = float((np.dot(b, b) - np.dot(diff, diff)) / np.dot(b, b))
    else:
        expl = float("nan")
    return cos, expl, na, nb


def _cos_per_projection(d_out: np.ndarray, d_tgt: np.ndarray) -> float:
    """逐投影余弦的六投影均值（辅口径）；任一分量无定义则整体 nan。"""
    a = d_out.reshape(6, -1)
    b = d_tgt.reshape(6, -1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if np.any(na == 0.0) or np.any(nb == 0.0):
        return float("nan")
    return float(np.mean(np.sum(a * b, axis=1) / (na * nb)))


def _nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _iqr(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 25)), float(np.percentile(finite, 75))]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """束级 bootstrap 95% CI（中位数）；nan 值按有限样本重抽。"""
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return {"median": float("nan"), "ci95": [float("nan"), float("nan")], "n_finite": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(finite[idx], axis=1)
    return {
        "median": float(np.median(finite)),
        "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "n_finite": int(n),
    }


def load_data_for_beam(h5: h5py.File, beam_id: int) -> dict:
    stg = h5[f"beam_{beam_id:04d}"][f"station{STATION}"]
    return {
        "target": stg["target_full"][...].astype(np.float64),
        "counts": np.stack([stg[f"tier_{N_LOW:06d}_sub_{s:02d}"][...] for s in SUBS]),
        "edges": np.asarray(stg[f"tier_{N_LOW:06d}_sub_{SUBS[0]:02d}"].attrs["bin_edges"], dtype=np.float64),
    }


def analyze_run(
    label: str, run_id: str, ckpt_name: str, model, cfg: dict, step: int,
    h5: h5py.File, beam_ids: list[int], perm: np.ndarray, prior: np.ndarray,
    device: str, batch: int, progress: bool = True,
) -> dict:
    """单 run 全流程（逐束推理 + 指标），返回 {"beam_rows": [...], "sub_rows": [...], "entry": {...}}。"""
    beam_rows: list[dict] = []
    sub_rows: list[dict] = []
    t0_run = time.time()
    for bi, beam_id in enumerate(beam_ids):
        pair_id = beam_ids[int(perm[bi])]
        own = load_data_for_beam(h5, beam_id)
        pair = load_data_for_beam(h5, pair_id)
        r_scale = own["target"].sum(axis=(1, 2))          # (6,) 该束逐投影参照总计数
        r_scale_pair = pair["target"].sum(axis=(1, 2))
        prior_n = prior * r_scale[:, None, None]
        prior_n_pair = prior * r_scale_pair[:, None, None]
        d_tgt = own["target"] - prior_n
        d_tgt_pair = pair["target"] - prior_n_pair
        cos_true, cos_scr, cos_scr_pair, cos_pp, expl, disp = [], [], [], [], [], []
        err_true, err_scr, nd_out, nd_tgt = [], [], [], []
        for lo in range(0, len(SUBS), batch):
            sl = slice(lo, min(lo + batch, len(SUBS)))
            out_true = predict_batch(model, cfg, own["counts"][sl], own["edges"], N_LOW, device)
            out_scr = predict_batch(model, cfg, pair["counts"][sl], pair["edges"], N_LOW, device)
            for k in range(out_true.shape[0]):
                d_out = out_true[k] * r_scale[:, None, None] - prior_n
                d_scr = out_scr[k] * r_scale[:, None, None] - prior_n
                c_t, e_t, na, nb = _cos_and_stats(d_out, d_tgt)
                c_s = _cos_and_stats(d_scr, d_tgt)[0]
                c_sp = _cos_and_stats(d_scr, d_tgt_pair)[0]
                denom = float(np.linalg.norm(d_out))
                disp_s = float(np.linalg.norm(d_scr - d_out) / denom) if denom > 0 else float("nan")
                c_pp = _cos_per_projection(d_out, d_tgt)
                err_t = group_rel_rmse(out_true[k], own["target"])
                err_s = group_rel_rmse(out_scr[k], own["target"])
                cos_true.append(c_t); cos_scr.append(c_s); cos_scr_pair.append(c_sp)
                cos_pp.append(c_pp); expl.append(e_t); disp.append(disp_s)
                err_true.append(err_t); err_scr.append(err_s); nd_out.append(na); nd_tgt.append(nb)
                sub_rows.append({
                    "run": label, "beam_id": beam_id, "sub_idx": SUBS[lo + k],
                    "cos_true": c_t, "cos_scr": c_s, "cos_scr_pair": c_sp, "cos_true_pp": c_pp,
                    "expl_frac": e_t, "disp_ratio": disp_s, "norm_d_out": na, "norm_d_tgt": nb,
                    "err_true": err_t, "err_scr": err_s,
                })
        # 第二参照：输入 = 均值图（prior×n_low，用该束自己的 bin_edges）
        out_mean = predict_batch(model, cfg, (prior * float(N_LOW))[None], own["edges"], N_LOW, device)[0]
        d_mean = out_mean * r_scale[:, None, None] - prior_n
        cos_mean = _cos_and_stats(d_mean, d_tgt)[0]
        beam_rows.append({
            "run": label, "beam_id": beam_id, "pair_beam_id": pair_id,
            "cos_true": _nanmedian(np.asarray(cos_true)),
            "cos_scr": _nanmedian(np.asarray(cos_scr)),
            "cos_scr_pair": _nanmedian(np.asarray(cos_scr_pair)),
            "cos_true_pp": _nanmedian(np.asarray(cos_pp)),
            "expl_frac": _nanmedian(np.asarray(expl)),
            "disp_ratio": _nanmedian(np.asarray(disp)),
            "norm_d_out": _nanmedian(np.asarray(nd_out)),
            "norm_d_tgt": _nanmedian(np.asarray(nd_tgt)),
            "err_true": _nanmedian(np.asarray(err_true)),
            "err_scr": _nanmedian(np.asarray(err_scr)),
            "cos_mean_input": cos_mean,
        })
        if progress and ((bi + 1) % 25 == 0 or bi + 1 == len(beam_ids)):
            print(f"  [{label}] beam {bi + 1}/{len(beam_ids)} {time.time() - t0_run:.0f}s", flush=True)

    arr = lambda key: np.asarray([float(r[key]) for r in beam_rows])  # noqa: E731
    rng = np.random.default_rng(np.random.SeedSequence(ROOT_SEED).spawn(1)[0])
    cos_t = bootstrap_ci(arr("cos_true"), rng)
    cos_s = bootstrap_ci(arr("cos_scr"), rng)
    entry = {
        "ckpt": str((RESULTS / run_id / ckpt_name).relative_to(ROOT)),
        "ckpt_sha256": sha256_of(RESULTS / run_id / ckpt_name),
        "step": step, "variant": str(cfg.get("variant")), "seed": int(cfg["seed"]),
        "n_beams": len(beam_rows), "n_subs": len(SUBS), "batch": batch,
        "cos_true": cos_t,
        "cos_true_pp": bootstrap_ci(arr("cos_true_pp"), rng),
        "cos_scr": cos_s,
        "cos_scr_pair": bootstrap_ci(arr("cos_scr_pair"), rng),
        "cos_mean_input": bootstrap_ci(arr("cos_mean_input"), rng),
        "expl_frac": bootstrap_ci(arr("expl_frac"), rng),
        "disp_ratio": bootstrap_ci(arr("disp_ratio"), rng),
        "err_true_median_of_beam_median": float(np.median(arr("err_true"))),
        "err_scr_median": float(np.median(arr("err_scr"))),
        "norm_d_out_median": float(np.median(arr("norm_d_out"))),
        "norm_d_tgt_median": float(np.median(arr("norm_d_tgt"))),
        "cos_true_iqr": _iqr(arr("cos_true")),
        "n_beam_nan_cos_true": int(np.sum(~np.isfinite(arr("cos_true")))),
        "n_beam_nan_cos_scr": int(np.sum(~np.isfinite(arr("cos_scr")))),
        "drop_frac_median_cos": (
            float((cos_t["median"] - cos_s["median"]) / cos_t["median"])
            if np.isfinite(cos_t["median"]) and cos_t["median"] != 0 else float("nan")
        ),
        "cos_true_minus_cos_scr_mean": float(_nanmean(arr("cos_true")) - _nanmean(arr("cos_scr"))),
        "n_beams_cos_true_gt_cos_scr": int(np.sum(arr("cos_true") > arr("cos_scr"))),
        "infer_s": round(time.time() - t0_run, 1),
    }
    # exp1 的官方 batch=1 评估对照（同口径；批次差异 ⇒ float32 核噪声级别，非逐位）
    suffix = "last_eval.json" if ckpt_name == "last.ckpt" else "bestval_eval.json"
    xref = EXP1_DIR / f"{run_id}_{suffix}"
    if xref.is_file() and len(beam_ids) == 150:
        payload = json.loads(xref.read_text(encoding="utf-8"))
        exp_ids = [int(b["beam_id"]) for b in payload["per_beam"]]
        exp_err = {int(b["beam_id"]): float(b["err"]) for b in payload["per_beam"]}
        assert exp_ids == beam_ids, f"{xref} 束序与 manifest 不一致"
        entry["xcheck_exp1"] = {
            "path": str(xref.relative_to(ROOT)),
            "max_abs_dev": float(np.max(np.abs(arr("err_true") - np.asarray([exp_err[b] for b in beam_ids])))),
            "exp1_err_median": float(payload["err_median"]),
            "exp1_batch": 1,
        }
    return {"beam_rows": beam_rows, "sub_rows": sub_rows, "entry": entry}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def judge(runs: dict, labels_last: list[str]) -> dict:
    """预注册判定（规则见交付说明）：≥4/5 run 的 cos_true 中位 CI 排除 0 且为正，
    且打乱使中位 cos 下降 ≥80% → 个体信息成立；CI 含 0 → 仅更好先验；
    打乱相对位移 <10% → 输入不敏感警报。"""
    detail = {}
    n_ci_pos = 0
    for label in labels_last:
        e = runs[label]
        ci_ok = bool(np.isfinite(e["cos_true"]["ci95"][0]) and e["cos_true"]["ci95"][0] > 0
                     and e["cos_true"]["ci95"][1] > 0)
        drop_ok = bool(np.isfinite(e["drop_frac_median_cos"]) and e["drop_frac_median_cos"] >= 0.8)
        disp = e["disp_ratio"]["median"]
        sensitive = bool(np.isfinite(disp) and disp >= 0.10)
        n_ci_pos += int(ci_ok)
        detail[label] = {
            "cos_true_median": e["cos_true"]["median"], "cos_true_ci95": e["cos_true"]["ci95"],
            "cos_scr_median": e["cos_scr"]["median"], "drop_frac": e["drop_frac_median_cos"],
            "disp_ratio_median": disp,
            "cos_true_ci_excludes_0_positive": ci_ok,
            "shuffle_drop_ge_80pct": drop_ok,
            "dispatch_sensitive_ge_10pct": sensitive,
        }
    all_drop = all(detail[l]["shuffle_drop_ge_80pct"] for l in labels_last)
    all_sens = all(detail[l]["dispatch_sensitive_ge_10pct"] for l in labels_last)
    individual = bool(n_ci_pos >= 4 and all_drop)
    prior_only = bool(all(detail[l]["cos_true_ci95"][0] <= 0 for l in labels_last))
    return {
        "n_runs": len(labels_last), "n_cos_ci_positive": n_ci_pos,
        "rule": "≥4/5 run 中位 cos 的 95% CI 排除 0 且为正，且打乱使中位 cos 下降 ≥80%",
        "verdict_individual_info": individual,
        "verdict_prior_only": prior_only,
        "verdict_input_insensitive_alarm": bool(not all_sens),
        "per_run": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T0 前置③：逐束个体信息检验（增量投影 + 输入打乱）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="", help="逗号分隔的标签子集（默认全部 6 个）")
    parser.add_argument("--limit-beams", type=int, default=0, help="只跑前 N 个束（冒烟用，0 = 全 150）")
    parser.add_argument("--batch", type=int, default=20, help="每束批量前向的子样本数（1 = 官方逐样本口径）")
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--assemble-only", action="store_true", help="只从 per-run JSON 重组总汇总（无 GPU）")
    args = parser.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True

    models = [m for m in MODELS if (not args.models or m[0] in args.models.split(","))]
    beam_ids_all = load_test_beams()
    beam_ids = beam_ids_all[: args.limit_beams] if args.limit_beams else beam_ids_all
    perm = derangement(len(beam_ids), ROOT_SEED)
    t_start = time.time()

    prior = build_prior()
    prior_meta = verify_prior(prior, beam_ids_all) if not args.limit_beams else {}
    print(f"[prior] n_train={prior_meta.get('n_train_beams')} "
          f"const_err_median={prior_meta.get('const_err_median')} "
          f"snapshot_dev={prior_meta.get('snapshot_const_errs_max_abs_dev')} "
          f"exp3_dev={prior_meta.get('exp3_err_const_max_abs_dev')}", flush=True)

    all_beam_rows: list[dict] = []
    all_sub_rows: list[dict] = []
    runs: dict = {}
    for label, run_id, ckpt_name in models:
        tag = tag_of(label)
        run_json = outdir / f"pre3_run_{tag}.json"
        beam_csv = outdir / f"pre3_run_{tag}_beams.csv"
        sub_csv = outdir / f"pre3_run_{tag}_subs.csv"
        if args.assemble_only or run_json.is_file():
            print(f"[skip] {label}：已有 {run_json.name}（复用）", flush=True)
            entry = json.loads(run_json.read_text(encoding="utf-8"))
            runs[label] = entry
            with beam_csv.open(encoding="utf-8") as fh:
                all_beam_rows.extend(list(csv.DictReader(fh)))
            with sub_csv.open(encoding="utf-8") as fh:
                all_sub_rows.extend(list(csv.DictReader(fh)))
            continue

        ckpt = RESULTS / run_id / ckpt_name
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model, cfg = load_model(ckpt)
        print(f"[run ] {label} step={int(payload.get('step', -1))} ckpt={ckpt.name}", flush=True)
        with h5py.File(DATA / f"{SPLIT}.h5", "r") as h5:
            res = analyze_run(label, run_id, ckpt_name, model, cfg, int(payload.get("step", -1)),
                              h5, beam_ids, perm, prior, args.device, args.batch)
        del model
        torch.cuda.empty_cache()
        # 增量落盘：本 run 全部产物先于下一个 run 落盘
        write_csv(beam_csv, res["beam_rows"])
        write_csv(sub_csv, res["sub_rows"])
        run_json.write_text(json.dumps(res["entry"], ensure_ascii=False, indent=2), encoding="utf-8")
        runs[label] = res["entry"]
        all_beam_rows.extend(res["beam_rows"])
        all_sub_rows.extend(res["sub_rows"])
        print(f"[done] {label} infer_s={res['entry']['infer_s']} "
              f"cos_true_med={res['entry']['cos_true']['median']:.4f} "
              f"ci={[round(v, 4) for v in res['entry']['cos_true']['ci95']]} "
              f"cos_scr_med={res['entry']['cos_scr']['median']:.4f} "
              f"xcheck={res['entry'].get('xcheck_exp1', {}).get('max_abs_dev')}", flush=True)

    labels_last = [m[0] for m in models if m[2] == "last.ckpt" and m[0] in runs]
    summary = {
        "meta": {
            "code_version": git_head(),
            "data_version": json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["data_version"],
            "spec_version": "line2-spec-v1.0",
            "split": SPLIT, "station": STATION, "n_low": N_LOW, "subs": list(SUBS),
            "n_beams": len(beam_ids), "beam_ids": beam_ids,
            "derangement_seed": ROOT_SEED,
            "derangement_pair": {str(b): int(beam_ids[int(perm[i])]) for i, b in enumerate(beam_ids)},
            "bootstrap": {"n_boot": N_BOOT, "root_seed": ROOT_SEED, "unit": "beam",
                          "quantile": [2.5, 97.5]},
            "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(),
            "exec_mode": {"deterministic_algorithms": True, "warn_only": True,
                          "cudnn_deterministic": True,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
            "argv": sys.argv,
            "batch_note": "批量前向（batch=%d）与官方 batch=1 存在 ~1e-8 级 float32 核差异" % args.batch,
            "elapsed_s": round(time.time() - t_start, 1),
            "infer_s_per_run": {k: v["infer_s"] for k, v in runs.items()},
            "products": {},
        },
        "prior": prior_meta,
        "runs": runs,
        "judgment": judge(runs, labels_last) if labels_last else {},
    }
    all_beam_rows.sort(key=lambda r: (r["run"], int(r["beam_id"])))
    all_sub_rows.sort(key=lambda r: (r["run"], int(r["beam_id"]), int(r["sub_idx"])))
    beam_path = outdir / "pre3_per_beam.csv"
    sub_path = outdir / "pre3_per_beam_sub.csv"
    write_csv(beam_path, all_beam_rows)
    write_csv(sub_path, all_sub_rows)
    summary["meta"]["products"] = {
        "per_beam_csv": str(beam_path.relative_to(ROOT)),
        "per_beam_sub_csv": str(sub_path.relative_to(ROOT)),
        "per_run_json_glob": str((outdir / "pre3_run_*.json").relative_to(ROOT)),
        "per_run_beam_csv_glob": str((outdir / "pre3_run_*_beams.csv").relative_to(ROOT)),
    }
    out_path = outdir / "pre3_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["judgment"], ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
