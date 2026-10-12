"""T0-实验2：物理先验条件化 κ 的推理期消融（诊断脚本；对 results/ data/ src/ 只读）。

判决性问题：E1 联合版（variant=joint）的 FiLM 条件向量 κ（11 维 = 输入图组 10 维
Σ 摘要 + log₁₀N_low）在推理期被置零 / 换成常数 / 跨束置换时，test 端指标是否变化？
不变化 ⇒ κ 是旁路（FiLM 退化为常偏置）。

κ 注入路径（与生产代码逐行同构，本脚本**不改模型内部结构**）：

    src/line2/training/unet.py:32   condition_vector(counts, edges)  → κ (B, 11)
    src/line2/experiments/evaluate.py:predict_group                  → κ → model(shapes, κ)
    src/models/unet.py:FilmLayer                                     → γ(z)·F + β(z)

消融在"κ 张量已算出、尚未交给模型"这一步替换 κ：本脚本把
``evaluate.condition_vector``（模块级名字）临时换成注入器，随后调用**未修改**的
``evaluate.evaluate_run``——前向路径、M1/M2 指标、观测向量、N_eff 与聚合次序
全部走生产代码。评估结束后立即还原。

四个变体（预注册）：
    baseline  原样 κ（同时逐样本核对 counts 校验和，验证评估顺序与内容与预扫描一致）
    zero      κ → 全零向量
    mean      κ → 测试集全体（束, 子样本）样本的均值 κ（常数替换）
    permuted  κ → 跨束置换：束位置 i 取束位置 π(i) 的同 sub 样本的 κ
              （π 由 SeedSequence(20260825) 派生，拒绝含不动点）

另含两个机制级旁证（非判决变体）：输出级探针（子集样本上 κ 干预前后概率图差异）
与 FiLM 灵敏度（各 FiLM 层调制量在测试集 κ 上的逐样本标准差 vs 其均值量级）。

用法（仓库根目录）：
    python studies/line2_particle_sr/reports/t0_diagnostics/exp2_kappa_ablation/exp2_kappa_ablation.py scan
    python studies/.../exp2_kappa_ablation.py eval --run e1_joint_s0 --variant baseline
    python studies/.../exp2_kappa_ablation.py drive --workers 3
    python studies/.../exp2_kappa_ablation.py analyze
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.line2.experiments import evaluate as ev  # noqa: E402
from src.line2.training.unet import condition_vector  # noqa: E402
from src.models.unet import FilmLayer  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "studies/line2_particle_sr/results"
DATA = ROOT / "studies/line2_particle_sr/data/line2_ds_v1"
CURVE = RESULTS / "e0_curve.json"
VARIANTS_DIR = HERE / "variants"
TABLE_NPZ = HERE / "kappa_table.npz"
TABLE_META = HERE / "kappa_table_meta.json"
MASTER_SEED = 20260825

RUNS = [f"e1_joint_s{i}" for i in range(5)]
VARIANTS = ("baseline", "zero", "mean", "permuted")
#: 机制对照臂（非判决）：只把前 10 维（Σ 矩）置零、保留第 11 维（log₁₀N）——
#: 若其输出与 baseline 逐位相同，则证明 Σ 维在 float32 中数值惰性，κ 的全部
#: 有效内容只是第 11 维那个 run 内恒定的常数。
CONTROL_VARIANTS = ("sigma_zero",)
ALL_VARIANTS = VARIANTS + CONTROL_VARIANTS
SPLIT = "test"
STATION = 0
N_LOW = 10000
SUBS = "0-19"

EVAL_ENV_DEVICE = "cuda:0"          # CUDA_VISIBLE_DEVICES 由 drive 钉到单张卡


# --------------------------------------------------------------------------- #
# 预扫描：按 evaluate_run 的迭代次序算出每样本 κ（供 mean / permuted 变体取用）
# --------------------------------------------------------------------------- #

def _tensors(counts, edges) -> tuple[torch.Tensor, torch.Tensor]:
    """与 ``evaluate.predict_group`` 完全同构的张量构造（float32 + 补 batch 维）。"""
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.float32).unsqueeze(0)
    edges_t = torch.as_tensor(np.asarray(edges), dtype=torch.float32).unsqueeze(0)
    return counts_t, edges_t


def _beam_ids(split: str = SPLIT) -> list[int]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    return [int(b) for b in manifest["splits"][split]]


def scan() -> dict:
    """测试集 κ 全表：逐样本 (beam, sub) → κ、counts 校验和、总计数。"""
    beam_ids = _beam_ids()
    sub_list = ev.parse_subs(SUBS)
    kappas: list[np.ndarray] = []
    shas: list[str] = []
    beam_col: list[int] = []
    sub_col: list[int] = []
    sums: list[float] = []
    t0 = time.time()
    with h5py.File(DATA / f"{SPLIT}.h5", "r") as h5:
        for beam_id in beam_ids:
            stg = h5[f"beam_{beam_id:04d}"][f"station{STATION}"]
            for sub_idx in sub_list:
                dset = stg[f"tier_{N_LOW:06d}_sub_{sub_idx:02d}"]
                counts = dset[...]
                edges = np.asarray(dset.attrs["bin_edges"], dtype=np.float64)
                counts_t, edges_t = _tensors(counts, edges)
                with torch.no_grad():
                    kappa = condition_vector(counts_t, edges_t)[0].to(torch.float64).numpy()
                kappas.append(kappa)
                shas.append(hashlib.sha1(counts_t.numpy().tobytes()).hexdigest())
                beam_col.append(beam_id)
                sub_col.append(sub_idx)
                sums.append(float(counts_t.sum()))
    kap = np.stack(kappas)  # (N, 11)
    np.savez(
        TABLE_NPZ,
        kappa=kap,
        beam_id=np.asarray(beam_col, dtype=np.int64),
        sub_idx=np.asarray(sub_col, dtype=np.int64),
        counts_sum=np.asarray(sums, dtype=np.float64),
        counts_sha1=np.asarray(shas, dtype="<U40"),
    )
    n_beams, n_subs = len(beam_ids), len(sub_list)
    stats = {
        "n_samples": int(kap.shape[0]),
        "n_beams": n_beams,
        "n_subs": n_subs,
        "beam_ids": beam_ids,
        "subs": sub_list,
        "elapsed_s": round(time.time() - t0, 1),
        "kappa_mean": kap.mean(axis=0).tolist(),
        "kappa_std": kap.std(axis=0).tolist(),
        "kappa_min": kap.min(axis=0).tolist(),
        "kappa_max": kap.max(axis=0).tolist(),
        # 逐维"逐样本变动 / 量级"比：≈0 表示该维在测试集上几乎是常数
        "kappa_rel_std": (kap.std(axis=0) / np.maximum(np.abs(kap.mean(axis=0)), 1e-300)).tolist(),
        "counts_sum_min": float(np.min(sums)),
        "counts_sum_max": float(np.max(sums)),
        "counts_sum_rel_std": float(np.std(sums) / np.mean(sums)),
        "n_duplicate_counts_sha1": int(len(shas) - len(set(shas))),
    }
    TABLE_META.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scan] {kap.shape} → {TABLE_NPZ.name}（{stats['elapsed_s']}s）", flush=True)
    return stats


def load_table() -> dict:
    npz = np.load(TABLE_NPZ, allow_pickle=False)
    meta = json.loads(TABLE_META.read_text(encoding="utf-8"))
    return {
        "kappa": npz["kappa"],
        "beam_id": npz["beam_id"],
        "sub_idx": npz["sub_idx"],
        "counts_sum": npz["counts_sum"],
        "counts_sha1": [str(s) for s in npz["counts_sha1"]],
        "meta": meta,
    }


def _derive_ss(*labels) -> np.random.SeedSequence:
    """由主种子与整数/字符串标签派生 SeedSequence（标签取 UTF-8 字节，确定性）。"""
    entropy: list[int] = [MASTER_SEED]
    for lab in labels:
        if isinstance(lab, int):
            entropy.append(int(lab))
        else:
            entropy.extend(
                int(b) for b in str(lab).encode("utf-8")
            )
    return np.random.SeedSequence(entropy)


def permutation_index(table: dict) -> np.ndarray:
    """跨束置换索引：位置 i（束位置）→ 源束位置；无不动点（重抽至多 50 次）。"""
    n_beams = table["meta"]["n_beams"]
    if n_beams < 2:
        raise ValueError("束数 < 2，无法构造跨束置换")
    ss = _derive_ss("exp2", "kappa_crossbeam_permutation")
    rng = np.random.default_rng(ss)
    for _ in range(50):
        perm = rng.permutation(n_beams)
        if not np.any(perm == np.arange(n_beams)):
            return perm
    raise RuntimeError("50 次抽签仍未得到无不动点的束置换")


# --------------------------------------------------------------------------- #
# κ 注入器（替换 evaluate 模块级 condition_vector，逐样本核对评估顺序）
# --------------------------------------------------------------------------- #

class KappaInjector:
    """``condition_vector(counts, edges)`` 的替身：算出原 κ 后按变体改写。"""

    def __init__(self, mode: str, table: dict, device: torch.device) -> None:
        self.mode = mode
        self.table = table
        self.device = device
        self.calls = 0
        self.max_abs_delta: list[float] = []   # 逐样本 |κ_variant − κ_base|∞
        self.n_nonzero_delta = 0
        kap = table["kappa"]
        if mode == "mean":
            mean_vec = torch.as_tensor(kap.mean(axis=0), dtype=torch.float32, device=device)
            self.mean_vec = mean_vec.unsqueeze(0)
        if mode == "permuted":
            perm = permutation_index(table)
            self.beam_pos = {int(b): i for i, b in enumerate(table["meta"]["beam_ids"])}
            self.perm = perm
            # 逐样本：调用序号 → （来源样本序号）；同 sub 对齐（源束位置 perm[束位置]）
            self.src_index = np.empty(len(kap), dtype=np.int64)
            for i in range(len(kap)):
                beam_pos = self.beam_pos[int(table["beam_id"][i])]
                sub_idx = int(table["sub_idx"][i])
                src_beam = int(table["meta"]["beam_ids"][int(perm[beam_pos])])
                src = np.where(
                    (table["beam_id"] == src_beam) & (table["sub_idx"] == sub_idx)
                )[0]
                if src.size != 1:
                    raise RuntimeError(f"置换源样本定位失败：beam={src_beam} sub={sub_idx}")
                self.src_index[i] = int(src[0])
            self.kappa_t = torch.as_tensor(
                kap, dtype=torch.float32, device=device
            )

    def __call__(self, counts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        base = condition_vector(counts, edges)
        i = self.calls
        if i >= len(self.table["kappa"]):
            raise RuntimeError(f"κ 注入器调用次数超出预扫描表（{i}）：评估遍历次序与预扫描不一致")
        sha = hashlib.sha1(counts.detach().cpu().numpy().tobytes()).hexdigest()
        if sha != self.table["counts_sha1"][i]:
            raise RuntimeError(
                f"第 {i} 次调用与预扫描样本不匹配（beam={self.table['beam_id'][i]} "
                f"sub={self.table['sub_idx'][i]}）：评估遍历次序改变，变体映射不可信"
            )
        if self.mode == "baseline":
            out = base
        elif self.mode == "zero":
            out = torch.zeros_like(base)
        elif self.mode == "sigma_zero":
            out = base.clone()
            out[:, :10] = 0.0
        elif self.mode == "mean":
            out = self.mean_vec.to(dtype=base.dtype, device=base.device).expand_as(base)
        elif self.mode == "permuted":
            out = self.kappa_t[self.src_index[i]].unsqueeze(0).to(
                dtype=base.dtype, device=base.device
            )
        else:
            raise ValueError(f"未知变体：{self.mode!r}")
        delta = float((out.to(torch.float64) - base.to(torch.float64)).abs().max())
        self.max_abs_delta.append(delta)
        if delta > 0:
            self.n_nonzero_delta += 1
        self.calls += 1
        return out


@contextlib.contextmanager
def injected(injector: KappaInjector):
    original = ev.condition_vector
    ev.condition_vector = injector
    try:
        yield injector
    finally:
        ev.condition_vector = original


def run_variant(run_id: str, variant: str, device: str, probe: bool = False) -> dict:
    """单 (run, 变体) 评估：κ 替换 + 生产评估管线（evaluate_run）。"""
    table = load_table()
    ckpt = RESULTS / run_id / "best_val.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    out = VARIANTS_DIR / f"{run_id}__{variant}.json"
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    injector = KappaInjector(variant, table, dev)
    t0 = time.time()
    with injected(injector):
        payload = ev.evaluate_run(
            ckpt=ckpt, data_dir=DATA, split=SPLIT, station=STATION, n_low=N_LOW,
            subs=SUBS, curve=CURVE, device=device, out=out,
        )
    info = {
        "run_id": run_id,
        "variant": variant,
        "n_calls": injector.calls,
        "n_kappa_delta_nonzero": injector.n_nonzero_delta,
        "kappa_max_abs_delta_overall": max(injector.max_abs_delta, default=0.0),
        "kappa_max_abs_delta_median": float(np.median(injector.max_abs_delta)),
        "err_median": payload["err_median"],
        "n_eff_flag": payload["n_eff_flag"],
        "elapsed_s": round(time.time() - t0, 1),
        "out": str(out),
        "device": device,
    }
    if probe:
        info["probe"] = probe_output(run_id, table, device)
    (VARIANTS_DIR / f"{run_id}__{variant}.runinfo.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[eval] {run_id} {variant:9s} err_median={payload['err_median']:.6f} "
        f"Δκmax={info['kappa_max_abs_delta_overall']:.3e} calls={injector.calls} "
        f"({info['elapsed_s']}s)",
        flush=True,
    )
    return info


# --------------------------------------------------------------------------- #
# 机制级旁证（非判决变体）：输出级探针
# --------------------------------------------------------------------------- #

PROBE_N = 30
PROBE_CTRL_SCALE = 1000.0


def probe_output(run_id: str, table: dict, device: str) -> dict:
    """子集样本上的输出级差异：κ 干预是否真的改变了概率图。

    对前 ``PROBE_N`` 个样本，分别在 baseline / zero / mean / permuted 四种 κ 下
    走 ``evaluate.predict_group``（生产前向），比较概率图的相对 L1 差与 M1。
    附阳性对照 scale1000：κ×1000（远超数据范围），检验 FiLM 通路是否活着。
    """
    from src.line2.metrics.metrics import group_rel_rmse

    ckpt = RESULTS / run_id / "best_val.ckpt"
    model, cfg = ev.load_model(ckpt)
    idx = list(range(min(PROBE_N, len(table["kappa"]))))
    perm = permutation_index(table)
    beam_pos = {int(b): i for i, b in enumerate(table["meta"]["beam_ids"])}
    rows = []
    with h5py.File(DATA / f"{SPLIT}.h5", "r") as h5:
        for i in idx:
            beam_id = int(table["beam_id"][i])
            sub_idx = int(table["sub_idx"][i])
            stg = h5[f"beam_{beam_id:04d}"][f"station{STATION}"]
            dset = stg[f"tier_{N_LOW:06d}_sub_{sub_idx:02d}"]
            counts = dset[...]
            edges = np.asarray(dset.attrs["bin_edges"], dtype=np.float64)
            ref = stg["target_full"][...].astype(np.float64)
            src_i = int(
                np.where(
                    (table["beam_id"] == int(table["meta"]["beam_ids"][int(perm[beam_pos[beam_id]])]))
                    & (table["sub_idx"] == sub_idx)
                )[0][0]
            )
            kappa_base = torch.as_tensor(
                table["kappa"][i], dtype=torch.float32, device=torch.device(device)
            ).reshape(1, 11)
            kappa_src = torch.as_tensor(
                table["kappa"][src_i], dtype=torch.float32, device=torch.device(device)
            ).reshape(1, 11)
            kappa_mean = torch.as_tensor(
                table["kappa"].mean(axis=0), dtype=torch.float32, device=torch.device(device)
            ).reshape(1, 11)
            variants = {
                "baseline": kappa_base,
                "zero": torch.zeros_like(kappa_base),
                "sigma_zero": torch.cat(
                    [torch.zeros_like(kappa_base[:, :10]), kappa_base[:, 10:]], dim=1
                ),
                "mean": kappa_mean,
                "permuted": kappa_src,
                "control_scale1000": kappa_base * PROBE_CTRL_SCALE,
            }
            probs: dict[str, np.ndarray] = {}
            for name, kappa in variants.items():
                holder = {"kappa": kappa}

                def fixed(counts_t, edges_t, _holder=holder):
                    return _holder["kappa"].to(counts_t.dtype)

                original = ev.condition_vector
                ev.condition_vector = fixed
                try:
                    probs[name] = ev.predict_group(model, cfg, counts, edges, N_LOW, device)
                finally:
                    ev.condition_vector = original
            base = probs["baseline"]
            dens = float(np.abs(base).sum())
            row = {"i": i, "beam_id": beam_id, "sub_idx": sub_idx,
                   "err_baseline": group_rel_rmse(base, ref)}
            for name in ("zero", "sigma_zero", "mean", "permuted", "control_scale1000"):
                p = probs[name]
                row[f"l1_rel_{name}"] = float(np.abs(p - base).sum() / max(dens, 1e-300))
                row[f"maxabs_{name}"] = float(np.abs(p - base).max())
                row[f"err_{name}"] = group_rel_rmse(p, ref)
            rows.append(row)
    out = {"run_id": run_id, "n_samples": len(rows), "rows": rows,
           "summary": {}}
    for name in ("zero", "sigma_zero", "mean", "permuted", "control_scale1000"):
        out["summary"][name] = {
            "l1_rel_median": float(np.median([r[f"l1_rel_{name}"] for r in rows])),
            "l1_rel_max": float(np.max([r[f"l1_rel_{name}"] for r in rows])),
            "maxabs_median": float(np.median([r[f"maxabs_{name}"] for r in rows])),
            "err_delta_median": float(np.median([r[f"err_{name}"] - r["err_baseline"] for r in rows])),
            "err_rel_delta_median": float(
                np.median([(r[f"err_{name}"] - r["err_baseline"]) / r["err_baseline"] for r in rows])
            ),
        }
    (HERE / f"output_probe_{run_id}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[probe] {run_id} n={len(rows)} → output_probe_{run_id}.json", flush=True)
    return out["summary"]


# --------------------------------------------------------------------------- #
# 机制级旁证：FiLM 灵敏度（κ 在测试集上的逐样本变动如何传导到调制量）
# --------------------------------------------------------------------------- #

def film_sensitivity(table: dict) -> dict:
    """逐 ckpt / 逐 FiLM 层：调制量 γ, β 在测试集 κ 上的逐样本标准差与量级之比。

    三个口径：
    - ``*_f64``：float64 解析计算（前 10 维贡献 / 第 11 维贡献 / 偏置）；
    - ``*_f32``：**直接调用生产 FiLM 的 Linear 模块**（float32），逐通道统计
      输出在 3000 个测试样本上的标准差与取值去重数——若逐通道去重数 = 1，
      则该层调制在测试集上逐位恒定，κ 的逐样本差异在 float32 中被完全吸收。
    """
    kap = table["kappa"]  # (N, 11) float64
    out: dict = {"n_samples": int(kap.shape[0]), "runs": {}}
    for run_id in RUNS:
        ckpt = RESULTS / run_id / "best_val.ckpt"
        model, _cfg = ev.load_model(ckpt)
        layers = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, FilmLayer):
                continue
            for tag, gen in (("gamma", mod.gamma_gen), ("beta", mod.beta_gen)):
                W = gen.weight.detach().cpu().double().numpy()   # (C, 11)
                b = gen.bias.detach().cpu().double().numpy()     # (C,)
                act = kap @ W.T + b                              # (N, C) float64
                act_sigma_only = kap[:, :10] @ W[:, :10].T       # 前 10 维贡献
                act_logn_only = kap[:, 10:11] @ W[:, 10:].T      # 第 11 维贡献
                level = float(np.abs(act.mean(axis=0)).mean())
                with torch.no_grad():
                    act32 = gen(torch.as_tensor(kap, dtype=torch.float32)).double().numpy()
                distinct = len({row.tobytes() for row in act32})
                layers[f"{name}.{tag}"] = {
                    "channels": int(W.shape[0]),
                    "mean_level": level,
                    "std_across_samples": float(act.std(axis=0).mean()),
                    "rel_std": float(act.std(axis=0).mean() / max(level, 1e-300)),
                    "std_from_sigma_dims": float(act_sigma_only.std(axis=0).mean()),
                    "std_from_logN_dim": float(act_logn_only.std(axis=0).mean()),
                    "std_across_samples_f32": float(act32.std(axis=0).mean()),
                    "distinct_vectors_f32": int(distinct),
                    "bias_abs_mean": float(np.abs(b).mean()),
                    "W_sigma_abs_mean": float(np.abs(W[:, :10]).mean()),
                    "W_logN_abs_mean": float(np.abs(W[:, 10]).mean()),
                }
        out["runs"][run_id] = layers
        del model
    summary = {}
    for tag in ("gamma", "beta"):
        vals = [lay for run in out["runs"].values() for key, lay in run.items()
                if key.endswith(f".{tag}")]
        rel = [x["rel_std"] for x in vals]
        s_sig = [x["std_from_sigma_dims"] for x in vals]
        s_log = [x["std_from_logN_dim"] for x in vals]
        s32 = [x["std_across_samples_f32"] for x in vals]
        dist = [x["distinct_vectors_f32"] for x in vals]
        summary[tag] = {
            "rel_std_max_all_layers": float(np.max(rel)),
            "rel_std_median_all_layers": float(np.median(rel)),
            "std_from_sigma_dims_median": float(np.median(s_sig)),
            "std_from_sigma_dims_max": float(np.max(s_sig)),
            "std_from_logN_dim_median": float(np.median(s_log)),
            "std_from_logN_dim_max": float(np.max(s_log)),
            "std_across_samples_f32_max": float(np.max(s32)),
            "distinct_vectors_f32_max": int(np.max(dist)),
            "n_layers_bitwise_constant_f32": int(np.sum(np.asarray(dist) == 1)),
            "n_layers_total": int(len(dist)),
        }
    out["summary"] = summary
    (HERE / "film_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# 汇总与判定
# --------------------------------------------------------------------------- #

def _load_variant(run_id: str, variant: str) -> dict:
    p = VARIANTS_DIR / f"{run_id}__{variant}.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _per_beam_err(payload: dict) -> dict[int, float]:
    return {int(b["beam_id"]): float(b["err"]) for b in payload["per_beam"]}


def analyze() -> dict:
    """逐 run 复现核对 + 四变体配对比较 + 噪声带 + 预注册判定。"""
    stored = {r: json.loads((RESULTS / r / "test_eval.json").read_text(encoding="utf-8")) for r in RUNS}
    base = {r: _load_variant(r, "baseline") for r in RUNS}
    variants = {r: {v: _load_variant(r, v) for v in ALL_VARIANTS} for r in RUNS}

    reproduce = {}
    for r in RUNS:
        sa, sb = _per_beam_err(stored[r]), _per_beam_err(base[r])
        ids = sorted(sa)
        d = np.array([sb[i] - sa[i] for i in ids])
        rel = np.abs(d) / np.array([sa[i] for i in ids])
        reproduce[r] = {
            "err_median_stored": stored[r]["err_median"],
            "err_median_rerun": base[r]["err_median"],
            "err_median_abs_diff": abs(stored[r]["err_median"] - base[r]["err_median"]),
            "per_beam_max_abs_diff": float(np.abs(d).max()),
            "per_beam_max_rel_diff": float(rel.max()),
            "n_per_beam_diff_nonzero": int((d != 0).sum()),
            "flags_stored": _flag_hist(stored[r]),
            "flags_rerun": _flag_hist(base[r]),
        }

    base_med = np.array([stored[r]["err_median"] for r in RUNS])
    noise = {
        "run_medians": base_med.tolist(),
        "range": float(base_med.max() - base_med.min()),
        "std": float(statistics.stdev(base_med.tolist())),
        "std_over_mean": float(statistics.stdev(base_med.tolist()) / base_med.mean()),
        "range_over_median": float((base_med.max() - base_med.min()) / np.median(base_med)),
    }

    per_variant: dict[str, dict] = {}
    for v in ALL_VARIANTS:
        rows = []
        for r in RUNS:
            eb = _per_beam_err(base[r])
            ev_ = _per_beam_err(variants[r][v])
            ids = sorted(eb)
            delta = np.array([ev_[i] - eb[i] for i in ids])
            rel_delta = np.array([(ev_[i] - eb[i]) / eb[i] for i in ids])
            med_b = float(np.median([eb[i] for i in ids]))
            med_v = float(np.median([ev_[i] for i in ids]))
            rows.append({
                "run_id": r,
                "err_median_baseline": med_b,
                "err_median_variant": med_v,
                "median_abs_change": med_v - med_b,
                "median_rel_change": (med_v - med_b) / med_b,
                "paired_per_beam_delta_median": float(np.median(delta)),
                "paired_per_beam_rel_delta_median": float(np.median(rel_delta)),
                "paired_per_beam_rel_delta_max_abs": float(np.abs(rel_delta).max()),
                "frac_beams_rel_delta_gt_1pct": float(np.mean(np.abs(rel_delta) > 0.01)),
                "frac_beams_exactly_equal": float(np.mean(np.array([ev_[i] == eb[i] for i in ids]))),
                "flags_variant": _flag_hist(variants[r][v]),
                "n_eff_variant": variants[r][v]["n_eff"],
                "n_eff_flag_variant": variants[r][v]["n_eff_flag"],
            })
        rel = np.array([x["median_rel_change"] for x in rows])
        paired = np.array([x["paired_per_beam_rel_delta_max_abs"] for x in rows])
        paired_med = np.array([abs(x["paired_per_beam_rel_delta_median"]) for x in rows])
        per_variant[v] = {
            "per_run": rows,
            "median_rel_change_mean": float(rel.mean()),
            "median_rel_change_min": float(rel.min()),
            "median_rel_change_max": float(rel.max()),
            "median_rel_change_abs_max": float(np.abs(rel).max()),
            "paired_per_beam_rel_delta_median_abs_max": float(paired_med.max()),
            "paired_per_beam_rel_delta_max_abs_over_runs": float(paired.max()),
            "n_runs_rel_change_lt_2pct": int((np.abs(rel) < 0.02).sum()),
            "n_runs_worse_gt_5pct": int((rel > 0.05).sum()),
            "n_runs_better_gt_5pct": int((rel < -0.05).sum()),
        }

    # 预注册判定（"< 噪声带 或 相对变化 < 2%" 按字面取或；另报更严的"< 2% 全部成立"口径）
    arms = ("zero", "mean", "permuted")
    small_literal = {
        v: bool(
            per_variant[v]["median_rel_change_abs_max"] < noise["range_over_median"]
            or per_variant[v]["median_rel_change_abs_max"] < 0.02
        )
        for v in arms
    }
    small_strict = {v: bool(per_variant[v]["median_rel_change_abs_max"] < 0.02) for v in arms}
    worse_4of5 = any(per_variant[v]["n_runs_worse_gt_5pct"] >= 4 for v in arms)
    if worse_4of5:
        verdict = "kappa_carries_information"
    elif all(small_literal.values()):
        verdict = "kappa_uninformative_bypass"
    else:
        verdict = "inconclusive"
    verdict_strict = (
        "kappa_carries_information" if worse_4of5
        else "kappa_uninformative_bypass" if all(small_strict.values())
        else "inconclusive"
    )

    # 变体 b（全零）的归因：κ 的逐样本变动在 float32 中被完全吸收（film_sensitivity），
    # 故"κ→0"等价于给每层 FiLM 加一个**样本无关**的常数偏移（γ(0)−γ(κ₀)、β(0)−β(κ₀)）。
    # b 的指标变化因此度量"网络对常偏置的敏感度"，不是 κ 的逐样本信息量——逐样本信息
    # 由 c/d 两臂判定（二者保留 κ 的均值水平、只打断"样本 ↔ κ"的对应）。
    cd = ["mean", "permuted"]
    floor_ref = per_variant["sigma_zero"]["paired_per_beam_rel_delta_max_abs_over_runs"]
    attribution = {
        "b_zero_median_rel_change_abs_max": per_variant["zero"]["median_rel_change_abs_max"],
        "cd_mean_permuted_median_rel_change_abs_max": max(
            per_variant[v]["median_rel_change_abs_max"] for v in cd
        ),
        "sigma_zero_control_abs_max": per_variant["sigma_zero"]["median_rel_change_abs_max"],
        # 数值噪声地板 = sigma_zero 对照（数学上应零影响）的逐束最大相对变化；
        # 各臂比值 ≈1 表示该臂的效果落在 float32 舍入噪声内（≈0 信息）
        "float_noise_floor_sigma_zero_paired_max_rel": float(floor_ref),
        "arm_paired_max_rel_over_floor": {
            v: float(per_variant[v]["paired_per_beam_rel_delta_max_abs_over_runs"] / max(floor_ref, 1e-300))
            for v in ALL_VARIANTS
        },
        "note": (
            "b 全零同时抹掉常数项（第 11 维，run 内恒定）与前 10 维（float32 惰性）；"
            "c/d 只打断样本↔κ 对应、保留均值水平。b 与 c/d 的量级差即'常偏置敏感度'"
            "与'逐样本信息'之别；sigma_zero（只抹前 10 维）给出'零影响对照'与其数值地板。"
        ),
    }

    summary = {
        "verdict": verdict,
        "verdict_strict_all_variants_lt_2pct": verdict_strict,
        "small_within_noise_literal": small_literal,
        "small_lt_2pct_strict": small_strict,
        "noise_band": noise,
        "reproduction": reproduce,
        "per_variant": per_variant,
        "variant_zero_attribution": attribution,
        "permutation": {
            "derivation": "SeedSequence([20260825, *utf8('exp2'), *utf8('kappa_crossbeam_permutation')]) → "
                          "default_rng().permutation(150)，重抽至无不动点",
            "index": permutation_index(load_table()).tolist(),
        },
        "runinfos": {
            r: {v: json.loads((VARIANTS_DIR / f"{r}__{v}.runinfo.json").read_text(encoding="utf-8"))
                for v in ALL_VARIANTS if (VARIANTS_DIR / f"{r}__{v}.runinfo.json").is_file()}
            for r in RUNS
        },
    }
    (HERE / "exp2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "noise_band": noise}, ensure_ascii=False, indent=2))
    return summary


def _flag_hist(payload: dict) -> dict:
    hist: dict[str, int] = {}
    for b in payload["per_beam"]:
        key = str(b["n_eff_flag"])
        hist[key] = hist.get(key, 0) + 1
    return hist


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _eval_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("EXP2_GPU", "1")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"
    return env


def drive(workers: int, device: str) -> int:
    if not TABLE_NPZ.is_file():
        scan()
    jobs = [(r, v) for r in RUNS for v in ALL_VARIANTS]
    argv_base = [sys.executable, "-u", str(Path(__file__).resolve())]
    probe_target = ("e1_joint_s0", "baseline")

    def one(job):
        run_id, variant = job
        argv = argv_base + ["eval", "--run", run_id, "--variant", variant, "--device", device]
        if job == probe_target:
            argv.append("--probe")
        log = HERE / "logs" / f"{run_id}__{variant}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with log.open("wb") as fh:
            rc = subprocess.run(argv, cwd=ROOT, env=_eval_env(), stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[drive] {run_id} {variant:9s} rc={rc} {time.time() - t0:6.1f}s", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rcs = list(pool.map(one, jobs))
    if any(rc != 0 for rc in rcs):
        print(f"[drive] 有 {sum(rc != 0 for rc in rcs)} 个作业失败，终止", flush=True)
        return 1
    film_sensitivity(load_table())
    analyze()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T0-实验2：κ 推理期消融（诊断）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="预扫描测试集 κ 全表")
    p_eval = sub.add_parser("eval", help="单 (run, 变体) 评估")
    p_eval.add_argument("--run", required=True, choices=RUNS)
    p_eval.add_argument("--variant", required=True, choices=ALL_VARIANTS)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--probe", action="store_true", help="附加输出级探针")
    p_drive = sub.add_parser("drive", help="全 5 run × 4 变体（子进程串行/并发）")
    p_drive.add_argument("--workers", type=int, default=3)
    p_drive.add_argument("--device", default=EVAL_ENV_DEVICE)
    sub.add_parser("analyze", help="汇总 + 预注册判定")
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        scan()
        return 0
    if args.cmd == "eval":
        run_variant(args.run, args.variant, args.device, probe=args.probe)
        return 0
    if args.cmd == "drive":
        return drive(args.workers, args.device)
    if args.cmd == "analyze":
        film_sensitivity(load_table())
        analyze()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
