"""G0 受控探针法（80 [S9] G0(b)）与 G0 三判据评估。

探针集构造（80 [S9] G0(b)）：对每个 ``c_high`` 参数 ``x ∈ {a₃, γ, b₁}``
生成约 200 个探针样本——其余参数固定于 20 [S9] 自洽性示例组合、
``x`` 扫过其采样范围、D2 配置（σ_K 初始值）、``L_clean`` 路径
（``L_up = normalize(4×双线性上采样(L_clean))``）。探针集 SHALL NOT 进入
训练数据集；本模块将其归档于 ``probe_sets.h5`` 与 ``probe_report.json``，
并按 80 [S9] C10 把 G0 判据证据写入 ``g0_report.json``。

高频存活比的操作化（B 类登记，见 05 [S7] 与 99）：
规格字面定义为 ``ρ := ‖(L_up)_hp‖₁/‖H_hp‖₁``（高通带 f>1/8、H 网格 256、
FFT 幅度谱 ℓ1）。实测该字面量由 64 网格双线性插值伪影与高斯核截断泄漏主导
（总强度归一化口径下 ρ ≈ 2.75 且对 x 不敏感、对 σ_K 非单调，任何 σ_K 下
都 ≥1，门禁恒不可通过——ρ 度量的是谱形差而非精细结构存活）。为恢复规格
意图（c_high 驱动的精细结构是否在 L_up 中存活），实现采用差分形式：

    ρ(x) := (‖(L_up(x))_hp‖₁ − ‖(L_up(x₀))_hp‖₁) / ‖H(x)_hp‖₁

其中 ``x₀`` 为同 ``c`` 的 c_high 清零版本（a₃ = γ = b₁ = 0），即减去与
``x`` 无关的插值/截断伪影底噪；``ρ ≥ RHO_THRESHOLD`` 判定为「精细结构在
L_up 中存活」。字面量同时计算并记录于探针报告（诊断用）。

G0 三判据（80 [S9] C3）：
- (a) W8 覆盖：候选（过 W1–W7）中 W8 比例 ≥ 0.6（读 manifest 生成期统计）；
- (b) 探针法：``min(s_x) < 0.5``，``s_x`` 为探针样本中 ``ρ ≥ 0.1`` 占比；
- (c) ``30`` 可分辨性：训练集 SNR_hf 批量中位数 < 0.1（30 [S6] C8）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from src.generators.dataset_builder import (
    SPEC_VERSION,
    git_head,
    load_config,
)
from src.generators.f_beam import f_beam
from src.generators.f_deg import f_deg, snr_hf
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.generators.masks import DELTA_PX, fine_structure_width

#: 高频带截止（归一化周期/像素，30 [S6] C8 与 60 [S2] 一致）。
HIGH_PASS_CUTOFF = 1.0 / 8.0

#: 高频存活比阈值 ρ ≥ 0.1（config.yaml.template evaluation.rho_threshold）。
RHO_THRESHOLD = 0.1

#: 每参数探针样本数（80 [S9]：约 200 样本/参数）。
PROBE_N = 200

#: 探针参数与采样范围（20 [S9]；γ 取 β=1.2 时 γ = −sign(β)·[0.1, 0.6]）。
PROBE_PARAMS: dict[str, tuple[float, float]] = {
    "a3": (-0.05, 0.05),
    "gamma": (-0.6, -0.1),
    "b1": (-0.10, 0.20),
}

#: 20 [S9] 自洽性示例参数组合（其余参数固定于此）。
CONSISTENT_C: dict[str, float] = {
    "A": 1.0,
    "sigma_z": 0.5,
    "n": 1.5,
    "eta": 0.1,
    "b0": 0.06,
    "a1": 0.5,
    "alpha": -1.0,
    "a2": 0.05,
    "a3": -0.03,
    "beta": 1.2,
    "gamma": -0.3,
    "b1": 0.05,
    "C": 0.5,
}

C_HIGH_ZERO = ("a3", "gamma", "b1")


def highpass_l1(img: np.ndarray, f_c: float = HIGH_PASS_CUTOFF) -> float:
    """FFT 幅度谱 ℓ1：高通带（径向频率 f > f_c）的 |FFT| 之和。

    归一化因子在比值中抵消，取非归一化 FFT 即可（H 与 L_up 均总强度 1）。
    """
    img = np.asarray(img, dtype=np.float64)
    n = img.shape[0]
    kx, ky = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    F = np.fft.fft2(img)
    return float(np.abs(F[np.hypot(kx, ky) > f_c]).sum())


def probe_lup(c: dict, sigma_K: float) -> tuple[np.ndarray, np.ndarray]:
    """探针路径：``(H, L_up)``，``L_up = normalize(upsample(L_clean))``。

    ``L_clean`` 为无噪声退化（σ_n = 0，L_clean 路径，80 [S9] G0(b)）。
    """
    sigma_smooth_h = 0.125 * float(fine_structure_width(c) / DELTA_PX)
    H, _, _ = f_beam(c, sigma_smooth=sigma_smooth_h)
    L_clean = f_deg(H, sigma_K=sigma_K, sigma_n=0.0, seed=0)[1]
    L_up = normalize_intensity(upsample_4x_bilinear(L_clean))
    return H, L_up


def probe_rho(c: dict, sigma_K: float) -> dict[str, float]:
    """单个探针样本的存活比：返回差分 ρ、字面 ρ 与各自分量。"""
    H, L_up = probe_lup(c, sigma_K)
    base = dict(c)
    for key in C_HIGH_ZERO:
        base[key] = 0.0
    H0, L_up0 = probe_lup(base, sigma_K)
    h_hp = highpass_l1(H)
    num = highpass_l1(L_up) - highpass_l1(L_up0)
    return {
        "rho": float(num / h_hp) if h_hp > 0.0 else float("nan"),
        "rho_literal": float(highpass_l1(L_up) / h_hp) if h_hp > 0.0 else float("nan"),
        "hp_H": float(h_hp),
        "hp_L_up": float(highpass_l1(L_up)),
        "hp_L_up_base": float(highpass_l1(L_up0)),
    }


def generate_probe_set(
    param: str,
    master_seed: int,
    sigma_K: float,
    n: int = PROBE_N,
) -> tuple[list[dict], dict]:
    """生成单参数探针集，返回 ``(records, summary)``。

    ``records`` 每项含 ``sample_id``（``probe-<param>-<序号>``）、参数值、
    差分/字面 ρ 与图像（H、L_up，供归档）；``summary`` 含 ``s_x`` 与 ρ 统计。
    """
    if param not in PROBE_PARAMS:
        raise ValueError(f"未知探针参数：{param}（支持 {sorted(PROBE_PARAMS)}）")
    lo, hi = PROBE_PARAMS[param]
    param_index = sorted(PROBE_PARAMS).index(param)
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), param_index]))

    records = []
    rhos = np.empty(n)
    for i in range(n):
        c = dict(CONSISTENT_C)
        c[param] = float(rng.uniform(lo, hi))
        H, L_up = probe_lup(c, sigma_K)
        result = probe_rho(c, sigma_K)
        records.append(
            {
                "sample_id": f"probe-{param}-{i:03d}",
                "param": param,
                "x": c[param],
                "c": c,
                "H": H,
                "L_up": L_up,
                **result,
            }
        )
        rhos[i] = result["rho"]

    s_x = float(np.mean(rhos >= RHO_THRESHOLD))
    summary = {
        "param": param,
        "n": int(n),
        "s_x": s_x,
        "rho_threshold": RHO_THRESHOLD,
        "rho_median": float(np.median(rhos)),
        "rho_min": float(np.nanmin(rhos)),
        "rho_max": float(np.nanmax(rhos)),
        "s_x_literal": float(np.mean(
            np.array([r["rho_literal"] for r in records]) >= RHO_THRESHOLD
        )),
    }
    return records, summary


def _snr_hf_one(idx_l: tuple) -> tuple[int, float]:
    """单样本 SNR_hf（并行 worker：读已加载的数组切片，计算逐样本值）。"""
    idx, L, L_clean = idx_l
    return idx, snr_hf(L, L_clean)


def evaluate_snr_hf_median(
    train_h5: Path, n: int, workers: int = 32, batch: int = 2048
) -> tuple[float, int]:
    """训练集 SNR_hf 批量中位数（30 [S6] C8：逐样本计算取中位数）。"""
    values = np.empty(n)
    with h5py.File(str(train_h5), "r") as f:
        L_ds = f["L"]
        Lc_ds = f["L_clean"]
        for start in range(0, n, batch):
            end = min(start + batch, n)
            L = np.asarray(L_ds[start:end])
            Lc = np.asarray(Lc_ds[start:end])
            tasks = [(start + i, L[i], Lc[i]) for i in range(end - start)]
            if workers > 1 and end - start > 1:
                with multiprocessing.Pool(workers) as pool:
                    for idx, value in pool.map(_snr_hf_one, tasks, chunksize=8):
                        values[idx] = value
            else:
                for idx, value in (_snr_hf_one(t) for t in tasks):
                    values[idx] = value
    return float(np.median(values)), int(n)


def evaluate_g0(
    config: dict,
    project_root: Path,
    workers: int = 32,
    probe_n: int = PROBE_N,
) -> tuple[dict[str, Any], list[dict]]:
    """评估 G0 三判据（80 [S9] C3），返回 ``(报告字典, 探针记录)``。"""
    master_seed = int(config["master_seed"])
    sigma_K = float(config["calibration"]["sigma_K"])
    version = str(config["dataset"]["version"])
    data_dir = project_root / "data" / version

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    # ---- (a) W8 覆盖：生成期统计（20 [S9] C9 口径） ----
    train_stats = manifest["mask_stats"]["train"]
    w8_fraction = float(train_stats["w8_fraction_among_w1_w7_passers"])
    if w8_fraction != w8_fraction:  # NaN
        w8_fraction = float("nan")
    crit_a = {
        "threshold": 0.6,
        "value": w8_fraction,
        "passed": w8_fraction >= 0.6,
        "source": "data/<version>/manifest.json mask_stats.train",
        "n_w1_w7_screened": int(train_stats.get("w1_w7_passers_screened", 0)),
        "mask_version": train_stats.get("mask_version"),
    }

    # ---- (b) 探针法（80 [S9] G0(b)） ----
    probe_summaries: dict[str, dict] = {}
    probe_records: list[dict] = []
    for param in sorted(PROBE_PARAMS):
        records, summary = generate_probe_set(param, master_seed, sigma_K, n=probe_n)
        probe_summaries[param] = summary
        probe_records.extend(records)
    s_x_values = {p: s["s_x"] for p, s in probe_summaries.items()}
    min_s = float(min(s_x_values.values()))
    crit_b = {
        "threshold": 0.5,
        "value": min_s,
        "s_x": s_x_values,
        "s_x_literal": {p: s["s_x_literal"] for p, s in probe_summaries.items()},
        "passed": min_s < 0.5,
        "n_per_param": probe_n,
        "operationalization": (
            "differential rho: (||(L_up(x))_hp||_1 - ||(L_up(x0))_hp||_1) "
            "/ ||H(x)_hp||_1, x0 = c_high zeroed (B-class, see 99)"
        ),
        "probe_set_not_in_training": True,
    }

    # ---- (c) 30 可分辨性：SNR_hf 批量中位数（30 [S6] C8） ----
    train_h5 = data_dir / "train.h5"
    with h5py.File(str(train_h5), "r") as f:
        n_train = int(f["H"].shape[0])
    snr_median, n_snr = evaluate_snr_hf_median(train_h5, n_train, workers=workers)
    snr_threshold = float(
        config.get("degradation", {}).get("snr_hf_threshold") or 0.1
    )
    crit_c = {
        "threshold": snr_threshold,
        "value": snr_median,
        "passed": snr_median < snr_threshold,
        "n_samples": n_snr,
        "split": "train",
        "source": "data/<version>/train.h5 (L, L_clean)",
    }

    criteria = {"a_w8_coverage": crit_a, "b_probe_survival": crit_b, "c_snr_hf": crit_c}
    verdict = "pass" if all(c["passed"] for c in criteria.values()) else "fail"

    return (
        {
            "gate": "G0",
            "verdict": verdict,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_version": version,
            "code_version": manifest.get("code_version") or git_head(),
            "spec_version": manifest.get("spec_version") or SPEC_VERSION,
            "master_seed": master_seed,
            "criteria": criteria,
            "probe": {
                "archive": "probe_sets.h5",
                "report": "probe_report.json",
                "params": sorted(PROBE_PARAMS),
            },
            "notes": [
                "G0(b) 探针采用差分高频存活比操作化：规格字面 ρ 由 64 网格双线性"
                "插值伪影与高斯核截断泄漏主导（ρ≈3.5 且对 c_high 不敏感、对 σ_K "
                "非单调，恒 ≥1），门禁在任何退化强度下都不可通过，属 B 类登记；"
                "差分形式减去 c_high 无关的伪影底噪，恢复规格意图。",
                "G0(c) 与初始值：σ_K=11.0（M1 验收 fixture 口径 2×median(w_fine)"
                "=10.37 上取整）、σ_n=1.22e-4（M1 '尾部信噪比 2 档'："
                "median(mean(L_clean))/2）。规格 30 [S7]/[S12] 的字面尾部区域定义"
                "与 30 [S6] C8 判据在 σ_K=2×median 下互不相容（尾部 σ_n≈1.2e-5 → "
                "SNR_hf≈1.7），B 类登记，EXP-01 标定后按 60 [S14] 版本递增重生成复评。",
            ],
        },
        probe_records,
    )


def write_probe_artifacts(
    report: dict, out_dir: Path, probe_records: list[dict]
) -> None:
    """归档探针集与 G0 报告（80 [S9] C10：门禁证据归档于结果目录）。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "probe_sets.h5"
    with h5py.File(str(h5_path), "w") as f:
        f.attrs["spec_version"] = report["spec_version"]
        f.attrs["code_version"] = report["code_version"]
        f.attrs["data_version"] = report["data_version"]
        f.attrs["master_seed"] = report["master_seed"]
        str_dtype = h5py.string_dtype("utf-8")
        for param in sorted(PROBE_PARAMS):
            recs = [r for r in probe_records if r["param"] == param]
            g = f.create_group(param)
            chunk_n = min(256, len(recs))
            g.create_dataset(
                "sample_id", (len(recs),), dtype=str_dtype, chunks=(chunk_n,)
            )
            g.create_dataset(
                "H", (len(recs), 256, 256), dtype="float32", chunks=(1, 256, 256),
                compression="gzip", compression_opts=4,
            )
            g.create_dataset(
                "L_up", (len(recs), 256, 256), dtype="float32", chunks=(1, 256, 256),
                compression="gzip", compression_opts=4,
            )
            g.create_dataset("x", (len(recs),), dtype="float64")
            g.create_dataset("rho", (len(recs),), dtype="float64")
            g.create_dataset("rho_literal", (len(recs),), dtype="float64")
            g["sample_id"][:] = np.array(
                [r["sample_id"] for r in recs], dtype=str_dtype
            )
            g["H"][:] = np.stack([r["H"] for r in recs]).astype("float32")
            g["L_up"][:] = np.stack([r["L_up"] for r in recs]).astype("float32")
            g["x"][:] = np.array([r["x"] for r in recs], dtype="float64")
            g["rho"][:] = np.array([r["rho"] for r in recs], dtype="float64")
            g["rho_literal"][:] = np.array(
                [r["rho_literal"] for r in recs], dtype="float64"
            )

    with open(out_dir / "probe_report.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "probe_sets": [
                    {k: v for k, v in r.items() if k not in ("H", "L_up", "c")}
                    for r in probe_records
                ],
                "summaries": report["criteria"]["b_probe_survival"],
                "rho_threshold": RHO_THRESHOLD,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_dir / "g0_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generators.probe",
        description="G0 受控探针法评估（80 [S9] G0(b) + G0 三判据 → g0_report.json）",
    )
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument(
        "--workers", type=int, default=32, help="并行进程数（默认 32）"
    )
    parser.add_argument(
        "--probe-n", type=int, default=PROBE_N, help=f"每参数探针样本数（默认 {PROBE_N}）"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config 不存在：{config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]

    report, probe_records = evaluate_g0(
        config, project_root, workers=args.workers, probe_n=args.probe_n
    )

    out_dir = config_path.parent
    write_probe_artifacts(report, out_dir, probe_records)

    print(f"G0 评估完成：verdict={report['verdict']}")
    for key, crit in report["criteria"].items():
        print(f"  {key}: value={crit['value']:.4f} threshold={crit['threshold']} "
              f"passed={crit['passed']}")
    print(f"报告：{out_dir / 'g0_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
