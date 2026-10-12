"""T0 实验 4 附：1e5 档裸子样本的观测量偏差分布（抽样噪声地板 + 容差线口径校验）。

用途两条：

1. **口径校验**：用 `metrics.tolerance.deviation_table` 的同一代码路径重算 1e5 档 95% 分位
   （val+test 300 束 × 站 0/1 × k=20 子样本），与 `results/tolerance.json` 的 ``100000`` 行
   逐分量比对——确认本实验对"容差线从何而来"的读法与官方产物一致。
2. **地板参照**：容差线本身是**裸子样本**（无模型）的 |Δ| 95% 分位，故裸分布给出了
   "不做任何估计"的误差形状与尺度。模型观测量的中位数/MAD 与该地板对比，可判模型误差
   是"在地板量级上的随机误差"还是"远超地板的共同成分"。

裸输入的**符号**统计一并产出（M4 的 11 分量 + M5/M6 的六投影符号均值）：参照估计量若
基本无符号偏移，则容差线是一条噪声线，模型的幅值中位数可直接读作"超出地板多少"。

只读 ``data/``；输出写本目录。运行：``python exp4_bare_floor.py``（约 6 分钟）。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
if str(REPO) not in sys.path:  # 以脚本路径直接运行时补项目根（`src.` 导入）
    sys.path.insert(0, str(REPO))

from src.line2.experiments.gates import OBSERVABLE_NAMES  # noqa: E402
from src.line2.metrics.metrics import (  # noqa: E402
    _HF_SIGMA_BINS,
    _corr,
    _halo_fraction_per_projection,
    reconstruct_sigma,
)
from src.line2.metrics.n_eff import _beam_split, _dataset_layout  # noqa: E402
from src.line2.metrics.tolerance import observable_vector  # noqa: E402

DATA_DIR = REPO / "studies" / "line2_particle_sr" / "data" / "line2_ds_v1"
RESULTS = REPO / "studies" / "line2_particle_sr" / "results"
TIER = 100_000
K = 20
STATIONS = (0, 1)

_PAIRS = (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))


def signed_observables(group: np.ndarray, ref: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """13 维**带符号**偏差向量（幅值口径之外的诊断量，不参与容差线）。

    M4 的 11 分量取 `sigma_errors` 的符号原值；M5/M6 取六投影符号差的均值
    （绝对值版即 `observable_vector` 的对应分量）。
    """
    s_e = reconstruct_sigma(group, edges)
    s_r = reconstruct_sigma(ref, edges)
    diag = [(0, 0), (1, 1), (2, 2), (3, 3)]
    off = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    signed = [(s_e[i, j] - s_r[i, j]) / abs(s_r[i, j]) for i, j in diag]
    rho_e, rho_r = _corr(s_e), _corr(s_r)
    signed += [rho_e[i, j] - rho_r[i, j] for i, j in off]
    eps_e, eps_r = np.sqrt(np.linalg.det(s_e)), np.sqrt(np.linalg.det(s_r))
    signed.append((eps_e - eps_r) / abs(eps_r))
    axis_of = {a: i for i, a in enumerate(("x", "xp", "y", "yp"))}
    halo_gaps, hf_gaps = [], []
    for gi, (ai, aj) in enumerate(_PAIRS):
        i, j = axis_of[ai], axis_of[aj]
        sub = np.array([[s_r[i, i], s_r[i, j]], [s_r[j, i], s_r[j, j]]])
        img_r = ref[gi].astype(np.float64)
        ui = 0.5 * (edges[i][:-1] + edges[i][1:])
        uj = 0.5 * (edges[j][:-1] + edges[j][1:])
        pr = img_r.T / img_r.sum()
        means = np.array([float((pr.sum(axis=1) * ui).sum()), float((pr.sum(axis=0) * uj).sum())])
        halo_gaps.append(_halo_fraction_per_projection(group[gi].T, sub, means, ui, uj)
                         - _halo_fraction_per_projection(img_r.T, sub, means, ui, uj))
        ratios = []
        for img in (group[gi].astype(np.float64), img_r):
            hp = img - gaussian_filter(img, _HF_SIGMA_BINS)
            ratios.append(float(np.abs(hp).sum() / max(img.sum(), 1e-12)))
        hf_gaps.append(ratios[0] - ratios[1])
    signed.append(float(np.mean(halo_gaps)))
    signed.append(float(np.mean(hf_gaps)))
    return np.asarray(signed, dtype=np.float64)


def main() -> int:
    manifest, data_dir = _dataset_layout(DATA_DIR)
    beams = [b for s in ("val", "test") for b in manifest["splits"][s]]
    tables = {st: [] for st in STATIONS}
    signed_tables = {st: [] for st in STATIONS}
    per_beam_mag = []  # (beam, station, 13) 子样本中位数
    files: dict[str, h5py.File] = {}
    try:
        for n_done, beam_id in enumerate(beams, start=1):
            split = _beam_split(manifest, beam_id)
            h5 = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            grp = h5[f"beam_{beam_id:04d}"]
            for st in STATIONS:
                stg = grp[f"station{st}"]
                ref = stg["target_full"][...]
                edges = stg["target_full"].attrs["bin_edges"]
                mags, signs = [], []
                for sub in range(K):
                    img = stg[f"tier_{TIER:06d}_sub_{sub:02d}"][...]
                    mags.append(observable_vector(img, ref, edges))
                    signs.append(signed_observables(img, ref, edges))
                mags = np.asarray(mags)
                tables[st].append(mags)
                signed_tables[st].append(np.asarray(signs))
                per_beam_mag.append((beam_id, st, np.median(mags, axis=0)))
            if n_done % 25 == 0:
                print(f"[bare] {n_done}/{len(beams)} 束", file=sys.stderr, flush=True)
    finally:
        for f in files.values():
            f.close()

    dump = json.loads((RESULTS / "tolerance.json").read_text(encoding="utf-8"))
    tol_row = np.asarray(dump["dev_table"][str(TIER)], dtype=np.float64)
    pooled = np.concatenate([np.concatenate(tables[st]) for st in STATIONS], axis=0)
    q95 = np.percentile(pooled, 95.0, axis=0)
    signed_pooled = np.concatenate([np.concatenate(signed_tables[st]) for st in STATIONS], axis=0)

    rows = []
    test_beams = set(manifest["splits"]["test"])
    test_station0 = np.asarray([m for b, st, m in per_beam_mag if b in test_beams and st == 0])
    for i, name in enumerate(OBSERVABLE_NAMES):
        x = pooled[:, i]
        med = float(np.median(x))
        sgn = signed_pooled[:, i]
        rows.append({
            "index": i, "name": name,
            "n_samples": int(x.size),
            "median": med,
            "mean": float(np.mean(x)),
            "mad": float(np.median(np.abs(x - med))),
            "mad_over_median": float(np.median(np.abs(x - med)) / med),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
            "signed_median": float(np.median(sgn)),
            "signed_mean": float(np.mean(sgn)),
            "signed_std": float(np.std(sgn, ddof=1)),
            "signed_median_over_std": float(abs(np.median(sgn)) / np.std(sgn, ddof=1)),
            "tol_landed_1e5": float(tol_row[i]),
            "q95_recomputed": float(q95[i]),
            "q95_abs_diff": float(abs(q95[i] - tol_row[i])),
            "q95_rel_diff": float(abs(q95[i] - tol_row[i]) / tol_row[i]),
            "test_beam_median_station0": float(np.median(test_station0[:, i])),
            "test_beam_median_over_tol": float(np.median(test_station0[:, i]) / tol_row[i]),
        })

    with (HERE / "exp4_bare_floor.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tier": TIER,
        "k": K,
        "stations": list(STATIONS),
        "n_beams": len(beams),
        "n_samples": int(pooled.shape[0]),
        "tolerance_parity": {
            "max_abs_diff": float(np.max(np.abs(q95 - tol_row))),
            "max_rel_diff": float(np.max(np.abs(q95 - tol_row) / tol_row)),
            "allclose_rtol_1e-6": bool(np.allclose(q95, tol_row, rtol=1e-6)),
        },
        "observables": rows,
    }
    (HERE / "exp4_bare_floor.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary["tolerance_parity"], ensure_ascii=False, indent=2))
    print("bare |Δ| 中位数 / 容差线：",
          {name: round(r["test_beam_median_over_tol"], 4) for name, r in
           zip(OBSERVABLE_NAMES, rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
