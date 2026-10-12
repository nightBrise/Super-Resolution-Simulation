"""前置① 第 1 层管线审计：抖动作用点、快照路径、bin_edges 约定与跨束域统计。

只读 src/、data/；只写本目录。CPU only。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent
DS_H5 = ROOT / "studies/line2_particle_sr/data/line2_ds_v1/test.h5"
PAD = 0.01
AXES = ("x", "xp", "y", "yp")


def edge_domain(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """edges (4,129) → 该轴真实 [min,max]（反解 1% 外扩）。"""
    lo = edges[:, 0] + PAD * (edges[:, -1] - edges[:, 0]) / (1.0 + 2 * PAD)
    hi = edges[:, -1] - PAD * (edges[:, -1] - edges[:, 0]) / (1.0 + 2 * PAD)
    return lo, hi


def main() -> int:
    with h5py.File(DS_H5, "r") as f:
        beams = sorted(int(k.split("_")[1]) for k in f.keys())
        dom = {0: [], 1: []}
        for b in beams:
            for st in (0, 1):
                e = f[f"beam_{b:04d}"][f"station{st}"]["target_full"].attrs["bin_edges"]
                dom[st].append(edge_domain(e))
        dom = {st: (np.stack([d[0] for d in dom[st]]), np.stack([d[1] for d in dom[st]]))
               for st in (0, 1)}
        # A1：h5 抖动 attrs 与生产派生器逐位对账（抖动是否确为该束自己的参数）
        from src.line2.data_pipeline.transport import jitter_params

        jitter_mismatch = []
        for b in beams:
            attrs = f[f"beam_{b:04d}"].attrs
            for k, v in jitter_params(b).items():
                if not np.isclose(attrs[f"jitter_{k}"], v, rtol=0.0, atol=0.0):
                    jitter_mismatch.append([b, k, float(attrs[f"jitter_{k}"]), v])

    report: dict = {"n_test_beams": len(beams), "per_axis": {},
                    "jitter_attrs_vs_producer_mismatches": jitter_mismatch}
    for st in (0, 1):
        lo, hi = dom[st]
        w = hi - lo
        c = 0.5 * (lo + hi)
        rec = {}
        for i, a in enumerate(AXES):
            rec[a] = {
                "window_width_median": float(np.median(w[:, i])),
                "window_width_rel_spread": float(w[:, i].std() / w[:, i].mean()),
                "center_median": float(np.median(c[:, i])),
                "center_abs_median_over_width": float(
                    np.median(np.abs(c[:, i])) / np.median(w[:, i])),
                "center_std_over_width": float(c[:, i].std() / np.median(w[:, i])),
                "window_width_minover_max": float(w[:, i].min() / w[:, i].max()),
            }
        report["per_axis"][f"station{st}"] = rec
    # 站间窗口宽度比（同一束）
    report["width_ratio_s1_over_s0"] = {
        a: float(np.median(dom[1][1][:, i] - dom[1][0][:, i])
                 / np.median(dom[0][1][:, i] - dom[0][0][:, i]))
        for i, a in enumerate(AXES)
    }
    report["note"] = (
        "窗口 = 该束该站全量表逐轴 [min,max]（外扩 1%）；production 口径为逐束自定标，"
        "站 0/站 1 各自独立，故两站窗口不可比、且逐束平移与逐轴尺度在分箱时被消去。"
    )
    (OUT / "pre1_edge_domains.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 逐束窗口宽度长表（供后续画图/核对）
    lines = ["beam,station," + ",".join(f"{a}_{s}" for a in AXES for s in ("lo", "hi", "w"))]
    for st in (0, 1):
        lo, hi = dom[st]
        for n, b in enumerate(beams):
            vals = []
            for i in range(4):
                vals += [f"{lo[n, i]:.10e}", f"{hi[n, i]:.10e}", f"{hi[n, i] - lo[n, i]:.10e}"]
            lines.append(f"{b},{st}," + ",".join(vals))
    (OUT / "pre1_edge_domains.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
