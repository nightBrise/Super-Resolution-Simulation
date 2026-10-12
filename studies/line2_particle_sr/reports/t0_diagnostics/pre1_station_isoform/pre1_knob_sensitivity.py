"""前置① 抖动旋钮 → target 图灵敏度：三个抖动参数各自对生产的 target 图有无可见影响。

单束实验（取 test 束 245 的抖动参数为基准，逐次只改一个旋钮）：
- θ（踢角）改动 = 平移 ⇒ 生产口径（逐束 min/max 自定标）理论上严格不可见；
- δ（薄透镜焦距 ±5%）、ℓ（前置漂移 ±1%）改动 = 线性映射 ⇒ 改变二阶矩形状。
每格同时给「生产口径 M1」与「公共域 M1（两表合并域，保留平移）」两个读数。

只读 src/、data/；只写本目录。CPU only。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from ocelot.cpbd.beam import ParticleArray  # noqa: E402
from ocelot.cpbd.track import track  # noqa: E402

from src.line2.data_pipeline.anchor import read_anchor_particles  # noqa: E402
from src.line2.data_pipeline.projections import (  # noqa: E402
    AXES, N_BINS, six_projections, station_edges,
)
from src.line2.data_pipeline.transport import (  # noqa: E402
    apply_jitter, fodo_lattice, jitter_params, transport_beam,
)

OUT = Path(__file__).resolve().parent
ANCHOR = ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
BEAM = 245


def group_m1(a: np.ndarray, b: np.ndarray) -> float:
    """a/b 总计数相同下的组级 M1（= group_rel_rmse 的等价式）。"""
    num = np.sqrt(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=(1, 2)))
    return float(np.mean(num / (b.sum(axis=(1, 2)) / (N_BINS * N_BINS))))


def union_edges(*tables: np.ndarray) -> np.ndarray:
    lo = np.min([t.min(axis=0) for t in tables], axis=0)
    hi = np.max([t.max(axis=0) for t in tables], axis=0)
    pad = (hi - lo) * 0.01
    return np.stack([np.linspace(lo[i] - pad[i], hi[i] + pad[i], N_BINS + 1) for i in range(4)])


def main() -> int:
    anchor = read_anchor_particles(ANCHOR)
    p0 = jitter_params(BEAM)
    variants = {
        "theta_x_+5urad": {**p0, "theta_x_rad": p0["theta_x_rad"] + 5e-6},
        "theta_x_-5urad": {**p0, "theta_x_rad": p0["theta_x_rad"] - 5e-6},
        "theta_y_+5urad": {**p0, "theta_y_rad": p0["theta_y_rad"] + 5e-6},
        "delta_+0.025": {**p0, "delta": p0["delta"] + 0.025,
                         "f_j": 10.0 * (1.0 + p0["delta"] + 0.025)},
        "delta_-0.025": {**p0, "delta": p0["delta"] - 0.025,
                         "f_j": 10.0 * (1.0 + p0["delta"] - 0.025)},
        "drift_+5mm": {**p0, "drift_m": p0["drift_m"] + 0.005},
        "drift_-5mm": {**p0, "drift_m": p0["drift_m"] - 0.005},
    }
    ref = transport_beam(anchor, beam_idx=BEAM)
    out: dict = {"beam": BEAM, "jitter_params": p0, "rows": {}}
    for name, p in variants.items():
        s0 = apply_jitter(anchor, p)
        # 变体参数走与生产 transport_beam 相同的 ocelot 路径（同一 lattice 单例）
        p_arr = ParticleArray(s0.shape[0])
        p_arr.rparticles[:4, :] = s0[:, :4].T
        p_arr.rparticles[4:6, :] = 0.0
        p_arr.E = 130.0
        _, p_arr = track(fodo_lattice(), p_arr, print_progress=False, calc_tws=False)
        s1 = s0.copy()
        s1[:, :4] = p_arr.rparticles[:4, :].T
        row = {}
        for st, (t_ref, t_var) in enumerate(zip(ref, (s0, s1))):
            img_ref = six_projections(t_ref, station_edges(t_ref))
            img_var = six_projections(t_var, station_edges(t_var))
            e = union_edges(t_ref, t_var)
            img_ref_c = six_projections(t_ref, e)
            img_var_c = six_projections(t_var, e)
            row[f"station{st}"] = {
                "M1_production": group_m1(img_var, img_ref),
                "M1_common_frame": group_m1(img_var_c, img_ref_c),
                "mean_shift_over_sigma": [
                    float(abs(t_var[:, i].mean() - t_ref[:, i].mean()) / t_ref[:, i].std(ddof=1))
                    for i in range(4)],
            }
        out["rows"][name] = row
    (OUT / "pre1_knob_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
