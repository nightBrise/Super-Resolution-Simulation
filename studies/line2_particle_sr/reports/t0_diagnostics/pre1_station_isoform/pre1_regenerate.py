"""前置① 数据再生：150 个 test 束的站 0/站 1 粒子表 → 逐束矩 + 四种分箱口径的图组。

只读 src/、data/；只写本目录。CPU only，无随机性（传输链确定性）。

分箱口径（每站各一套；px = 投影索引，六投影同口径）：
- V1 production：edges = 该束该站全量表逐轴 [min,max] 外扩 1%（= 生产 h5 `target_full` 口径）；
- V2 common    ：edges = 150 束池化逐轴 [min,max] 外扩 1%（固定公共物理域，平移+尺度+形状全保留）；
- V2b centered ：逐束逐轴减自身均值后再池化定标（去掉平移，保留尺度+形状）；
- V2c shape    ：逐束逐轴标准化 u=(x−μ)/σ，公共窗口 ±K_SIGMA σ（去平移+尺度，只留形状）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.line2.data_pipeline.anchor import read_anchor_particles  # noqa: E402
from src.line2.data_pipeline.projections import (  # noqa: E402
    N_BINS, six_projections, station_edges,
)
from src.line2.data_pipeline.transport import transport_beam  # noqa: E402

OUT = Path(__file__).resolve().parent
ANCHOR = ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
DS_H5 = ROOT / "studies/line2_particle_sr/data/line2_ds_v1/test.h5"
PAD = 1.01
K_SIGMA = 9.0  # V2c 公共窗口半宽（σ 单位；先测最大 |u| 再定，见 assert）


def pooled_edges(tables: list[np.ndarray]) -> np.ndarray:
    lo = np.min([t.min(axis=0) for t in tables], axis=0)
    hi = np.max([t.max(axis=0) for t in tables], axis=0)
    pad = (hi - lo) * (PAD - 1.0)
    return np.stack([np.linspace(lo[i] - pad[i], hi[i] + pad[i], N_BINS + 1) for i in range(4)])


def main() -> int:
    t0 = time.time()
    anchor = read_anchor_particles(ANCHOR)
    with h5py.File(DS_H5, "r") as f:
        beams = sorted(int(k.split("_")[1]) for k in f.keys())
    print(f"[regen] {len(beams)} test 束，锚点 {anchor.shape}", flush=True)

    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n, b in enumerate(beams):
        s0, s1 = transport_beam(anchor, beam_idx=b)
        tables[b] = (s0[:, :4], s1[:, :4])
        if (n + 1) % 50 == 0:
            print(f"[regen] 传输 {n + 1}/{len(beams)}  ({time.time() - t0:.1f}s)", flush=True)

    stats = {0: [], 1: []}          # 逐束 (mean(4), std(4), min(4), max(4))
    for b in beams:
        for st in (0, 1):
            t = tables[b][st]
            stats[st].append(np.concatenate([t.mean(axis=0), t.std(axis=0, ddof=1),
                                             t.min(axis=0), t.max(axis=0)]))
    stats = {st: np.stack(stats[st]) for st in (0, 1)}
    u_max = float(np.max([np.abs((tables[b][st] - stats[st][n, :4]) / stats[st][n, 4:8]).max()
                          for n, b in enumerate(beams) for st in (0, 1)]))
    assert u_max < K_SIGMA, f"V2c 窗口 {K_SIGMA}σ 不足以覆盖全部粒子（max|u|={u_max}）"

    pooled = {st: pooled_edges([tables[b][st] for b in beams]) for st in (0, 1)}
    centered = {st: [tables[b][st] - stats[st][n, :4] for n, b in enumerate(beams)]
                for st in (0, 1)}
    cen_pooled = {st: pooled_edges(centered[st]) for st in (0, 1)}
    sigma_edges = np.stack([np.linspace(-K_SIGMA, K_SIGMA, N_BINS + 1)] * 4)

    imgs = {v: {0: [], 1: []} for v in ("V1", "V2", "V2b", "V2c")}
    covs = {0: [], 1: []}
    means = {0: [], 1: []}
    prod_edges = {0: [], 1: []}
    for n, b in enumerate(beams):
        for st in (0, 1):
            tbl = tables[b][st]
            e = station_edges(tbl)
            prod_edges[st].append(e)
            imgs["V1"][st].append(six_projections(tbl, e))
            imgs["V2"][st].append(six_projections(tbl, pooled[st]))
            imgs["V2b"][st].append(six_projections(centered[st][n], cen_pooled[st]))
            u = (tbl - stats[st][n, :4]) / stats[st][n, 4:8]
            imgs["V2c"][st].append(six_projections(u, sigma_edges))
            covs[st].append(np.cov(tbl.T, ddof=1))
            means[st].append(tbl.mean(axis=0))
    payload = {"beams": np.array(beams),
               "pooled_edges": np.stack([pooled[0], pooled[1]]),
               "cen_pooled_edges": np.stack([cen_pooled[0], cen_pooled[1]]),
               "sigma_edges": sigma_edges,
               "prod_edges": np.stack([np.stack(prod_edges[0]), np.stack(prod_edges[1])]),
               "covs": np.stack([np.stack(covs[0]), np.stack(covs[1])]),
               "means": np.stack([np.stack(means[0]), np.stack(means[1])]),
               "stats": np.stack([stats[0], stats[1]])}
    for v in imgs:
        for st in (0, 1):
            payload[f"{v}_s{st}"] = np.stack(imgs[v][st])
    np.savez_compressed(OUT / "pre1_regenerated.npz", **payload)

    repro = {}
    with h5py.File(DS_H5, "r") as f:
        for st in (0, 1):
            dmax, e_ok = 0, True
            for n, b in enumerate(beams):
                ds = f[f"beam_{b:04d}"][f"station{st}"]["target_full"]
                ref = ds[:].astype(np.int64)
                dmax = max(dmax, int(np.abs(ref - imgs["V1"][st][n].astype(np.int64)).max()))
                e_ok = e_ok and bool(np.array_equal(ds.attrs["bin_edges"], prod_edges[st][n]))
            repro[f"station{st}"] = {"max_abs_count_diff": dmax, "edges_bitwise_equal": e_ok}
    summary = {
        "n_beams": len(beams), "elapsed_s": round(time.time() - t0, 1),
        "h5_reproduction": repro,
        "max_sigma_units_u": u_max, "v2c_window_sigma": K_SIGMA,
    }
    (OUT / "pre1_regen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
