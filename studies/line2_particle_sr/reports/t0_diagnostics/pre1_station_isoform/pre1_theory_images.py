"""前置① 理论→观测的桥：以「同协方差的高斯锚点 + 生产同一套映射」复算两站图组的逐对 M1。

目的：把理论层的形状参数收缩翻译成 M1 单位，供预注册判据「理论收缩比 vs 观测 5.5× 」直接对表。
构造：高斯锚点（协方差 = 真实锚点 4×4 横向 Σ，SeedSequence(20260913) 派生）→ 生产
`transport_beam` 同一套抖动映射与 FODO → 生产 `station_edges` 自定标 → `six_projections`
→ 逐对 `group_rel_rmse`（与 `pre1_analyze.pairwise_m1` 同式）。

只读 src/、data/；只写本目录。CPU only，随机性经 SeedSequence(20260913) 派生。
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
from src.line2.data_pipeline.projections import six_projections, station_edges  # noqa: E402
from src.line2.data_pipeline.transport import transport_beam  # noqa: E402

OUT = Path(__file__).resolve().parent
ANCHOR = ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
DS_H5 = ROOT / "studies/line2_particle_sr/data/line2_ds_v1/test.h5"
MASTER_SEED = 20260913


def pairwise_m1(imgs: np.ndarray) -> np.ndarray:
    n, nproj, nb, _ = imgs.shape
    flat = imgs.reshape(n, nproj, nb * nb).astype(np.float64)
    mean_r = flat.sum(axis=2) / (nb * nb)
    gram = np.einsum("ipb,jpb->ijp", flat, flat)
    sq = np.einsum("ipb,ipb->ip", flat, flat)
    d2 = sq[:, None, :] + sq[None, :, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    return np.mean(np.sqrt(d2 / (nb * nb)) / mean_r[None, :, :], axis=2)


def main() -> int:
    t0 = time.time()
    real = read_anchor_particles(ANCHOR)
    sigma = np.cov(real[:, :4].T, ddof=1)
    n = real.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(97)[96])
    gauss = rng.multivariate_normal(np.zeros(4), sigma, size=n)
    synth = np.zeros((n, 6))
    synth[:, :4] = gauss
    synth[:, 4] = real[:, 4]
    synth[:, 5] = real[:, 5]          # 纵向列不参与六投影
    print(f"[theory-img] 高斯锚点已构造（Σ 同真实锚点，max|ΔΣ|={np.abs(np.cov(synth[:, :4].T, ddof=1) - sigma).max():.3e}）",
          flush=True)

    with h5py.File(DS_H5, "r") as f:
        beams = sorted(int(k.split("_")[1]) for k in f.keys())

    imgs = {0: [], 1: []}
    for i, b in enumerate(beams):
        s0, s1 = transport_beam(synth, beam_idx=b)
        for st, tbl in ((0, s0), (1, s1)):
            imgs[st].append(six_projections(tbl, station_edges(tbl)))
        if (i + 1) % 50 == 0:
            print(f"[theory-img] {i + 1}/{len(beams)}  ({time.time() - t0:.1f}s)", flush=True)

    m1 = {}
    for st in (0, 1):
        x = np.stack(imgs[st])
        mat = pairwise_m1(x)
        iu = np.triu_indices(len(beams), 1)
        m1[f"station{st}"] = {"median": float(np.median(mat[iu])), "mean": float(mat[iu].mean())}
        np.save(OUT / f"pre1_theory_pairwise_m1_s{st}.npy", mat)
        del x
    m1["ratio_s1_over_s0"] = float(m1["station1"]["median"] / m1["station0"]["median"])

    real_json = json.loads((OUT / "pre1_analysis.json").read_text(encoding="utf-8"))
    obs = real_json["pairwise_M1"]["V1_s0"]["median"], real_json["pairwise_M1"]["V1_s1"]["median"]
    m1["observed_V1"] = {"station0": obs[0], "station1": obs[1], "ratio_s1_over_s0": obs[1] / obs[0]}
    m1["ratio_theory_over_observed"] = float(m1["ratio_s1_over_s0"] / (obs[1] / obs[0]))
    (OUT / "pre1_theory_images.json").write_text(
        json.dumps(m1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m1, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
