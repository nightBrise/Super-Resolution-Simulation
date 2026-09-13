"""L1 [S5] 束级切分与数据集落盘（spec line2/10 [S5]）。

- 切分：SeedSequence(master).spawn(SPLIT_BRANCH) 置换束 id，train/val/test = 70/15/15；
- 落盘：每切分一个 HDF5（gzip 4，int32），束 group → 站 subgroup →
  ``target_full`` + 五档 × k=20 子样本数据集，attr 含 bin_edges/jitter/溯源元数据；
- 端到端确定性：同 master_seed 全链重跑逐位一致（`00` C3）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from .projections import AXES, N_BINS, six_projections, station_edges
from .seeds import MASTER_SEED, SPLIT_BRANCH
from .subsample import SUBSAMPLE_K, SUBSAMPLE_TIERS, subsample_index_set
from .transport import (
    JITTER_DELTA_RANGE,
    JITTER_DRIFT_MM,
    JITTER_KICK_RAD,
    JITTER_NOMINAL_F,
    FODO_MU_DEG,
    FODO_N_CELLS,
    FODO_HALF_DRIFT_M,
    QUAD_LENGTH_M,
    transport_beam,
)

__all__ = ["beam_splits", "build_dataset", "git_head"]


def git_head() -> str:
    """git HEAD 完整 40 位 hash（GC7）。"""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def beam_splits(n_beams: int, master_seed: int = MASTER_SEED) -> dict[str, np.ndarray]:
    """束级切分（GC1）：互斥、并集完备、可复现。"""
    ss = np.random.SeedSequence(master_seed).spawn(SPLIT_BRANCH + 1)[SPLIT_BRANCH]
    perm = np.random.default_rng(ss).permutation(n_beams)
    n_train = int(round(0.70 * n_beams))
    n_val = int(round(0.15 * n_beams))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def _write_beam_group(
    h5: h5py.File,
    beam_idx: int,
    split: str,
    anchor_table: np.ndarray,
) -> None:
    s0, s1 = transport_beam(anchor_table, beam_idx=beam_idx)
    grp = h5.create_group(f"beam_{beam_idx:04d}")
    jitter = None
    if beam_idx != 0:
        from .transport import jitter_params

        jitter = jitter_params(beam_idx)
    for station_idx, table in ((0, s0), (1, s1)):
        st = grp.create_group(f"station{station_idx}")
        edges = station_edges(table)  # 该站全量表定标（[S4] C1）
        target = six_projections(table, edges)
        ds = st.create_dataset(
            "target_full", data=target, compression="gzip", compression_opts=4,
            track_times=False,  # 逐字节确定性（[S5] C4 全文件 sha256 断言）
        )
        subs = subsample_index_set(table.shape[0], beam_idx, station_idx)
        for tier in SUBSAMPLE_TIERS:
            for sub_idx, index in enumerate(subs[tier]):
                img = six_projections(table[index], edges)
                dset = st.create_dataset(
                    f"tier_{tier:06d}_sub_{sub_idx:02d}",
                    data=img,
                    compression="gzip",
                    compression_opts=4,
                    track_times=False,
                )
                dset.attrs["bin_edges"] = edges
                dset.attrs["axes"] = np.array(AXES, dtype=h5py.string_dtype())
                dset.attrs["bin_edges"] = edges
                dset.attrs["axes"] = np.array(AXES, dtype=h5py.string_dtype())
        ds.attrs["bin_edges"] = edges
        ds.attrs["axes"] = np.array(AXES, dtype=h5py.string_dtype())
        st.attrs["n_particles_full"] = table.shape[0]
    grp.attrs["beam_id"] = beam_idx
    grp.attrs["split"] = split
    grp.attrs["master_seed"] = MASTER_SEED
    if jitter is not None:
        for k, v in jitter.items():
            grp.attrs[f"jitter_{k}"] = v


def build_dataset(
    out_dir: str | Path,
    anchor_table: np.ndarray,
    n_beams: int = 1000,
    master_seed: int = MASTER_SEED,
    data_version: str = "line2_ds_v1",
    print_every: int = 50,
) -> Path:
    """全链数据集构建（生产入口；单元测试用小 n_beams）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if anchor_table.shape[0] < max(SUBSAMPLE_TIERS):
        raise ValueError(
            f"锚点表行数 {anchor_table.shape[0]} 不足以支撑最大档位 {max(SUBSAMPLE_TIERS)}"
        )
    splits = beam_splits(n_beams, master_seed)
    split_of = {int(b): s for s, ids in splits.items() for b in ids}

    files = {s: out_dir / f"{s}.h5" for s in splits}
    h5s = {s: h5py.File(f, "w") for s, f in files.items()}
    try:
        for beam_idx in range(n_beams):
            split = split_of[beam_idx]
            _write_beam_group(h5s[split], beam_idx, split, anchor_table)
            if (beam_idx + 1) % print_every == 0 or beam_idx + 1 == n_beams:
                print(f"[build_dataset] {beam_idx + 1}/{n_beams} beams", flush=True)
    finally:
        for h in h5s.values():
            h.close()

    manifest = {
        "data_version": data_version,
        "code_version": git_head(),
        "spec_version": "line2-spec-v1.0",
        "master_seed": int(master_seed),
        "n_beams": int(n_beams),
        "splits": {s: [int(b) for b in ids] for s, ids in splits.items()},
        "tiers": list(SUBSAMPLE_TIERS),
        "subsamples_per_tier": SUBSAMPLE_K,
        "fodo": {
            "n_cells": FODO_N_CELLS,
            "mu_deg": FODO_MU_DEG,
            "half_drift_m": FODO_HALF_DRIFT_M,
            "quad_length_m": QUAD_LENGTH_M,
        },
        "jitter": {
            "nominal_f_m": JITTER_NOMINAL_F,
            "delta_range": list(JITTER_DELTA_RANGE),
            "kick_rad": JITTER_KICK_RAD,
            "drift_mm": list(JITTER_DRIFT_MM),
        },
        "anchor_md5": "f39d905f96abd5588091dad6dd3884b2",
        "n_bins": N_BINS,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_dir
