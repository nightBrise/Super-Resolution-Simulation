"""L1 [S5] 束级切分与数据集落盘（spec line2/10 [S5]）。

- 切分：SeedSequence(master).spawn(SPLIT_BRANCH) 置换束 id，train/val/test = 70/15/15；
- 落盘：每切分一个 HDF5（gzip 4，int32），束 group → 站 subgroup →
  ``target_full`` + 五档 × k=20 子样本数据集，attr 含 bin_edges/jitter/溯源元数据；
- 端到端确定性：同 master_seed 全链重跑逐位一致（`00` C3）。

两个分箱口径（`50` [S2] D8 只改边、不改粒子）：

- **v1（默认，``data_version`` 非 line2_ds_v2）**：站级边界由该站全量表逐束自定标
  （`10` [S4] C1，外扩 1%）；
- **v2（``data_version="line2_ds_v2"``）**：同站同轴边界跨束共享，由 train 束池化
  [min, max] 外扩 5% 导出（`50` [S1] 公共域）；域外粒子夹取进最近边界格并逐
  (束, 站) 计账（`50` [S2] C6），全数据集夹取占比在写盘前预检、超上限即中止。

``reconcile_particles_with_v1`` 提供 C1 的重传对账（以 v1 口径重生成图组、与
line2_ds_v1 落盘 target_full 逐束逐站比对，pre1 同法）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from .projections import (
    AXES,
    N_BINS,
    POOLED_PAD_FRAC,
    PROJECTION_ORDER,
    clamp_table,
    count_oob_particles,
    edges_sha256,
    pooled_station_edges,
    six_projections,
    station_edges,
)
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
    jitter_params,
    transport_beam,
)

__all__ = [
    "beam_splits", "build_dataset", "git_head",
    "train_pooled_edges", "reconcile_particles_with_v1",
    "V2_DATA_VERSION", "EDGE_RULE_V2", "MAX_CLAMP_FRAC_V2", "SPEC_VERSION_V2",
]

#: v2 数据集标识（`50` [S2] C1/C5）；仅该值走公共域路径
V2_DATA_VERSION = "line2_ds_v2"
#: manifest 的边层规则版本串（`50` [S1] C6 要求）
EDGE_RULE_V2 = "common_train_pooled_min_max_pad5"
#: manifest 的 spec_version：v1 件保持冻结批次 v1.0；v2 件由 Phase 1.5 批次（99 v1.1）管辖
SPEC_VERSION_V1 = "line2-spec-v1.0"
SPEC_VERSION_V2 = "line2-spec-v1.1"
#: v2 全数据集夹取占比上限（`50` [S2] C6 保险丝；分母 = 各束各站全量表行数之和）
MAX_CLAMP_FRAC_V2 = 0.001
#: 逐 (束, 站) 夹取计数落盘文件名（manifest 内记其 SHA256）
CLAMP_ACCOUNTING_FILE = "clamp_accounting.csv"


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


def train_pooled_edges(
    anchor_table: np.ndarray, train_beams: np.ndarray
) -> dict[int, np.ndarray]:
    """公共域（`50` [S1]）：某 (轴, 站) 的 129 边界 = **train 束**该站全量表池化
    [min, max] 外扩 5%（:data:`~.projections.POOLED_PAD_FRAC`）。

    返回 ``{station_idx: (4, 129)}``。池化取逐轴精确 min/max（无累加误差），故与
    束遍历顺序无关、可复现；val/test 束不参与（`50` Global Constraints 防泄露）。
    """
    lo = np.full((2, len(AXES)), np.inf)
    hi = np.full((2, len(AXES)), -np.inf)
    for beam_idx in train_beams:
        for station_idx, table in enumerate(transport_beam(anchor_table, beam_idx=int(beam_idx))):
            lo[station_idx] = np.minimum(lo[station_idx], table[:, :4].min(axis=0))
            hi[station_idx] = np.maximum(hi[station_idx], table[:, :4].max(axis=0))
    return {
        station_idx: pooled_station_edges(
            np.stack([lo[station_idx], hi[station_idx]], axis=1)
        )
        for station_idx in (0, 1)
    }


def _clamp_precheck(
    anchor_table: np.ndarray, n_beams: int, edges_by_station: dict[int, np.ndarray]
) -> np.ndarray:
    """逐 (束, 站) 夹取**粒子数**预检——写盘前算出全数据集占比（`50` [S2] C6 保险丝）。

    口径 = :func:`~.projections.count_oob_particles`（任一轴越界计一次，`50` D16），
    与写盘路径同口径，供构建末尾的逐 (束, 站) 一致性守卫逐位比对。
    """
    counts = np.zeros((n_beams, 2), dtype=np.int64)
    for beam_idx in range(n_beams):
        for station_idx, table in enumerate(transport_beam(anchor_table, beam_idx=beam_idx)):
            counts[beam_idx, station_idx] = count_oob_particles(
                table, edges_by_station[station_idx]
            )
    return counts


def _write_clamp_accounting(
    out_dir: Path, clamp_counts: np.ndarray, n_particles: int
) -> tuple[int, str]:
    """逐 (束, 站) 夹取**粒子数**落盘为 CSV，返回 (总夹取粒子数, 文件 SHA256)。

    列 ``n_clamped`` 的口径 = 任一轴越界的粒子数（`50` [S2] C6 / D16），与预检、
    h5 站 attr 及 manifest 的 ``clamp`` 段一致。
    """
    lines = ["beam_id,station,n_particles,n_clamped"]
    for beam_idx in range(clamp_counts.shape[0]):
        for station_idx in (0, 1):
            lines.append(
                f"{beam_idx},{station_idx},{n_particles},{int(clamp_counts[beam_idx, station_idx])}"
            )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / CLAMP_ACCOUNTING_FILE).write_bytes(payload)
    return int(clamp_counts.sum()), hashlib.sha256(payload).hexdigest()


def _write_beam_group(
    h5: h5py.File,
    beam_idx: int,
    split: str,
    anchor_table: np.ndarray,
    edges_by_station: dict[int, np.ndarray] | None = None,
    tiers: tuple[int, ...] = SUBSAMPLE_TIERS,
    k: int = SUBSAMPLE_K,
) -> list[tuple[int, int]]:
    """写一个束 group；返回逐站 ``(n_particles, n_clamped)``。

    ``edges_by_station=None`` 走 v1 逐束自定标（该站全量表，[S4] C1）；否则用公共域
    边界并把域外粒子夹取进最近边界格（`50` [S2] C6）。
    """
    s0, s1 = transport_beam(anchor_table, beam_idx=beam_idx)
    grp = h5.create_group(f"beam_{beam_idx:04d}")
    jitter = None
    if beam_idx != 0:
        jitter = jitter_params(beam_idx)
    report: list[tuple[int, int]] = []
    for station_idx, table in ((0, s0), (1, s1)):
        if edges_by_station is None:
            edges = station_edges(table)  # 该站全量表定标（[S4] C1）
            binned_table, n_clamped = table, 0
        else:
            edges = edges_by_station[station_idx]
            binned_table, _ = clamp_table(table, edges)  # 逐轴计数为诊断口径，报告用粒子数
            n_clamped = count_oob_particles(table, edges)
        st = grp.create_group(f"station{station_idx}")
        target = six_projections(binned_table, edges)
        ds = st.create_dataset(
            "target_full", data=target, compression="gzip", compression_opts=4,
            track_times=False,  # 逐字节确定性（[S5] C4 全文件 sha256 断言）
        )
        subs = subsample_index_set(table.shape[0], beam_idx, station_idx, tiers=tiers, k=k)
        for tier in tiers:
            for sub_idx, index in enumerate(subs[tier]):
                img = six_projections(binned_table[index], edges)
                dset = st.create_dataset(
                    f"tier_{tier:06d}_sub_{sub_idx:02d}",
                    data=img,
                    compression="gzip",
                    compression_opts=4,
                    track_times=False,
                )
                dset.attrs["bin_edges"] = edges
                dset.attrs["axes"] = np.array(AXES, dtype=h5py.string_dtype())
        ds.attrs["bin_edges"] = edges
        ds.attrs["axes"] = np.array(AXES, dtype=h5py.string_dtype())
        st.attrs["n_particles_full"] = table.shape[0]
        if edges_by_station is not None:
            st.attrs["n_clamped"] = n_clamped
        report.append((int(table.shape[0]), n_clamped))
    grp.attrs["beam_id"] = beam_idx
    grp.attrs["split"] = split
    grp.attrs["master_seed"] = MASTER_SEED
    if jitter is not None:
        for key, value in jitter.items():
            grp.attrs[f"jitter_{key}"] = value
    return report


def build_dataset(
    out_dir: str | Path,
    anchor_table: np.ndarray,
    n_beams: int = 1000,
    master_seed: int = MASTER_SEED,
    data_version: str = "line2_ds_v1",
    print_every: int = 50,
    tiers: tuple[int, ...] = SUBSAMPLE_TIERS,
    k: int = SUBSAMPLE_K,
    max_clamp_frac: float | None = None,
) -> Path:
    """全链数据集构建（生产入口；单元测试用小 n_beams/tiers）。

    ``data_version="line2_ds_v2"`` 走公共域路径（`50` [S2]）：bin_edges 由 train 束
    池化导出、跨束共享，域外粒子夹取并逐 (束, 站) 计账；写盘前预检全数据集夹取占比
    （上限默认 :data:`MAX_CLAMP_FRAC_V2`，超限即中止且不写盘）。其余 data_version
    保持 v1 逐束自定标路径逐位不变。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if anchor_table.shape[0] < max(tiers):
        raise ValueError(
            f"锚点表行数 {anchor_table.shape[0]} 不足以支撑最大档位 {max(tiers)}"
        )
    splits = beam_splits(n_beams, master_seed)
    split_of = {int(b): s for s, ids in splits.items() for b in ids}

    edges_by_station: dict[int, np.ndarray] | None = None
    clamp_precheck: np.ndarray | None = None
    clamp_limit: float | None = None
    if data_version == V2_DATA_VERSION:
        clamp_limit = MAX_CLAMP_FRAC_V2 if max_clamp_frac is None else float(max_clamp_frac)
        edges_by_station = train_pooled_edges(anchor_table, splits["train"])
        clamp_precheck = _clamp_precheck(anchor_table, n_beams, edges_by_station)
        total_particles = 2 * n_beams * int(anchor_table.shape[0])
        clamp_frac = int(clamp_precheck.sum()) / total_particles
        print(
            f"[build_dataset] v2 公共域夹取预检：{int(clamp_precheck.sum())}/{total_particles}"
            f" = {clamp_frac:.3e}（上限 {clamp_limit:.3e}）",
            flush=True,
        )
        if clamp_frac > clamp_limit:
            raise RuntimeError(
                f"夹取占比 {clamp_frac:.3e} 超过上限 {clamp_limit:.3e}（`50` [S2] C6）——"
                "公共域定标规则或输入束群有误；构建中止（未写盘）"
            )

    files = {s: out_dir / f"{s}.h5" for s in splits}
    h5s = {s: h5py.File(f, "w") for s, f in files.items()}
    clamp_counts = np.zeros((n_beams, 2), dtype=np.int64)
    try:
        for beam_idx in range(n_beams):
            split = split_of[beam_idx]
            report = _write_beam_group(
                h5s[split], beam_idx, split, anchor_table,
                edges_by_station=edges_by_station, tiers=tiers, k=k,
            )
            if edges_by_station is not None:
                clamp_counts[beam_idx] = [n_clamped for _, n_clamped in report]
            if (beam_idx + 1) % print_every == 0 or beam_idx + 1 == n_beams:
                print(f"[build_dataset] {beam_idx + 1}/{n_beams} beams", flush=True)
    finally:
        for h in h5s.values():
            h.close()

    manifest = {
        "data_version": data_version,
        "code_version": git_head(),
        "spec_version": SPEC_VERSION_V2 if edges_by_station is not None else SPEC_VERSION_V1,
        "master_seed": int(master_seed),
        "n_beams": int(n_beams),
        "splits": {s: [int(b) for b in ids] for s, ids in splits.items()},
        "tiers": list(tiers),
        "subsamples_per_tier": k,
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
    if edges_by_station is not None:
        assert clamp_precheck is not None and clamp_limit is not None
        # 预检与写盘同口径（`50` [S2] C6 / D16：任一轴越界的粒子数），逐 (束, 站) 逐位比对
        if not np.array_equal(clamp_precheck, clamp_counts):
            raise RuntimeError("写盘夹取计数与预检不一致（构建非确定性）")
        n_particles = int(anchor_table.shape[0])
        stacked_edges = np.stack([edges_by_station[0], edges_by_station[1]])
        clamped_total, accounting_sha = _write_clamp_accounting(
            out_dir, clamp_counts, n_particles
        )
        manifest["edge_rule"] = EDGE_RULE_V2
        manifest["edge_rule_detail"] = {
            "pad_frac": POOLED_PAD_FRAC,
            "statistic": "pooled_min_max",
            "derived_from": "train",
            "n_train_beams": int(len(splits["train"])),
            "axis_order": list(AXES),
        }
        manifest["edges"] = {
            f"station{station_idx}": {
                axis: [float(v) for v in edges_by_station[station_idx][axis_idx]]
                for axis_idx, axis in enumerate(AXES)
            }
            for station_idx in (0, 1)
        }
        manifest["edges_sha256"] = {
            f"station{station_idx}": {
                axis: edges_sha256(edges_by_station[station_idx][axis_idx])
                for axis_idx, axis in enumerate(AXES)
            }
            for station_idx in (0, 1)
        } | {"combined": edges_sha256(stacked_edges)}
        manifest["projection_axes"] = [list(pair) for pair in PROJECTION_ORDER]
        manifest["clamp"] = {
            "max_frac_allowed": clamp_limit,
            "total_particles": 2 * n_beams * n_particles,
            "clamped_particles": clamped_total,
            "clamped_frac": clamped_total / (2 * n_beams * n_particles),
            "max_per_beam_station": int(clamp_counts.max()),
            "per_beam_file": CLAMP_ACCOUNTING_FILE,
            "per_beam_file_sha256": accounting_sha,
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_dir


def reconcile_particles_with_v1(
    anchor_table: np.ndarray,
    v1_dir: str | Path,
    beams: np.ndarray | None = None,
    progress_every: int = 200,
) -> dict:
    """C1 重传对账：以 v1 逐束定标口径重生成图组，与 ``line2_ds_v1`` 落盘
    ``target_full`` 逐束逐站比对（pre1 同法）。

    粒子表由锚点 + 同一 SeedSequence(20260913) 链重生成；比对量 = 计数最大绝对差
    （int64 逐格）与 bin_edges 逐位相等。``beams=None`` 时覆盖 v1 数据目录内的全部束。
    """
    v1_dir = Path(v1_dir)
    handles: dict[str, h5py.File] = {}
    split_of: dict[int, str] = {}
    try:
        for split in ("train", "val", "test"):
            handles[split] = h5py.File(v1_dir / f"{split}.h5", "r")
            for name in handles[split]:
                split_of[int(name.split("_")[1])] = split
        if beams is None:
            beams = np.array(sorted(split_of), dtype=np.int64)
        stats = {
            station_idx: {"max_abs_count_diff": 0, "n_compared": 0, "edges_bitwise_equal": True}
            for station_idx in (0, 1)
        }
        mismatches: list[dict] = []
        for n, beam_idx in enumerate(beams):
            beam_idx = int(beam_idx)
            grp = handles[split_of[beam_idx]][f"beam_{beam_idx:04d}"]
            for station_idx, table in enumerate(transport_beam(anchor_table, beam_idx=beam_idx)):
                ds = grp[f"station{station_idx}"]["target_full"]
                edges = station_edges(table)
                diff = int(
                    np.abs(ds[...].astype(np.int64) - six_projections(table, edges)).max()
                )
                station_stats = stats[station_idx]
                station_stats["n_compared"] += 1
                station_stats["max_abs_count_diff"] = max(
                    station_stats["max_abs_count_diff"], diff
                )
                if not np.array_equal(ds.attrs["bin_edges"], edges):
                    station_stats["edges_bitwise_equal"] = False
                if diff:
                    mismatches.append(
                        {"beam_id": beam_idx, "station": station_idx, "max_abs_count_diff": diff}
                    )
            if progress_every and (n + 1) % progress_every == 0:
                print(f"[reconcile] {n + 1}/{len(beams)} beams", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "n_beams": int(len(beams)),
        "per_station": {str(station_idx): stats[station_idx] for station_idx in (0, 1)},
        "mismatches": mismatches,
    }
