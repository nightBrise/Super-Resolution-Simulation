"""L1 [S4]/[S5] 验收测试（spec line2/10）：分箱守恒/归一化可逆/切分互斥/HDF5 可追溯与确定性。

合成表取 1×10⁵ 粒子以覆盖最大档位（五档 × k=20 全部落盘）。
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest

from src.line2.data_pipeline.dataset import beam_splits, build_dataset
from src.line2.data_pipeline.projections import (
    N_BINS,
    denormalize_axis,
    normalize_axis,
    six_projections,
    station_edges,
)

pytestmark = pytest.mark.unit


def _synthetic_table(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 2.4e-12, n),
            rng.normal(254.16, 0.27, n),
        ],
        axis=1,
    )


# ---------------------------------------------------------------- [S4]
def test_projection_conservation_and_dtype() -> None:
    """[S4] C1/C2：∑counts = 粒子数（三档核对）、dtype int32、六通道、非负。"""
    table = _synthetic_table(5_000, seed=1)
    edges = station_edges(table)
    rng = np.random.default_rng(0)
    for n_low in (1_000, 3_000, 5_000):
        sub = table[rng.choice(5_000, n_low, replace=False)]
        imgs = six_projections(sub, edges)
        assert imgs.dtype == np.int32 and imgs.shape == (6, N_BINS, N_BINS)
        assert (imgs >= 0).all()
        for gi in range(6):
            assert int(imgs[gi].sum()) == n_low


def test_projection_canonical_order() -> None:
    """[S4] C2：通道顺序可由构造性分布区分（x 平移只影响含 x 通道的列质心）。"""
    table = _synthetic_table(2_000, seed=2)
    table[:, 0] += 1e-3
    edges = station_edges(table)
    imgs = six_projections(table, edges)

    def col_centroid(gi: int) -> float:
        img = imgs[gi].astype(float)
        cols = np.arange(N_BINS)
        return float((img.sum(axis=0) * cols).sum() / img.sum())

    assert col_centroid(0) > col_centroid(1)  # (x,xp) vs (y,yp)
    assert col_centroid(2) > col_centroid(1)  # (x,y) vs (y,yp)


def test_normalization_roundtrip() -> None:
    """[S4] C3：归一化可逆（往返误差 < 1e-10）。"""
    rng = np.random.default_rng(3)
    coords = rng.normal(0.0, 1e-4, 1_000)
    edges = np.linspace(coords.min() * 1.01 - 1e-6, coords.max() * 1.01 + 1e-6, N_BINS + 1)
    u = normalize_axis(coords, edges)
    back = denormalize_axis(u, edges)
    assert float(np.abs(back - coords).max()) < 1e-10


# ---------------------------------------------------------------- [S5]
def test_beam_splits_exclusive_and_complete() -> None:
    """[S5] C1（n_beams=1000）：互斥、并集完备、700/150/150、可复现。"""
    splits = beam_splits(1000, master_seed=20260913)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (700, 150, 150)
    all_ids = np.concatenate(list(splits.values()))
    assert len(np.unique(all_ids)) == 1000
    again = beam_splits(1000, master_seed=20260913)
    for s in splits:
        np.testing.assert_array_equal(splits[s], again[s])


def test_build_dataset_structure(tmp_path: pathlib.Path) -> None:
    """[S5] C2/C3：端到端落盘——HDF5 结构、档位×k 完整、manifest 三元组与切分表。"""
    anchor = _synthetic_table(100_001, seed=42)
    out = build_dataset(tmp_path / "ds", anchor, n_beams=6, data_version="test_v1")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_beams"] == 6
    assert len(manifest["code_version"]) == 40
    assert manifest["master_seed"] == 20260913
    all_ids = sorted(b for ids in manifest["splits"].values() for b in ids)
    assert all_ids == list(range(6))

    expected_tiers = {
        f"tier_{t:06d}_sub_{k:02d}"
        for t in (1_000, 3_000, 10_000, 30_000, 100_000)
        for k in range(20)
    }
    structure_checked = False
    for split in ("train", "val", "test"):
        with h5py.File(out / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                grp = h5[beam_name]
                beam_id = int(grp.attrs["beam_id"])
                assert grp.attrs["split"] == split
                assert manifest["splits"][split].count(beam_id) == 1  # 归属与 manifest 一致
                if not structure_checked:
                    for st in ("station0", "station1"):
                        stg = grp[st]
                        assert set(stg.keys()) == {"target_full"} | expected_tiers
                        ds0 = stg["tier_001000_sub_00"]
                        assert ds0.shape == (6, N_BINS, N_BINS)
                        assert ds0.dtype == np.int32
                        assert all(int(ds0[gi].sum()) == 1_000 for gi in range(6))  # [S4] C1 逐投影守恒
                        assert ds0.attrs["bin_edges"].shape == (4, N_BINS + 1)
                    structure_checked = True
    assert structure_checked


def test_build_dataset_deterministic_and_traceable(tmp_path: pathlib.Path) -> None:
    """[S5] C4/C5：两次独立构建**数据级全量一致**（全部数据集逐位 + 全部 attrs）；四元组唯一定位。

    注：HDF5 group 对象头内嵌创建时间（h5py 仅对 dataset 暴露 track_times=False，
    dataset 级已关闭），故整文件 sha256 会因容器时间戳跨秒抖动——数据级全量快照
    对比是语义正确的确定性断言。
    """
    import hashlib

    def snapshot(path: pathlib.Path) -> dict:
        snap: dict[str, tuple] = {}
        with h5py.File(path, "r") as f:
            def visit(name: str, obj: h5py.HLObject) -> None:
                if isinstance(obj, h5py.Dataset):
                    data_hash = hashlib.sha256(np.ascontiguousarray(obj[...]).tobytes()).hexdigest()
                    attrs = sorted((k, str(obj.attrs[k])) for k in obj.attrs)
                    snap[name] = ("D", data_hash, tuple(attrs))
                else:
                    attrs = sorted((k, str(obj.attrs[k])) for k in obj.attrs)
                    snap[name] = ("G", tuple(attrs))
            f.visititems(visit)
        return snap

    anchor = _synthetic_table(100_001, seed=7)
    out_a = build_dataset(tmp_path / "a", anchor, n_beams=3, data_version="t_a")
    out_b = build_dataset(tmp_path / "b", anchor, n_beams=3, data_version="t_b")
    for split in ("train", "val", "test"):
        assert snapshot(out_a / f"{split}.h5") == snapshot(out_b / f"{split}.h5")
    splits = beam_splits(3, master_seed=20260913)
    beam_id = int(splits["train"][0])
    path = f"beam_{beam_id:04d}/station0/tier_003000_sub_05"
    with h5py.File(out_a / "train.h5", "r") as ha:
        assert ha[path].shape == (6, N_BINS, N_BINS)
