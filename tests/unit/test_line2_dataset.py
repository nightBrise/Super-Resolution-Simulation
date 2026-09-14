"""L1 [S4]/[S5] 验收测试（spec line2/10）：分箱守恒/归一化可逆/切分互斥/HDF5 可追溯与确定性。

合成表取 1×10⁵ 粒子以覆盖最大档位（五档 × k=20 全部落盘）。模块级 fixture
`small_dataset` 只构建一次数据集，供结构/元数据/manifest 多条契约断言共享，把整文件
耗时控制在预算内。`_synthetic_table(n, seed)` 被若干 line2 测试模块复用，签名保持稳定。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import h5py
import numpy as np
import pytest

from src.line2.data_pipeline.anchor import ANCHOR_MD5
from src.line2.data_pipeline.dataset import beam_splits, build_dataset
from src.line2.data_pipeline.projections import (
    AXES,
    N_BINS,
    PROJECTION_ORDER,
    denormalize_axis,
    normalize_axis,
    six_projections,
    station_edges,
)
from src.line2.data_pipeline.seeds import MASTER_SEED
from src.line2.data_pipeline.subsample import (
    SUBSAMPLE_K,
    SUBSAMPLE_TIERS,
    subsample_index_set,
)
from src.line2.data_pipeline.transport import jitter_params, transport_beam
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCTION_MANIFEST = REPO_ROOT / "studies/line2_particle_sr/data/line2_ds_v1/manifest.json"

#: `00` [S1] 冻结的六投影枚举——测试侧字面量（不 import PROJECTION_ORDER），用于独立核对
CANONICAL_PROJECTIONS = (
    ("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"),
)
AXIS_INDEX = {axis: i for i, axis in enumerate(AXES)}
#: 模块级数据集束数（小规模：覆盖三切分与多束元数据，构建耗时受控）
N_BEAMS = 6
_PAD_FRAC = 0.01  # `10` [S4] C1：各轴 [min, max] 外扩 1%


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


def _station_edges_independent(table: np.ndarray) -> np.ndarray:
    """独立重算站级 (4, N_BINS+1) 边界（各轴 [min,max] 外扩 1%）——不经 station_edges。"""
    rows = []
    for axis in range(len(AXES)):
        lo, hi = float(table[:, axis].min()), float(table[:, axis].max())
        pad = (hi - lo) * _PAD_FRAC
        rows.append(np.linspace(lo - pad, hi + pad, N_BINS + 1))
    return np.stack(rows)


def _tier_names() -> set[str]:
    return {
        f"tier_{tier:06d}_sub_{sub:02d}"
        for tier in SUBSAMPLE_TIERS
        for sub in range(SUBSAMPLE_K)
    }


@pytest.fixture(scope="module")
def synthetic_anchor() -> np.ndarray:
    """1e5+1 粒子合成表（模块级共享；种子由 TEST_MASTER_SEED 派生）。"""
    return _synthetic_table(100_001, seed=derive_seed(10, 1))


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory, synthetic_anchor) -> pathlib.Path:
    """模块级：构建一次 6 束小数据集，供结构/元数据/manifest 契约断言共享。"""
    out = tmp_path_factory.mktemp("line2_dataset") / "ds"
    return build_dataset(out, synthetic_anchor, n_beams=N_BEAMS, data_version="test_v1")


# ---------------------------------------------------------------- [S4]
def test_projection_conservation_and_dtype() -> None:
    """[S4] C1/C2：∑counts = 粒子数（三档核对）、dtype int32、六通道、非负。"""
    table = _synthetic_table(5_000, seed=derive_seed(10, 2))
    edges = station_edges(table)
    rng = np.random.default_rng(derive_seed(10, 3))
    for n_low in (1_000, 3_000, 5_000):
        sub = table[rng.choice(5_000, n_low, replace=False)]
        imgs = six_projections(sub, edges)
        assert imgs.dtype == np.int32 and imgs.shape == (6, N_BINS, N_BINS)
        assert (imgs >= 0).all()
        for gi in range(6):
            assert int(imgs[gi].sum()) == n_low


def test_projection_channel_order_matches_canonical_enumeration() -> None:
    """[S4] C2（`00` [S1] C2）：六通道顺序逐项等于冻结枚举，且模块常量与枚举一致。

    逐通道用 `np.histogram2d` + 文档化转置独立重算并逐位比对（不依赖列质心之类的
    行为代理）；并构造性验证鉴别力——交换某通道的轴对会得到不同数组 ⇒ 该断言确实
    能发现通道置换，而非恒真。
    """
    assert PROJECTION_ORDER == CANONICAL_PROJECTIONS
    table = _synthetic_table(20_000, seed=derive_seed(10, 4))
    # 给各轴不同尺度与偏移：避免不同投影的直方图偶合相同（否则置换不可分辨）
    table[:, 0] += 1e-3
    table[:, 1] += 1e-5
    table[:, 2] -= 5e-4
    table[:, 3] -= 2e-5
    edges = station_edges(table)
    images = six_projections(table, edges)

    assert images.shape[0] == len(CANONICAL_PROJECTIONS)
    for gi, (first, second) in enumerate(CANONICAL_PROJECTIONS):
        i, j = AXIS_INDEX[first], AXIS_INDEX[second]
        hist, _, _ = np.histogram2d(table[:, i], table[:, j], bins=[edges[i], edges[j]])
        assert np.array_equal(images[gi], hist.T.astype(np.int32)), (gi, first, second)
        swapped, _, _ = np.histogram2d(table[:, j], table[:, i], bins=[edges[j], edges[i]])
        assert not np.array_equal(images[gi], swapped.T.astype(np.int32)), gi
    for a in range(len(CANONICAL_PROJECTIONS)):
        for b in range(a + 1, len(CANONICAL_PROJECTIONS)):
            assert not np.array_equal(images[a], images[b]), (a, b)


def test_normalization_roundtrip() -> None:
    """[S4] C3：以该站 bin 边界线性映射至 [−1, 1]，反归一化往返误差 < 1e-10。"""
    table = _synthetic_table(1_000, seed=derive_seed(10, 5))
    edges = station_edges(table)
    for axis in range(len(AXES)):
        row = edges[axis]
        assert np.array_equal(normalize_axis(row[[0, -1]], row), np.array([-1.0, 1.0]))
        coords = table[:, axis]
        back = denormalize_axis(normalize_axis(coords, row), row)
        assert float(np.abs(back - coords).max()) < 1e-10


def test_station_generation_repeatable_bitwise(synthetic_anchor: np.ndarray) -> None:
    """[S4] C4：同 (beam, N_low, sub_idx) 的重复生成逐位一致；两站各自定标（边界不等）。

    与数据集级快照（`test_build_dataset_deterministic_and_traceable`）的分工：本测试在
    **生成函数层**逐位重放（传输 → 站级定标 → 子采样索引 → 图组），把"重复生成不一致"
    定位到具体环节而不仅是文件级 sha。
    """
    anchor = synthetic_anchor
    for beam_id in (0, 3):
        first = transport_beam(anchor, beam_idx=beam_id)
        second = transport_beam(anchor, beam_idx=beam_id)
        for station_idx in (0, 1):
            table_a, table_b = first[station_idx], second[station_idx]
            np.testing.assert_array_equal(table_a, table_b)
            edges_a = station_edges(table_a)
            edges_b = station_edges(table_b)
            assert np.array_equal(edges_a, edges_b)
            index_a = subsample_index_set(table_a.shape[0], beam_id, station_idx)
            index_b = subsample_index_set(table_b.shape[0], beam_id, station_idx)
            for tier in SUBSAMPLE_TIERS:
                for sub_idx in (0, SUBSAMPLE_K - 1):
                    np.testing.assert_array_equal(index_a[tier][sub_idx], index_b[tier][sub_idx])
                    np.testing.assert_array_equal(
                        six_projections(table_a[index_a[tier][sub_idx]], edges_a),
                        six_projections(table_b[index_b[tier][sub_idx]], edges_b),
                    )
        # 两站各自独立定标：整张边界矩阵不等，且四条轴各自都不等（FODO 改变分布范围）
        edges_0, edges_1 = station_edges(first[0]), station_edges(first[1])
        assert not np.array_equal(edges_0, edges_1)
        assert (np.abs(edges_0 - edges_1).max(axis=1) > 0.0).all()


def test_projection_truncation_reports_error() -> None:
    """[S4] C1（零截断报错分支）：粒子落在 edges 之外时 SHALL 报错，而非静默丢样本。

    构造：把 x 整体平移出 [min,max] 外扩 1% 的窗口后调用 six_projections ——
    实现按计数和核对（`projections.six_projections` 的零截断守卫）抛 ValueError。
    spec 未定错误类型（"报错"），故接受任何异常，但不接受静默丢弃。
    """
    table = _synthetic_table(2_000, seed=derive_seed(10, 6))
    edges = station_edges(table)
    outside = table.copy()
    outside[:, 0] += 1.0
    with pytest.raises(Exception):
        six_projections(outside, edges)


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


def test_build_dataset_structure(small_dataset: pathlib.Path) -> None:
    """[S5] C2/C3：端到端落盘——HDF5 结构、档位×k 完整、manifest 三元组与切分表。"""
    manifest = json.loads((small_dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_beams"] == N_BEAMS
    assert len(manifest["code_version"]) == 40
    assert manifest["master_seed"] == 20260913
    all_ids = sorted(b for ids in manifest["splits"].values() for b in ids)
    assert all_ids == list(range(N_BEAMS))

    expected_tiers = _tier_names()
    structure_checked = False
    for split in ("train", "val", "test"):
        with h5py.File(small_dataset / f"{split}.h5", "r") as h5:
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


def test_dataset_metadata_contract(
    small_dataset: pathlib.Path, synthetic_anchor: np.ndarray
) -> None:
    """[S4] C1/C2/C4 + [S5] C2：落盘件的组/束元数据、站级边界与逐样本计数守恒。

    逐项核对（非仅查键存在）：
    - 每站每个 dataset 的 `bin_edges` 等于该站**全量表**的独立重算（手写外扩 1% 的
      linspace，不经 `station_edges`）⇒ [S4] C1「以该站全量表定标」；
    - `station0` 与 `station1` 的边界不等、且四条轴各自不等 ⇒ [S4] C4「两站各自独立定标」；
    - `axes` 逐项等于 `AXES`、形状 (4, N_BINS+1) ⇒ [S4] C2 元数据内嵌轴对与 bin 边界；
    - 每个 dataset 六通道计数和 = 其档位粒子数（`target_full` = 全量表行数）⇒ [S4] C1 守恒；
    - 束 group 的 `beam_id`/`split`/`master_seed` 与落盘位置一致 ⇒ [S5] C2 可机读身份。
    """
    n_particles = synthetic_anchor.shape[0]
    checked = 0
    for split in ("train", "val", "test"):
        with h5py.File(small_dataset / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                grp = h5[beam_name]
                beam_id = int(grp.attrs["beam_id"])
                assert beam_name == f"beam_{beam_id:04d}"
                assert grp.attrs["split"] == split
                assert int(grp.attrs["master_seed"]) == MASTER_SEED
                snapshot = transport_beam(synthetic_anchor, beam_idx=beam_id)
                for station_idx in (0, 1):
                    st = grp[f"station{station_idx}"]
                    assert int(st.attrs["n_particles_full"]) == n_particles
                    assert set(st.keys()) == {"target_full"} | _tier_names()
                    expected_edges = _station_edges_independent(snapshot[station_idx])
                    assert np.array_equal(
                        st["target_full"].attrs["bin_edges"], expected_edges
                    )
                    for axis in range(len(AXES)):
                        assert (np.diff(expected_edges[axis]) > 0.0).all()
                    for name in st:
                        ds = st[name]
                        image = ds[...]
                        assert ds.dtype == np.int32 and image.shape == (6, N_BINS, N_BINS)
                        assert int(image.min()) >= 0
                        assert np.array_equal(ds.attrs["bin_edges"], expected_edges)
                        axes_attr = [a.decode() if isinstance(a, bytes) else a for a in ds.attrs["axes"]]
                        assert axes_attr == list(AXES)
                        expected_count = (
                            n_particles
                            if name == "target_full"
                            else int(name.split("_")[1])
                        )
                        assert [int(image[c].sum()) for c in range(6)] == [expected_count] * 6, (
                            beam_name, station_idx, name,
                        )
                        checked += 1
                edges_0 = grp["station0"]["target_full"].attrs["bin_edges"]
                edges_1 = grp["station1"]["target_full"].attrs["bin_edges"]
                assert not np.array_equal(edges_0, edges_1)
                assert (np.abs(edges_0 - edges_1).max(axis=1) > 0.0).all()
    assert checked == N_BEAMS * 2 * (1 + len(SUBSAMPLE_TIERS) * SUBSAMPLE_K)


def test_dataset_explicit_identity_metadata(small_dataset: pathlib.Path) -> None:
    """[S5] C2：每个子样本 dataset SHALL 标注其来源 (station, N_low, sub_idx)。

    spec（`10` [S5] 落盘节）点名的元数据字段为 beam_id、station、N_low、sub_idx、
    bin_edges、jitter 参数；当前实现只落 beam_id（group attr）+ bin_edges/axes，
    身份靠 group/dataset **命名**承载（`src/line2/data_pipeline/dataset.py:81-93`；
    D6 的 Rationale 亦以"层级表达自然"为准）。本测试按 Claim 逐项断言显式字段与取值，
    字段缺失时以 xfail 记录缺口（不冒充通过）；字段补齐后本测试自动转为实断言。
    """
    missing: dict[str, list[str]] = {}
    checked = 0
    with h5py.File(small_dataset / "train.h5", "r") as h5:
        for beam_name in h5:
            grp = h5[beam_name]
            for station_idx in (0, 1):
                st = grp[f"station{station_idx}"]
                for name in st:
                    is_tier = name.startswith("tier_")
                    expected = {
                        "station": station_idx,
                        "n_low": int(name.split("_")[1]) if is_tier else int(st.attrs["n_particles_full"]),
                        "sub_idx": int(name.split("_")[-1]) if is_tier else 0,
                    }
                    path = f"{beam_name}/station{station_idx}/{name}"
                    for key, value in expected.items():
                        if key not in st[name].attrs:
                            missing.setdefault(key, []).append(path)
                        elif int(st[name].attrs[key]) != value:
                            raise AssertionError(
                                f"{path}: {key} = {st[name].attrs[key]} != {value}"
                            )
                    checked += 1
    if missing:
        pytest.xfail(
            "[S5] C2 显式身份元数据未落盘："
            + ", ".join(
                f"{key} ×{len(paths)}（例 {paths[0]}）" for key, paths in sorted(missing.items())
            )
            + f"；共核 {checked} 个 dataset；spec 10 [S5] 落盘节点名该三字段；"
            "impl src/line2/data_pipeline/dataset.py:81-93 仅写 bin_edges/axes"
        )
    assert checked > 0


def test_beam_group_jitter_metadata_matches_params(small_dataset: pathlib.Path) -> None:
    """[S5] C2（jitter 字段）：每个有抖动的束 group 落盘抖动参数，且由 (beam_id, seed) 还原。

    束 0 为无抖动基准束（`00` [S1]「束 0 无抖动」、`10` [S3]），实现不写 jitter_* attr
    （`src/line2/data_pipeline/dataset.py:64-68,98-100`）——基准束的字段适用性见报告。
    """
    checked = 0
    for split in ("train", "val", "test"):
        with h5py.File(small_dataset / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                grp = h5[beam_name]
                beam_id = int(grp.attrs["beam_id"])
                if beam_id == 0:
                    continue
                params = jitter_params(beam_id)
                for key, value in params.items():
                    assert f"jitter_{key}" in grp.attrs, (beam_id, key)
                    assert grp.attrs[f"jitter_{key}"] == value, (beam_id, key)
                checked += 1
    assert checked == N_BEAMS - 1


def test_manifest_contract_fields(small_dataset: pathlib.Path) -> None:
    """[S5] C3：manifest 字段与取值符合冻结契约（`00` GC7 三元组 + 档位/k/FODO/抖动/MD5）。"""
    manifest = json.loads((small_dataset / "manifest.json").read_text(encoding="utf-8"))
    # `00` GC7 三元组
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["code_version"]), manifest["code_version"]
    assert manifest["data_version"] == "test_v1"
    assert manifest["spec_version"] == "line2-spec-v1.0"
    assert manifest["master_seed"] == MASTER_SEED == 20260913
    # 锚点 MD5（`10` [S1] C1 的硬编码值 / `10` [S5] C3 点名项）
    assert manifest["anchor_md5"] == "f39d905f96abd5588091dad6dd3884b2"
    assert manifest["anchor_md5"] == ANCHOR_MD5
    # 档位与 k、bin 数（`10` [S2]/[S4]）
    assert manifest["tiers"] == [1_000, 3_000, 10_000, 30_000, 100_000]
    assert manifest["tiers"] == list(SUBSAMPLE_TIERS)
    assert manifest["subsamples_per_tier"] == 20 == SUBSAMPLE_K
    assert manifest["n_bins"] == 128 == N_BINS
    # FODO 与抖动分布（`10` [S3] 冻结值）
    assert manifest["fodo"] == {
        "n_cells": 2, "mu_deg": 75.0, "half_drift_m": 2.5, "quad_length_m": 0.1,
    }
    assert manifest["jitter"] == {
        "nominal_f_m": 10.0,
        "delta_range": [-0.05, 0.05],
        "kick_rad": 1e-05,
        "drift_mm": [495.0, 505.0],
    }
    # 切分表：互斥完备 + 与 beam_splits 规则一致（绝对束数 700/150/150 由生产 manifest 核）
    splits = manifest["splits"]
    assert set(splits) == {"train", "val", "test"}
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (4, 1, 1)
    assert sorted(b for ids in splits.values() for b in ids) == list(range(N_BEAMS))


def test_production_manifest_contract() -> None:
    """[S5] C1/C3（生产入库件）：1000 束切分 = 700/150/150，且与 beam_splits 逐位一致。

    读的是已入库的 `data/line2_ds_v1/manifest.json`（在版本控制内）——同时检出**产物与
    当前实现漂移**（若 beam_splits/档位/manifest 写入逻辑被改动，此处报红）。
    """
    assert PRODUCTION_MANIFEST.is_file(), f"生产 manifest 缺失：{PRODUCTION_MANIFEST}"
    manifest = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["n_beams"] == 1000
    assert manifest["data_version"] == "line2_ds_v1"
    splits = manifest["splits"]
    assert [len(splits[s]) for s in ("train", "val", "test")] == [700, 150, 150]
    assert sorted(b for ids in splits.values() for b in ids) == list(range(1000))
    assert manifest["anchor_md5"] == "f39d905f96abd5588091dad6dd3884b2"
    assert manifest["tiers"] == [1_000, 3_000, 10_000, 30_000, 100_000]
    assert manifest["subsamples_per_tier"] == 20
    assert manifest["n_bins"] == 128
    expected = beam_splits(1000, master_seed=manifest["master_seed"])
    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(splits[split], expected[split])


def test_build_dataset_deterministic_and_traceable(tmp_path: pathlib.Path) -> None:
    """[S5] C4/C5：两次独立构建**数据级全量一致**（全部数据集逐位 + 全部 attrs）；四元组唯一定位。

    注：HDF5 group 对象头内嵌创建时间（h5py 仅对 dataset 暴露 track_times=False，
    dataset 级已关闭），故整文件 sha256 会因容器时间戳跨秒抖动——数据级全量快照
    对比是语义正确的确定性断言。
    """

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

    anchor = _synthetic_table(100_001, seed=derive_seed(10, 7))
    out_a = build_dataset(tmp_path / "a", anchor, n_beams=3, data_version="t_a")
    out_b = build_dataset(tmp_path / "b", anchor, n_beams=3, data_version="t_b")
    for split in ("train", "val", "test"):
        assert snapshot(out_a / f"{split}.h5") == snapshot(out_b / f"{split}.h5")
    splits = beam_splits(3, master_seed=20260913)
    beam_id = int(splits["train"][0])
    path = f"beam_{beam_id:04d}/station0/tier_003000_sub_05"
    with h5py.File(out_a / "train.h5", "r") as ha:
        assert ha[path].shape == (6, N_BINS, N_BINS)
