"""line2 Phase 1.5 v2 数据集验收（spec `50` [S2] C1–C6 与 `50` [S1] C6）。

被验对象 = ``data_version="line2_ds_v2"`` 的构建路径：公共域定标（train 束池化 +
5% 外扩，跨束共享）、域外夹取计账、manifest 边层载荷。v1 逐束自定标路径的验收
仍在 ``tests/unit/test_line2_dataset.py``。

规模与代价：模块级 fixture 用小 n_beams/档位构建一次 v2 数据集（契约断言共享）；
生产语料（1000 束、``*.h5`` 不入库）相关的断言在其存在时执行——C1 的全量对账件
由构建 CLI 的 ``--reconcile-only`` 落盘，C2 的跨束一致性读生产 h5 的 attrs。
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import h5py
import numpy as np
import pytest

from src.line2.data_pipeline.anchor import ANCHOR_MD5, read_anchor_particles
from src.line2.data_pipeline.dataset import (
    EDGE_RULE_V2,
    V2_DATA_VERSION,
    beam_splits,
    build_dataset,
    git_head,
    reconcile_particles_with_v1,
)
from src.line2.data_pipeline.projections import (
    AXES,
    N_BINS,
    POOLED_PAD_FRAC,
    clamp_table,
    edges_sha256,
    six_projections,
    station_edges,
)
from src.line2.data_pipeline.subsample import SUBSAMPLE_K, SUBSAMPLE_TIERS
from src.line2.data_pipeline.transport import (
    IDENTITY_JITTER,
    JITTER_NOMINAL_F,
    jitter_params,
    transport_beam,
    transport_with_params,
)
from src.line2.metrics.metrics import group_rel_rmse

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANCHOR_PATH = REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
V1_DIR = REPO_ROOT / "studies/line2_particle_sr/data/line2_ds_v1"
V2_DIR = REPO_ROOT / "studies/line2_particle_sr/data/line2_ds_v2"

#: 模块级小规模构建参数（契约断言共享；覆盖三切分且构建耗时受控）
N_BEAMS = 6
SMALL_TIERS = (1_000, 3_000)
SMALL_K = 2
#: C1 抽样对账束数（每切分各取前若干个）
RECONCILE_PER_SPLIT = 5
#: `00` [S1] 冻结的六投影枚举——测试侧字面量（不 import PROJECTION_ORDER），用于独立核对
CANONICAL_PROJECTIONS = (
    ("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"),
)
#: C3 阈值：保守判 θ 通道"可见"的下限（`50` [S2] C3 / D12；公共域口径实测 ±5 µrad ≈ 0.73–0.86）
M1_FLOOR = 1e-3


def _independent_edges(anchor_table: np.ndarray, beams: np.ndarray) -> dict[int, np.ndarray]:
    """独立重算公共域（手写池化循环 + linspace，不经 ``train_pooled_edges``）。"""
    edges: dict[int, np.ndarray] = {}
    for station_idx in (0, 1):
        lo = np.full(len(AXES), np.inf)
        hi = np.full(len(AXES), -np.inf)
        for beam_idx in beams:
            table = transport_beam(anchor_table, beam_idx=int(beam_idx))[station_idx]
            for axis in range(len(AXES)):
                lo[axis] = min(lo[axis], float(table[:, axis].min()))
                hi[axis] = max(hi[axis], float(table[:, axis].max()))
        edges[station_idx] = np.stack([
            np.linspace(
                lo[axis] - (hi[axis] - lo[axis]) * POOLED_PAD_FRAC,
                hi[axis] + (hi[axis] - lo[axis]) * POOLED_PAD_FRAC,
                N_BINS + 1,
            )
            for axis in range(len(AXES))
        ])
    return edges


def _canonical_digest(edges: np.ndarray) -> str:
    """小端 float64 连续字节的 SHA256——独立于 ``projections.edges_sha256`` 的实现。"""
    return hashlib.sha256(np.asarray(edges, dtype="<f8").tobytes()).hexdigest()


def _manifest_edges(manifest: dict) -> dict[int, np.ndarray]:
    return {
        station_idx: np.stack([
            np.array(manifest["edges"][f"station{station_idx}"][axis], dtype=np.float64)
            for axis in AXES
        ])
        for station_idx in (0, 1)
    }


def _knob_params(**knobs: float) -> dict[str, float]:
    """单旋钮参数：未点名的旋钮取零（δ=0 → f_j = 标称焦距、θ=0、ℓ=0）。"""
    params = {
        "f_j": JITTER_NOMINAL_F * (1.0 + knobs.get("delta", 0.0)),
        "theta_x_rad": knobs.get("theta_x_rad", 0.0),
        "theta_y_rad": knobs.get("theta_y_rad", 0.0),
        "drift_m": knobs.get("drift_mm", 0.0) / 1000.0,
    }
    return params


def _station_images(
    anchor_table: np.ndarray, params: dict[str, float], edges_by_station: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """给定参数与公共域的逐站 (6, 128, 128) 目标图组（域外粒子夹取）。"""
    out: dict[int, np.ndarray] = {}
    for station_idx, table in enumerate(transport_with_params(anchor_table, params)):
        binned, _ = clamp_table(table, edges_by_station[station_idx])
        out[station_idx] = six_projections(binned, edges_by_station[station_idx])
    return out


def _v1_station_images(anchor_table: np.ndarray, params: dict[str, float]) -> dict[int, np.ndarray]:
    """v1 逐束自定标口径的逐站图组（对照用）。"""
    out: dict[int, np.ndarray] = {}
    for station_idx, table in enumerate(transport_with_params(anchor_table, params)):
        edges = station_edges(table)
        out[station_idx] = six_projections(table, edges)
    return out


@pytest.fixture(scope="module")
def anchor() -> np.ndarray:
    """锚点粒子表（200,001×6，入库件；`10` [S1]）。"""
    return read_anchor_particles(ANCHOR_PATH)


@pytest.fixture(scope="module")
def small_v2(tmp_path_factory, anchor: np.ndarray) -> pathlib.Path:
    """模块级：构建一次小规模 v2 数据集（契约断言共享）。

    ``max_clamp_frac=1.0`` 只关掉小束池的保险丝——小规模池化域比生产 700 束域窄，
    夹取占比不作为生产口径核对（生产上限由 manifest 的 ``max_frac_allowed`` 承载）。
    """
    out = tmp_path_factory.mktemp("line2_v2_dataset") / "ds"
    return build_dataset(
        out, anchor, n_beams=N_BEAMS, data_version=V2_DATA_VERSION,
        tiers=SMALL_TIERS, k=SMALL_K, max_clamp_frac=1.0, print_every=100,
    )


# ---------------------------------------------------------------- [S2] C1
def test_v2_particles_match_v1_by_seed(anchor: np.ndarray) -> None:
    """[S2] C1：粒子内容与 v1 同种子推导逐位一致（计数最大绝对差 = 0）。

    口径 = pre1 同法：以 v1 逐束自定标重生成图组，与 ``line2_ds_v1`` 落盘的
    ``target_full`` 逐束逐站比对计数与 bin_edges。抽样 15 束（三切分各 5 束）在
    本测试内实跑；全量 1000 束 × 2 站的对账件（构建 CLI ``--reconcile-only`` 落盘）
    在其存在时由本测试一并断言——该件是 v2 数据集的交付证据，缺失时退化为抽样口径。
    """
    if not V1_DIR.is_dir():
        pytest.skip(f"line2_ds_v1 生产数据不在本机：{V1_DIR}（*.h5 不入库）")
    splits = beam_splits(1000, master_seed=20260913)
    beams = np.concatenate([
        splits[split][:RECONCILE_PER_SPLIT] for split in ("train", "val", "test")
    ])
    report = reconcile_particles_with_v1(anchor, V1_DIR, beams=beams, progress_every=0)
    assert report["n_beams"] == len(beams)
    assert report["mismatches"] == [], report["mismatches"][:5]
    for station_idx in ("0", "1"):
        stats = report["per_station"][station_idx]
        assert stats["n_compared"] == len(beams)
        assert stats["max_abs_count_diff"] == 0
        assert stats["edges_bitwise_equal"] is True

    full = V2_DIR / "reconciliation_v1.json"
    if full.is_file():
        payload = json.loads(full.read_text(encoding="utf-8"))
        assert payload["n_beams"] == 1000
        assert payload["mismatches"] == [], payload["mismatches"][:5]
        for station_idx in ("0", "1"):
            stats = payload["per_station"][station_idx]
            assert stats["n_compared"] == 1000
            assert stats["max_abs_count_diff"] == 0
            assert stats["edges_bitwise_equal"] is True


# ---------------------------------------------------------------- [S2] C2
def test_edges_common_across_beams(anchor: np.ndarray, small_v2: pathlib.Path) -> None:
    """[S2] C2：同站同轴边界跨全部束逐位相同，且由 train 束池化 + 5% 外扩确定性导出。

    四路核对：(a) 落盘件每个 dataset（target_full 与全部子样本）的 bin_edges 逐位等于
    manifest 的 8 组边界；(b) manifest 边界 = 独立重算的 train 池化（证明只由 train 导出，
    val/test 不参与）；(c) 两次独立构建的边界 SHA256 一致；(d) 生产 1000 束件逐束核对。
    """
    manifest = json.loads((small_v2 / "manifest.json").read_text(encoding="utf-8"))
    edges = _manifest_edges(manifest)
    splits = manifest["splits"]
    train_beams = np.array(splits["train"], dtype=np.int64)

    # (a) 落盘件：全部束、全部 dataset 的边界等于 manifest 边界
    checked = 0
    for split in ("train", "val", "test"):
        with h5py.File(small_v2 / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                for station_idx in (0, 1):
                    for name in h5[beam_name][f"station{station_idx}"]:
                        ds = h5[beam_name][f"station{station_idx}"][name]
                        assert np.array_equal(ds.attrs["bin_edges"], edges[station_idx]), (
                            beam_name, station_idx, name,
                        )
                        checked += 1
    assert checked == N_BEAMS * 2 * (1 + len(SMALL_TIERS) * SMALL_K)

    # (b) 独立重算：只由 train 束池化导出
    independent = _independent_edges(anchor, train_beams)
    for station_idx in (0, 1):
        assert np.array_equal(independent[station_idx], edges[station_idx])
    # 池化域随束集变化 ⇒ 实现确实只吃 train（若误用全部束，边界会变宽）
    all_beams = np.arange(N_BEAMS)
    wider = _independent_edges(anchor, all_beams)
    assert not np.array_equal(wider[0], edges[0]) or not np.array_equal(wider[1], edges[1])

    # (c) 二次构建：边界与 SHA256 逐位一致
    again = build_dataset(
        small_v2.parent / "ds_again", anchor, n_beams=N_BEAMS, data_version=V2_DATA_VERSION,
        tiers=SMALL_TIERS, k=SMALL_K, max_clamp_frac=1.0, print_every=100,
    )
    manifest_again = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_again["edges_sha256"] == manifest["edges_sha256"]
    assert _canonical_digest(np.stack([edges[0], edges[1]])) == manifest["edges_sha256"]["combined"]
    for station_idx in (0, 1):
        for axis_idx, axis in enumerate(AXES):
            assert _canonical_digest(edges[station_idx][axis_idx]) == (
                manifest["edges_sha256"][f"station{station_idx}"][axis]
            )

    # (d) 生产件（1000 束）：跨束边界逐位相同（生产 h5 不入库，缺失时退化为上述口径）
    production_manifest = V2_DIR / "manifest.json"
    if production_manifest.is_file():
        prod = json.loads(production_manifest.read_text(encoding="utf-8"))
        prod_edges = _manifest_edges(prod)
        n_checked = 0
        for split in ("train", "val", "test"):
            with h5py.File(V2_DIR / f"{split}.h5", "r") as h5:
                for beam_name in h5:
                    for station_idx in (0, 1):
                        attrs = h5[beam_name][f"station{station_idx}"]["target_full"].attrs["bin_edges"]
                        assert np.array_equal(attrs, prod_edges[station_idx]), (beam_name, station_idx)
                        n_checked += 1
        assert n_checked == 2 * prod["n_beams"] == 2000


# ---------------------------------------------------------------- [S2] C3
def test_theta_knob_visible_in_v2(anchor: np.ndarray, small_v2: pathlib.Path) -> None:
    """[S2] C3（回归钉）：单旋钮 θ 扫掠的目标图组对基准束 M1 ≥ 1×10⁻³。

    "其余旋钮为零" ⇒ δ=0（f_j = 标称焦距）、ℓ=0；基准束 = 同族零旋钮成员（θ=0），
    故扫掠与基准只差一个 θ 平移量。扫掠 = θ ∈ {±5, ±10} µrad（x 与 y 两通道）。
    同一对比在 v1 逐束自定标下 M1 = 0.0（纯平移被逐束定标精确对齐删除，缺陷本体），
    v2 公共域口径下恢复到 pre1 实测的 0.74–0.81 量级，阈值 1×10⁻³ 为保守判"可见"
    下限（`50` D12）。
    """
    manifest = json.loads((small_v2 / "manifest.json").read_text(encoding="utf-8"))
    edges = _manifest_edges(manifest)
    base = _station_images(anchor, _knob_params(), edges)
    measured: dict[str, float] = {}
    for knob, value in (
        ("theta_x_rad", -10e-6), ("theta_x_rad", -5e-6),
        ("theta_x_rad", +5e-6), ("theta_x_rad", +10e-6),
        ("theta_y_rad", -10e-6), ("theta_y_rad", +10e-6),
    ):
        kicked = _station_images(anchor, _knob_params(**{knob: value}), edges)
        for station_idx in (0, 1):
            m1 = group_rel_rmse(kicked[station_idx], base[station_idx])
            measured[f"{knob}{value * 1e6:+.0f}urad_s{station_idx}"] = m1
            assert m1 >= M1_FLOOR, (knob, value, station_idx, m1)
    # 对照钉：v1 逐束自定标把同一 θ 平移完整吸收 ⇒ M1 恒为 0
    v1_base = _v1_station_images(anchor, _knob_params())
    v1_kicked = _v1_station_images(anchor, _knob_params(theta_x_rad=+5e-6))
    for station_idx in (0, 1):
        assert group_rel_rmse(v1_kicked[station_idx], v1_base[station_idx]) == 0.0
    assert len(measured) == 12


# ---------------------------------------------------------------- [S2] C4
def test_v2_transport_matches_v1_non_theta(anchor: np.ndarray, small_v2: pathlib.Path) -> None:
    """[S2] C4：θ 之外的 δ/ℓ 旋钮与站间传输与 v1 逐位一致。

    (a) 全部 1000 束的种子抖动参数与 v1 落盘 ``jitter_*`` 属性逐位相等（同一抖动分布）；
    (b) 抽样束的 v2 落盘 target_full = 同一 ``transport_beam`` 路径 + 公共域夹取的重生成
        （v1 与 v2 共用该传输函数，逐位一致 ⇒ 站间传输行为未被改动）；
    (c) δ 单旋钮与 ℓ 单旋钮对基准束产生非零 M1（θ 外通道未被边层抹平），且两站图组不同。
    """
    # (a) 抖动参数逐位溯源
    if not V1_DIR.is_dir():
        pytest.skip(f"line2_ds_v1 生产数据不在本机：{V1_DIR}（*.h5 不入库）")
    n_checked = 0
    for split in ("train", "val", "test"):
        with h5py.File(V1_DIR / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                beam_id = int(h5[beam_name].attrs["beam_id"])
                if beam_id == 0:
                    assert not any(k.startswith("jitter_") for k in h5[beam_name].attrs)
                    continue
                params = jitter_params(beam_id)
                for key, value in params.items():
                    assert h5[beam_name].attrs[f"jitter_{key}"] == value, (beam_id, key)
                n_checked += 1
    assert n_checked == 999

    # (b) v2 落盘件 = 同一传输路径 + 公共域夹取的重生成
    manifest = json.loads((small_v2 / "manifest.json").read_text(encoding="utf-8"))
    edges = _manifest_edges(manifest)
    sample = (int(manifest["splits"]["train"][0]), int(manifest["splits"]["test"][-1]))
    for beam_id in sample:
        split = next(s for s, ids in manifest["splits"].items() if beam_id in ids)
        with h5py.File(small_v2 / f"{split}.h5", "r") as h5:
            for station_idx, table in enumerate(transport_beam(anchor, beam_idx=beam_id)):
                stored = h5[f"beam_{beam_id:04d}"][f"station{station_idx}"]["target_full"]
                binned, _ = clamp_table(table, edges[station_idx])
                expected = six_projections(binned, edges[station_idx])
                assert np.array_equal(stored[...], expected), (beam_id, station_idx)
                assert np.array_equal(stored.attrs["bin_edges"], edges[station_idx])

    # (c) δ / ℓ 通道与站间传输（基准 = 同族零旋钮成员，单通道隔离）
    base = _station_images(anchor, _knob_params(), edges)
    assert not np.array_equal(base[0], base[1])  # FODO 改变站 1 分布（站间传输生效）
    for params in (_knob_params(delta=0.05), _knob_params(drift_mm=5.0)):
        moved = _station_images(anchor, params, edges)
        for station_idx in (0, 1):
            assert group_rel_rmse(moved[station_idx], base[station_idx]) > 0.0, (params, station_idx)


# ---------------------------------------------------------------- [S2] C5
def test_dataset_v2_manifest(anchor: np.ndarray, small_v2: pathlib.Path) -> None:
    """[S2] C5：manifest 三元组与边层字段（数据版本、git HEAD、主种子、锚点 MD5）。

    小规模构建件与生产入库件（1000 束）分别核对：生产件的束数/切分/档位按生产口径断言，
    小规模件的 ``tiers``/``subsamples_per_tier`` 反映其构建参数。生产件的 code_version
    只验格式（构建后再提交会使它与当前 HEAD 不同步，与 v1 生产件测试同口径）。
    """
    production_manifest = V2_DIR / "manifest.json"
    assert production_manifest.is_file(), f"生产 v2 manifest 缺失：{production_manifest}"
    small = json.loads((small_v2 / "manifest.json").read_text(encoding="utf-8"))
    prod = json.loads(production_manifest.read_text(encoding="utf-8"))
    for payload, n_beams, tiers, k in (
        (small, N_BEAMS, list(SMALL_TIERS), SMALL_K),
        (prod, 1000, list(SUBSAMPLE_TIERS), SUBSAMPLE_K),
    ):
        assert payload["data_version"] == V2_DATA_VERSION == "line2_ds_v2"
        assert payload["master_seed"] == 20260913
        assert payload["anchor_md5"] == ANCHOR_MD5 == "f39d905f96abd5588091dad6dd3884b2"
        assert payload["n_beams"] == n_beams
        assert payload["tiers"] == tiers
        assert payload["subsamples_per_tier"] == k
        assert payload["n_bins"] == N_BINS
        assert payload["edge_rule"] == EDGE_RULE_V2
        assert (len(payload["code_version"]) == 40
                and all(c in "0123456789abcdef" for c in payload["code_version"]))
    assert small["code_version"] == git_head()  # 与生成本次构建的 HEAD 对拍
    assert [len(prod["splits"][s]) for s in ("train", "val", "test")] == [700, 150, 150]
    expected = beam_splits(1000, master_seed=prod["master_seed"])
    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(prod["splits"][split], expected[split])
    assert prod["clamp"]["max_frac_allowed"] <= 0.001
    assert prod["clamp"]["clamped_frac"] <= prod["clamp"]["max_frac_allowed"]


# ---------------------------------------------------------------- [S2] C6
def test_clamp_accounting(anchor: np.ndarray, tmp_path: pathlib.Path) -> None:
    """[S2] C6：域外粒子夹取至最近边界格、计数守恒、逐束计账，占比超限即构建失败。

    机制（合成用例）：越界粒子被夹到该轴边界值，投影计数总和不变。
    构建（真实锚点小规模）：manifest/CSV 的逐 (束, 站) 夹取计数与独立重算一致；
    每个 dataset 六通道计数和 = 其粒子数（``00`` GC5 在夹取后仍守恒）；
    ``max_clamp_frac=0.0`` 时同一构建以 RuntimeError 中止且不落盘。
    """
    # (a) 机制：越界粒子被夹到最近边界格
    rng = np.random.default_rng(np.random.SeedSequence([20260825, 13, 1]))
    table = np.stack([rng.normal(0.0, 0.5, 500)] * 4, axis=1)
    edges = np.stack([np.linspace(-6.0, 6.0, N_BINS + 1)] * 4)
    outside = table.copy()
    outside[0, 0], outside[1, 0] = -8.0, +8.0   # x 下 / 上越界
    outside[2, 1], outside[3, 1] = -9.0, +9.0   # xp 下 / 上越界
    clamped, axis_counts = clamp_table(outside, edges)
    assert axis_counts.tolist() == [2, 2, 0, 0]
    assert clamped[0, 0] == edges[0][0] and clamped[1, 0] == edges[0][-1]
    assert clamped[2, 1] == edges[1][0] and clamped[3, 1] == edges[1][-1]
    np.testing.assert_array_equal(clamped[[0, 1], 1], outside[[0, 1], 1])
    np.testing.assert_array_equal(clamped[[2, 3], 0], outside[[2, 3], 0])
    np.testing.assert_array_equal(clamped[4:, :4], outside[4:, :4])
    images = six_projections(clamped, edges)
    for gi in range(6):
        assert int(images[gi].sum()) == table.shape[0]

    # (b) 构建：逐束计账 + 计数守恒
    out = build_dataset(
        tmp_path / "clamped", anchor, n_beams=N_BEAMS, data_version=V2_DATA_VERSION,
        tiers=(SMALL_TIERS[0],), k=1, max_clamp_frac=1.0, print_every=100,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    edges_by_station = _manifest_edges(manifest)
    n_particles = int(anchor.shape[0])
    with (out / "clamp_accounting.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == N_BEAMS * 2
    recounted = {
        (int(r["beam_id"]), int(r["station"])):
        int(clamp_table(transport_beam(anchor, beam_idx=int(r["beam_id"]))[int(r["station"])],
                        edges_by_station[int(r["station"])])[1].sum())
        for r in rows
    }
    for row in rows:
        key = (int(row["beam_id"]), int(row["station"]))
        assert int(row["n_particles"]) == n_particles
        assert int(row["n_clamped"]) == recounted[key]
    reported_total = sum(recounted.values())
    assert manifest["clamp"]["clamped_particles"] == reported_total > 0
    assert manifest["clamp"]["total_particles"] == 2 * N_BEAMS * n_particles
    assert manifest["clamp"]["clamped_frac"] == pytest.approx(
        reported_total / manifest["clamp"]["total_particles"]
    )
    assert manifest["clamp"]["per_beam_file_sha256"] == hashlib.sha256(
        (out / "clamp_accounting.csv").read_bytes()
    ).hexdigest()

    for split in ("train", "val", "test"):
        with h5py.File(out / f"{split}.h5", "r") as h5:
            for beam_name in h5:
                beam_id = int(h5[beam_name].attrs["beam_id"])
                for station_idx in (0, 1):
                    st = h5[beam_name][f"station{station_idx}"]
                    assert int(st.attrs["n_clamped"]) == recounted[(beam_id, station_idx)]
                    for name in st:
                        expected_count = (
                            n_particles if name == "target_full"
                            else int(name.split("_")[1])  # tier 档位粒子数
                        )
                        image = st[name][...]
                        assert [int(image[gi].sum()) for gi in range(6)] == [expected_count] * 6

    # (c) 保险丝：上限 0 且确有夹取 ⇒ 中止且不落盘
    with pytest.raises(RuntimeError, match="夹取占比"):
        build_dataset(
            tmp_path / "fused", anchor, n_beams=N_BEAMS, data_version=V2_DATA_VERSION,
            tiers=(SMALL_TIERS[0],), k=1, max_clamp_frac=0.0, print_every=100,
        )
    assert not (tmp_path / "fused" / "manifest.json").exists()
    assert not (tmp_path / "fused" / "train.h5").exists()


# ---------------------------------------------------------------- [S1] C6
def test_manifest_edge_rule(small_v2: pathlib.Path) -> None:
    """[S1] C6：manifest 含 edge_rule 版本串、8 组轴边界数值及其 SHA256、投影轴配对元数据。

    8 组 = 4 轴 × 2 站；SHA256 由独立实现（小端 float64 字节）重算逐组核对，
    投影轴配对逐项等于 `00` [S1] 冻结枚举。生产件的 ``n_train_beams`` 按 `50` [S2] 口径为 700。
    """
    payloads = [json.loads((small_v2 / "manifest.json").read_text(encoding="utf-8"))]
    production_manifest = V2_DIR / "manifest.json"
    if production_manifest.is_file():
        production = json.loads(production_manifest.read_text(encoding="utf-8"))
        assert production["edge_rule_detail"]["n_train_beams"] == 700
        payloads.append(production)
    for manifest in payloads:
        assert isinstance(manifest["edge_rule"], str) and manifest["edge_rule"]
        assert manifest["edge_rule_detail"]["pad_frac"] == POOLED_PAD_FRAC
        assert manifest["edge_rule_detail"]["derived_from"] == "train"
        assert manifest["projection_axes"] == [list(pair) for pair in CANONICAL_PROJECTIONS]
        edges = _manifest_edges(manifest)
        for station_idx in (0, 1):
            for axis_idx, axis in enumerate(AXES):
                row = edges[station_idx][axis_idx]
                assert row.shape == (N_BINS + 1,)
                assert (np.diff(row) > 0.0).all()
                assert _canonical_digest(row) == (
                    manifest["edges_sha256"][f"station{station_idx}"][axis]
                )
        assert _canonical_digest(np.stack([edges[0], edges[1]])) == (
            manifest["edges_sha256"]["combined"]
        )
        assert edges_sha256(np.stack([edges[0], edges[1]])) == (
            manifest["edges_sha256"]["combined"]
        )
