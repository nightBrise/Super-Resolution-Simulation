"""L2 [S2]/[S3] 验收测试（spec line2/20）：误差–N 曲线协议与 N_eff 恒等测试。"""

from __future__ import annotations

import json
import math
import pathlib

import h5py
import numpy as np
import pytest

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.metrics.metrics import group_rel_rmse
from src.line2.metrics.n_eff import error_n_curve, n_eff, reference_noise_floor

pytestmark = pytest.mark.unit

#: `20` [S2] 预注册档位（2e5 = 参照本身，不设节点，D7）。
CURVE_TIERS = (1_000, 3_000, 10_000, 30_000, 100_000)
#: 每档独立子样本数（`10` [S2]）；相邻不相交对 → k//2 对/束/站（20 D14）。
SUBSAMPLE_K = 20
STATIONS = (0, 1)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """4 束 × 1×10⁵ 粒子的小数据集（曲线协议测试用；生产 1000 束由后台构建）。"""
    from tests.unit.test_line2_dataset import _synthetic_table

    out = tmp_path_factory.mktemp("ds")
    anchor = _synthetic_table(100_001, seed=99)
    return build_dataset(out, anchor, n_beams=4, data_version="test_curve")


@pytest.fixture(scope="module")
def curve(small_dataset: pathlib.Path):
    curve = error_n_curve(small_dataset, beam_ids=list(range(4)))
    return curve


def test_curve_structure(small_dataset: pathlib.Path, curve, tmp_path: pathlib.Path) -> None:
    """[S2] C1/C2：五档齐全、中位数>0、互差地板>0、原始值落盘、参照噪声地板>0。"""
    import json as _json

    assert [p.n_low for p in curve.points] == [1_000, 3_000, 10_000, 30_000, 100_000]
    for p in curve.points:
        assert p.median > 0 and p.iqr_lo > 0 and p.iqr_hi > 0
        assert p.n_samples == 4 * 2 * 20  # 束 × 站 × k
        assert p.iqr_lo <= p.median <= p.iqr_hi
    assert all(v > 0 for v in curve.mutual_floor.values())
    assert curve.reference_noise_floor > 0
    # F3：原始逐样本值落盘
    dump = tmp_path / "curve.json"
    error_n_curve(small_dataset, beam_ids=list(range(4)), dump=dump)
    payload = _json.loads(dump.read_text(encoding="utf-8"))
    assert set(payload["raw_samples"]) == {"1000", "3000", "10000", "30000", "100000"}
    assert len(payload["raw_samples"]["10000"]) == 4 * 2 * 20


def _manual_pair_m1(data_dir: pathlib.Path, beam_ids: list[int]) -> dict[int, list[float]]:
    """按 D14 口径（相邻不相交对 (0,1),(2,3),…）从数据集手工重算每档 M1 互差样本。

    独立于 ``error_n_curve``：直接读 HDF5 子样本图组、自己配对、自己调 M1，
    用于对拍实现返回的地板值（不引入外部随机性，`20` [S2] C2）。
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    split_of = {int(b): s for s, ids in manifest["splits"].items() for b in ids}
    pairs: dict[int, list[float]] = {t: [] for t in CURVE_TIERS}
    files: dict[str, h5py.File] = {}
    try:
        for beam_id in beam_ids:
            split = split_of[beam_id]
            h5 = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            for station_idx in STATIONS:
                stg = h5[f"beam_{beam_id:04d}/station{station_idx}"]
                for tier in CURVE_TIERS:
                    imgs = [
                        stg[f"tier_{tier:06d}_sub_{sub_idx:02d}"][...]
                        for sub_idx in range(SUBSAMPLE_K)
                    ]
                    for a in range(0, SUBSAMPLE_K - 1, 2):
                        pairs[tier].append(group_rel_rmse(imgs[a], imgs[a + 1]))
    finally:
        for f in files.values():
            f.close()
    return pairs


def test_floor_definitions_match_manual_recompute(small_dataset: pathlib.Path, curve) -> None:
    """[S2] C2/C3：两条地板线与曲线点的口径 = 手工重算值（rtol 1e-12）。

    - 互差地板 = 同档两两独立子样本 M1 差的中位数；配对口径取 D14 钉死的
      **相邻不相交对** (0,1),(2,3),…,(18,19)（k=20 → 10 对/束/站）。若实现改用
      全组合 C(20,2)=190 对，参与中位数的样本集不同，数值对拍即失败——配对口径
      由此可证伪。
    - 参照噪声地板 = 1e5 档**同一配对集合**的中位数 / √2（C3），故它必须恰等于
      ``mutual_floor[100_000] / √2``（逐位，不允许各自独立估计）。
    - 曲线点 = ``raw_samples`` 的中位数与 25/75 分位（C1 的落盘复核口径）。
    """
    beam_ids = list(range(4))
    pairs = _manual_pair_m1(small_dataset, beam_ids)
    n_pairs_expected = len(beam_ids) * len(STATIONS) * (SUBSAMPLE_K // 2)
    for tier in CURVE_TIERS:
        assert len(pairs[tier]) == n_pairs_expected
        np.testing.assert_allclose(
            curve.mutual_floor[tier],
            float(np.median(pairs[tier])),
            rtol=1e-12,
            atol=0.0,
            err_msg=f"档 {tier} 互差地板与手工重算不符",
        )
    assert curve.reference_noise_floor == curve.mutual_floor[100_000] / np.sqrt(2.0)

    for p in curve.points:
        vals = np.asarray(curve.raw_samples[p.n_low], dtype=np.float64)
        assert p.n_samples == vals.size == len(beam_ids) * len(STATIONS) * SUBSAMPLE_K
        assert p.median == float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        assert (p.iqr_lo, p.iqr_hi) == (float(lo), float(hi))


def test_curve_without_reference_tier_marks_floor_nan(
    small_dataset: pathlib.Path, curve, tmp_path: pathlib.Path
) -> None:
    """[S2] C3：tiers 不含 1e5 档 → 参照噪声地板取 nan（不抛 KeyError）；含该档时口径不变。

    两者只在"有无 1e5 档"上不同（同束、同站、同 k），故共享档的曲线点与互差地板必须
    逐位一致——把缺档的影响限定在参照噪声地板这一个字段上，而非静默换一种估计方式。
    """
    beam_ids = list(range(4))
    shared = (1_000, 3_000, 10_000)
    partial = error_n_curve(
        small_dataset, beam_ids=beam_ids, stations=(0,), tiers=shared, k=2
    )
    assert [p.n_low for p in partial.points] == list(shared)
    assert set(partial.mutual_floor) == set(shared)
    assert math.isnan(partial.reference_noise_floor)
    assert math.isnan(reference_noise_floor(partial))          # 字段与访问器同源

    full = error_n_curve(
        small_dataset, beam_ids=beam_ids, stations=(0,), tiers=(*shared, 100_000), k=2
    )
    for p in partial.points:
        assert p.median == full.median_at(p.n_low)
        assert partial.mutual_floor[p.n_low] == full.mutual_floor[p.n_low]
    assert not math.isnan(full.reference_noise_floor)
    assert full.reference_noise_floor == full.mutual_floor[100_000] / math.sqrt(2.0)

    # dump 路径同样不因缺档而中断
    dump = tmp_path / "partial_curve.json"
    error_n_curve(
        small_dataset, beam_ids=beam_ids, stations=(0,), tiers=shared, k=2, dump=dump
    )
    payload = json.loads(dump.read_text(encoding="utf-8"))
    assert math.isnan(payload["reference_noise_floor"])


def test_error_n_curve_repeat_call_identical(small_dataset: pathlib.Path, curve) -> None:
    """[S2] C1：同数据集同参数两次调用 → 曲线点/两条地板/原始样本逐位一致。"""
    again = error_n_curve(small_dataset, beam_ids=list(range(4)))

    def point_tuples(result) -> list[tuple]:
        return [
            (p.n_low, p.median, p.iqr_lo, p.iqr_hi, p.n_samples) for p in result.points
        ]

    assert point_tuples(again) == point_tuples(curve)
    assert again.mutual_floor == curve.mutual_floor
    assert again.reference_noise_floor == curve.reference_noise_floor
    assert again.raw_samples == curve.raw_samples  # dict[int, list[float]] → 逐位比较


def test_n_eff_identity(small_dataset: pathlib.Path, curve) -> None:
    """[S3] C3 恒等测试：裸 1e4 子样本（tier_010000）作为"输出"，N_eff ∈ [3e3, 3e4]。"""
    manifest = json.loads((small_dataset / "manifest.json").read_text(encoding="utf-8"))
    beam_id = manifest["splits"]["train"][0]
    with h5py.File(small_dataset / "train.h5", "r") as h5:
        stg = h5[f"beam_{beam_id:04d}/station0"]
        out = stg["tier_010000_sub_00"][...]  # 1e4 档（六位补零：010000）
        ref = stg["target_full"][...]
    value, flag = n_eff(out, ref, curve)
    assert flag is None
    assert 3_000 <= value <= 30_000


def test_n_eff_overflow_and_determinism(small_dataset: pathlib.Path, curve) -> None:
    """[S3] C2：完美输出（=参照自身）→ 上溢（10× 达成）；同输入逐位确定。"""
    manifest = json.loads((small_dataset / "manifest.json").read_text(encoding="utf-8"))
    beam_id = manifest["splits"]["train"][0]
    with h5py.File(small_dataset / "train.h5", "r") as h5:
        ref = h5[f"beam_{beam_id:04d}/station0/target_full"][...]
    value, flag = n_eff(ref.copy(), ref, curve)
    assert (value, flag) == (100_000.0, "overflow")
    value2, flag2 = n_eff(ref.copy(), ref, curve)
    assert (value2, flag2) == (value, flag)


def test_n_eff_underflow(small_dataset: pathlib.Path, curve) -> None:
    """[S3] C4 消费端：全零输出 → 下溢标记。"""
    zero = np.zeros((6, 128, 128), dtype=np.int32)
    ref = np.full((6, 128, 128), 10, dtype=np.int32)
    value, flag = n_eff(zero, ref, curve)
    assert (value, flag) == (1_000.0, "underflow")
