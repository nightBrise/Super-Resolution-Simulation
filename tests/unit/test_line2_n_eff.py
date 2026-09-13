"""L2 [S2]/[S3] 验收测试（spec line2/20）：误差–N 曲线协议与 N_eff 恒等测试。"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.metrics.n_eff import error_n_curve, n_eff

pytestmark = pytest.mark.unit


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


def test_curve_structure(small_dataset: pathlib.Path, curve) -> None:
    """[S2] C1/C2：五档齐全、中位数>0、互差地板>0、原始值落盘、参照噪声地板>0。"""
    assert [p.n_low for p in curve.points] == [1_000, 3_000, 10_000, 30_000, 100_000]
    for p in curve.points:
        assert p.median > 0 and p.iqr_lo > 0 and p.iqr_hi > 0
        assert p.n_samples == 4 * 2 * 20  # 束 × 站 × k
        assert p.iqr_lo <= p.median <= p.iqr_hi
    assert all(v > 0 for v in curve.mutual_floor.values())
    assert curve.reference_noise_floor > 0


def test_n_eff_identity(small_dataset: pathlib.Path, curve) -> None:
    """[S3] C3 恒等测试：裸 1e4 子样本（tier_010000）作为"输出"，N_eff ∈ [3e3, 3e4]。"""
    import h5py

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
    import h5py

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
