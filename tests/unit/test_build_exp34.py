"""build_exp34.py 重退化测试集生成测试（80 [S6]、60 [S14] 契约）。

覆盖（合成源文件，纯 CPU）：
- 输出 H5Dataset 可直接读取（顶层 H/L/L_up/P2 + c/m/masks/m_L 契约）；
- H 与源逐样本逐位一致（80 [S6] 同一 H 配对）；L 重退化（≠ 源 L）；
- 重退化噪声 seed 与源划分不相交（60 [S8] C4）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generators.dataset_builder import (
    _create_split_file,
    _write_batch,
    SPLIT_BRANCH,
    git_head,
)
from src.generators.build_exp34 import (
    EXP_SEED_BRANCH,
    build_redegraded_exp,
    SOURCE_SPLITS,
)
from src.generators.f_beam import f_beam
from src.generators.f_deg import f_deg
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.utils.h5data import H5Dataset

pytestmark = pytest.mark.unit

MASTER_SEED = 20260825


def _make_source_split(path, n: int, split: str) -> None:
    """构造与主划分同 schema 的合成源文件（2 样本，D2 退化）。"""
    from src.generators.dataset_builder import (
        SIGMA_SMOOTH_P_MULTIPLE,
        apply_masks,
        fine_structure_width,
        sample_parameters,
        DELTA_PX,
    )

    seed = int(np.random.SeedSequence(MASTER_SEED).spawn(8)[SPLIT_BRANCH[split]]
               .generate_state(1, dtype=np.uint32)[0])
    params, _ = sample_parameters(n, master_seed=seed, sigma_K=11.0)
    seed_seq = np.random.SeedSequence(MASTER_SEED).spawn(8)[SPLIT_BRANCH[split]]
    sample_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in seed_seq.spawn(n)]

    h5 = _create_split_file(path, split, n, "v1", MASTER_SEED, git_head())
    records = []
    for i, c in enumerate(params):
        sigma_smooth_h = 0.125 * float(fine_structure_width(c) / DELTA_PX)
        H, m, c_rec = f_beam(c, sigma_smooth=sigma_smooth_h)
        L, L_clean, _d, m_L = f_deg(H, sigma_K=11.0, sigma_n=1.22e-4, seed=sample_seeds[i])
        L_up = normalize_intensity(upsample_4x_bilinear(L))
        P2, _ = f_prior(c, level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h)
        records.append(
            {
                "_sample_id": f"{split}-{i:03d}",
                "H": H, "H_neg_ch": H, "L": L, "L_clean": L_clean,
                "L_up": L_up, "P2": P2,
                "c": c_rec, "m": m, "masks": apply_masks(c, 11.0),
                "seed_i": int(sample_seeds[i]), "deg_level": "D2", "m_L": m_L,
            }
        )
    try:
        _write_batch(h5, records, 0)
    finally:
        h5.close()


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """在 tmp 数据目录造 test_id.h5 + test_pb.h5，并把 build_exp34 的 PROJECT_ROOT 指向它。"""
    import src.generators.build_exp34 as be

    data_dir = tmp_path / "data" / "v1"
    data_dir.mkdir(parents=True)
    _make_source_split(data_dir / "test_id.h5", 3, "test_id")
    _make_source_split(data_dir / "test_pb.h5", 2, "test_pb")
    monkeypatch.setattr(be, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_output_readable_by_h5dataset_and_h_preserved(fake_data_dir):
    be = pytest.importorskip("src.generators.build_exp34")
    manifest = be.build_redegraded_exp("exp03", sigma_K=22.0, sigma_n=1.22e-4,
                                       data_version="v1", master_seed=MASTER_SEED, workers=1)
    assert manifest["count"] == 5
    ds = H5Dataset(manifest["out_path"], "exp03")
    assert len(ds) == 5
    sample = ds[0]
    for key in ("H", "L", "L_up", "P2"):
        assert key in sample
    assert "c_prior_raw" in sample and "c_high" in sample and "m" in sample

    # H 与源逐位一致（同一样本 id）
    import h5py
    with h5py.File(fake_data_dir / "data" / "v1" / "test_id.h5", "r") as src:
        sid = sample["sample_id"]
        idx = [i for i in range(3) if src["sample_id"][i].decode() == sid][0]
        assert np.array_equal(sample["H"].squeeze(0), src["H"][idx])


def test_l_redegraded_and_seeds_disjoint(fake_data_dir):
    be = pytest.importorskip("src.generators.build_exp34")
    manifest = be.build_redegraded_exp("exp03", sigma_K=22.0, sigma_n=1.22e-4,
                                       data_version="v1", master_seed=MASTER_SEED, workers=1)
    import h5py
    with h5py.File(fake_data_dir / "data" / "v1" / "test_id.h5", "r") as src, \
         h5py.File(manifest["out_path"], "r") as new:
        assert not np.array_equal(new["L"][0], src["L"][0])
        src_seeds = set(src["seed_i"][:].tolist())
        new_seeds = set(new["seed_i"][:].tolist())
        assert not (src_seeds & new_seeds)


def test_exp_seed_branches_not_colliding_with_splits():
    used = set(SPLIT_BRANCH.values())
    for branch in EXP_SEED_BRANCH.values():
        assert branch not in used
    assert EXP_SEED_BRANCH["exp03"] != EXP_SEED_BRANCH["exp04"]
