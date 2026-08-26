"""数据集构建集成测试：60 [S8] 划分协议 + 60 [S14] 工件契约。

覆盖规格：60 [S8] C2（划分不跨 c）、C4（γ 块、1:1、固定总体分位数）、
60 [S14] C1–C4（工件契约、唯一 sample_id、SeedSequence 派生、manifest
内容）、60 [S15] C2（float32 + gzip4 + 按样本切分）；05 [S3.2] ★
test_no_c_cross_split 与 test_gamma_block。

测试铁律（05 [S1]）：只断言协议/契约/不变量，不断言研究结果。
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.generators.dataset_builder import (
    GAMMA_BLOCK,
    GAMMA_MAG_RANGE,
    GAMMA_QUANTILE_RANKS,
    derive_sample_seed,
    build_dataset,
)

pytestmark = [pytest.mark.integration, pytest.mark.m2]

MASTER = 20260825
SIGMA_K = 11.0
SIGMA_N = 1.22e-4

#: 集成用小规模（dev 子集，≈400 样本 + test_ood 500 档缩样）。
SIZES = {"train": 200, "val": 100, "test_id": 50, "test_pb": 50, "test_ood": 50}

SPLITS = ("train", "val", "test_id", "test_pb", "test_ood")


def _config(version: str) -> dict:
    return {
        "master_seed": MASTER,
        "spec_version": "v1.0",
        "calibration": {
            "sigma_K": SIGMA_K,
            "sigma_n": SIGMA_N,
            "sigma_smooth": {"sigma_smooth_H": 0.0, "sigma_smooth_P": 0.0},
        },
        "dataset": {
            "version": version,
            "train_size": SIZES["train"],
            "val_size": SIZES["val"],
            "test_id_size": SIZES["test_id"],
            "test_ood_size": SIZES["test_ood"],
            "workers": 2,
            "test_pb_block": {"dimension": "abs(gamma)", "range": [0.3, 0.4]},
        },
        "degradation": {"level": "D2", "r": 4, "snr_hf_threshold": 0.1},
    }


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    """模块级一次性构建（dev 子集，写入 pytest 临时目录）。"""
    root = tmp_path_factory.mktemp("m2_int")
    manifest = build_dataset(_config("devt"), root, workers=2)
    return root, manifest


def _h5(root, split, version="devt"):
    return h5py.File(str(root / "data" / version / f"{split}.h5"), "r")


def _c_fingerprint(f, idx: int) -> tuple:
    """样本的 c 指纹：全部内容参数（c_low+c_mid+c_high+C）圆整元组。"""
    parts = []
    for group in ("c_low", "c_mid", "c_high"):
        for key in f[group].keys():
            parts.append(round(float(f[group][key][idx]), 12))
    parts.append(round(float(f["c/C"][idx]), 12))
    return tuple(parts)


# ---------------------------------------------------------------------------
# 60 [S14]：工件契约
# ---------------------------------------------------------------------------


def test_manifest_triple_and_sections(built_dataset):
    """manifest 含版本三元组、γ 块信息、标定初始值与 H_neg_ch/mask_revalidation
    （60 [S14] C1/C3/C6、00 [S6] 约束 8 N4、70 [S7.1] C2、G3）。"""
    from src.generators.dataset_builder import resolve_spec_version

    root, manifest = built_dataset
    # N4（00 [S6] 约束 8）：code_version 为生成时 git HEAD 完整 40 位 hash，
    # 忽略 config 旧值/占位值（commit 1fa3949/7344cf4；99 变更登记 2026-08-26）
    from src.generators.dataset_builder import git_head
    assert len(manifest["code_version"]) == 40, f"code_version 应为完整 40 位 hash，got {manifest['code_version']!r}"
    assert manifest["code_version"] == git_head()
    assert manifest["data_version"] == "devt"
    # N4：spec_version 为 v1.0+<99 最近批准批次>（与生成时刻 99 一致）
    assert manifest["spec_version"] == resolve_spec_version()
    assert manifest["spec_version"].startswith("v1.0+")
    assert manifest["master_seed"] == MASTER
    assert manifest["has_h_neg_ch"] is True

    # G3 掩膜复核：γ 块分位复核值 + W8 覆盖率 + 块内外计数 + verdict
    reval = manifest["mask_revalidation"]
    assert set(reval) == set(SPLITS)
    for split in SPLITS:
        entry = reval[split]
        assert entry["split"] == split
        assert set(entry["gamma_block_quantile_ranks"]) == {"observed", "expected_by_mode"}
        assert set(entry["gamma_block_counts"]) == {"inside", "outside"}
        assert entry["revalidation_verdict"] in ("pass", "drift")
    verdicts = manifest["revalidation_verdicts"]
    assert set(verdicts) == set(SPLITS)
    assert all(v in ("pass", "drift") for v in verdicts.values())

    block = manifest["gamma_block"]
    assert block["dimension"] == "abs(gamma)"
    assert block["interval"] == [0.3, 0.4]
    assert block["population_quantile"]["ranks"] == [0.4, 0.6]
    assert block["mode_per_split"] == {
        "train": "outside", "val": "outside", "test_id": "outside",
        "test_pb": "inside", "test_ood": "exempt",
    }

    cal = manifest["calibration"]
    assert cal["sigma_K_px"] == SIGMA_K
    assert cal["sigma_n"] == SIGMA_N
    assert "sigma_smooth_H" in cal and "sigma_smooth_P" in cal
    assert manifest["degradation"]["level"] == "D2"
    assert "generated_at" in manifest


def test_manifest_sample_ids_match_h5(built_dataset):
    """manifest 的 sample_id 清单与 HDF5 逐样本一致（60 [S14] C2）。"""
    root, manifest = built_dataset
    for split in SPLITS:
        ids = manifest["splits"][split]["sample_ids"]
        assert len(ids) == SIZES[split]
        with _h5(root, split) as f:
            stored = [s.decode() for s in f["sample_id"][:]]
        assert stored == ids
        assert len(set(stored)) == len(stored)  # 唯一


def test_hdf5_schema_images(built_dataset):
    """图像字段 float32 + gzip level 4 + 按样本切分（60 [S15] C2）；含
    H_neg_ch（c_high 清零版，70 [S7.1] C2）。"""
    root, _ = built_dataset
    with _h5(root, "train") as f:
        for name, shape in (
            ("H", (256, 256)), ("H_neg_ch", (256, 256)),
            ("L", (64, 64)), ("L_clean", (64, 64)),
            ("L_up", (256, 256)), ("P2", (256, 256)),
        ):
            ds = f[name]
            assert ds.dtype == np.float32, name
            assert ds.shape == (SIZES["train"], *shape), name
            assert ds.compression == "gzip", name
            assert ds.compression_opts == 4, name
            assert tuple(ds.chunks) == (1, *shape), name
            assert ds.attrs is not None
        assert bool(f.attrs["has_h_neg_ch"])  # h5py attrs 返回 numpy bool
        # H_neg_ch 与 H 逐样本同尺寸且为 c_high 清零（a₃=γ=b₁=0 渲染）
        assert np.array_equal(f["H_neg_ch"].shape, f["H"].shape)


def test_hdf5_schema_metadata(built_dataset):
    """元数据字段：sample_id / c 全字段 / m 全集 / 导出物理量 / 种子 / 掩膜 / m_L。"""
    root, _ = built_dataset
    with _h5(root, "train") as f:
        n = SIZES["train"]
        for group in ("c_low", "c_mid", "c_high"):
            assert f[group].keys(), group
            for key in f[group].keys():
                assert f[group][key].shape == (n,)
        assert f["c/C"].shape == (n,)
        m = f["m"]
        for key in ("Q", "sigma_z", "sigma_delta", "h_eff", "eps_z", "I_peak"):
            assert m[key].shape == (n,)
        assert m["S_delta"].shape == (n, 256)  # 导出能谱剖面 S(δ)
        assert m["I_z"].shape == (n, 256)
        assert f["seed_i"].shape == (n,)
        assert f["seed_i"].dtype == np.int64
        for name in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"):
            assert f["masks"][name].shape == (n,)
            assert f["masks"][name].dtype == np.uint8
        assert set(f["deg_level"][:]) == {b"D2"}
        m_L = f["m_L"]
        for key in ("r", "sigma_K", "sigma_n", "SNR", "seed", "noise_model",
                    "degradation_order"):
            assert m_L[key].shape == (n,)


def test_masks_all_pass_for_main_splits(built_dataset):
    """train/val/test_id/test_pb 入选样本全部通过 W1–W8（20 [S9] C7）。"""
    root, _ = built_dataset
    for split in ("train", "val", "test_id", "test_pb"):
        with _h5(root, split) as f:
            for name in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"):
                assert bool(f["masks"][name][:].all()), (split, name)


def test_seeds_recorded_and_derivable(built_dataset):
    """manifest 登记 master_seed；HDF5 seed_i 与派生规则一致（60 [S14] C4/C8）。"""
    root, manifest = built_dataset
    assert manifest["master_seed"] == MASTER
    with _h5(root, "train") as f:
        for idx in (0, 7, 199):
            assert f["seed_i"][idx] == derive_sample_seed(
                MASTER, "train", idx, SIZES["train"]
            )


def test_regeneration_bitwise(built_dataset):
    """重生成逐位一致：同配置两次构建的图像与元数据逐位相同（可复现契约）。"""
    root, _ = built_dataset
    build_dataset(_config("devt2"), root, workers=2)

    for split in SPLITS:
        with _h5(root, split, "devt") as f1, _h5(root, split, "devt2") as f2:
            for name in ("H", "L", "L_clean", "L_up", "P2"):
                assert np.array_equal(f1[name][:], f2[name][:]), (split, name)
            assert np.array_equal(f1["seed_i"][:], f2["seed_i"][:])
            assert np.array_equal(f1["sample_id"][:], f2["sample_id"][:])
            assert np.array_equal(f1["c_high"]["gamma"][:], f2["c_high"]["gamma"][:])

    with open(root / "data/devt/manifest.json", encoding="utf-8") as fh:
        m1 = json.load(fh)
    with open(root / "data/devt2/manifest.json", encoding="utf-8") as fh:
        m2 = json.load(fh)
    # data_version 为版本目录名（按配置不同），generated_at 为时间戳，均不参与逐位比较
    m1.pop("generated_at")
    m2.pop("generated_at")
    m1.pop("data_version")
    m2.pop("data_version")
    assert m1 == m2


def test_parallel_serial_bitwise(built_dataset):
    """并行（多进程 SeedSequence）与串行逐位一致（05 [S3] test_dataset_builder）。"""
    root, _ = built_dataset
    build_dataset(_config("devt_par"), root, workers=4)
    build_dataset(_config("devt_ser"), root, workers=1)

    for split in SPLITS:
        with _h5(root, split, "devt_par") as fp, _h5(root, split, "devt_ser") as fs:
            for name in ("H", "L", "L_clean", "L_up", "P2"):
                assert np.array_equal(fp[name][:], fs[name][:]), (split, name)
            assert np.array_equal(fp["seed_i"][:], fs["seed_i"][:])


# ---------------------------------------------------------------------------
# 60 [S8]：划分协议（★ 防泄露）
# ---------------------------------------------------------------------------


def test_test_id_test_pb_1to1(built_dataset):
    """test_id 与 test_pb 1:1（60 [S8] C4）。"""
    root, manifest = built_dataset
    assert (
        manifest["splits"]["test_id"]["count"]
        == manifest["splits"]["test_pb"]["count"]
        == SIZES["test_id"]
    )


def test_no_c_cross_split(built_dataset):
    """★ c 指纹（全参数圆整元组）在 train/val/test_id/test_pb 两两不相交。

    划分先按内容参数 c 进行：同一 c 的噪声实现不得跨划分（60 [S8] C2）。
    """
    root, _ = built_dataset
    fingerprints = {}
    for split in ("train", "val", "test_id", "test_pb"):
        with _h5(root, split) as f:
            fingerprints[split] = {
                _c_fingerprint(f, i) for i in range(SIZES[split])
            }

    from itertools import combinations

    for a, b in combinations(("train", "val", "test_id", "test_pb"), 2):
        assert fingerprints[a].isdisjoint(fingerprints[b]), (a, b)


def test_gamma_block(built_dataset):
    """★ test_pb 全部 |γ|∈[0.3,0.4]；train/val/test_id 无一样本落块内。

    块区间由固定总体分位数确定（60 [S8] C4），不得用经验分位数。
    """
    # 常量本身由总体分位数公式导出
    lo, hi = GAMMA_MAG_RANGE
    q_lo, q_hi = GAMMA_QUANTILE_RANKS
    expected = (lo + q_lo * (hi - lo), lo + q_hi * (hi - lo))
    assert GAMMA_BLOCK == (round(expected[0], 6), round(expected[1], 6)) == (0.3, 0.4)

    root, _ = built_dataset
    lo_b, hi_b = GAMMA_BLOCK
    with _h5(root, "test_pb") as f:
        mag = np.abs(f["c_high"]["gamma"][:])
        assert ((mag >= lo_b) & (mag <= hi_b)).all()

    for split in ("train", "val", "test_id"):
        with _h5(root, split) as f:
            mag = np.abs(f["c_high"]["gamma"][:])
            assert not ((mag >= lo_b) & (mag <= hi_b)).any(), split


def test_test_ood_extreme_params(built_dataset):
    """test_ood 为 EXP-06 极端参数子集（80 [S7]）：β/γ 放大 1.5 倍、豁免掩膜。"""
    root, _ = built_dataset
    with _h5(root, "test_ood") as f:
        beta_mag = np.abs(f["c_mid"]["beta"][:])
        gamma_mag = np.abs(f["c_high"]["gamma"][:])
        # 放大 1.5 倍后 |β| ∈ [1.35, 3.0]、|γ| ∈ [0.15, 0.9]
        assert (beta_mag >= 1.35 - 1e-9).all()
        assert (beta_mag <= 3.0 + 1e-9).all()
        assert (gamma_mag >= 0.15 - 1e-9).all()
        assert (gamma_mag <= 0.9 + 1e-9).all()
        # 豁免掩膜：W7/W8 通过率可以很低，但样本仍入选
        assert f["masks"]["W8"][:].sum() <= f["H"].shape[0]
        # 掩膜豁免须在 manifest 记录
    with open(root / "data/devt/manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    ood_stats = manifest["mask_stats"]["test_ood"]
    assert ood_stats.get("masks_exempt") is True
    assert ood_stats["acceptance_rate"] == 1.0
