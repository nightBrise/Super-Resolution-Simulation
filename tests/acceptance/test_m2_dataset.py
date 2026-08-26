"""里程碑 M2 验收测试：G0 数据有效性门禁 + 规模/划分/工件契约。

覆盖规格：80 [S9] C3（G0 三判据：(a) W8 覆盖 ≥60%、(b) 探针法
min(s_x)<0.5、(c) SNR_hf 批量中位数 <0.1）、60 [S8] C1/C4（规模、
γ 块、1:1）、60 [S14] C1–C6（工件契约、三元组、唯一 sample_id、种子）、
05 [S5] M2 绑定。

测试铁律（05 [S1]）：G0 阈值与规模/契约均为规格定死的门禁与不变量，断言
它们是协议检查；不断言任何研究结果。L3 验收读已生成产物
（data/dev1、data/v1、results/M2_dataset），不重复训练；唯一例外是
G0(b) 探针为纯 CPU 生成（05 [S2] L3 例外条款，slow 标记）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.generators.dataset_builder import GAMMA_BLOCK, derive_sample_seed
from src.generators.probe import PROBE_N, PROBE_PARAMS, generate_probe_set

pytestmark = [pytest.mark.acceptance, pytest.mark.m2]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DEV1 = PROJECT_ROOT / "data" / "dev1"
DATA_V1 = PROJECT_ROOT / "data" / "v1"
M2_RESULTS = PROJECT_ROOT / "results" / "M2_dataset"

#: 各档划分规模（60 [S8] C1/C4 + 80 [S7]：test_ood 固定 500）。
EXPECTED_COUNTS = {
    "dev1": {"train": 2000, "val": 500, "test_id": 250, "test_pb": 250, "test_ood": 500},
    "v1": {"train": 20000, "val": 2000, "test_id": 1000, "test_pb": 1000, "test_ood": 500},
}


def _manifest(data_dir: Path) -> dict:
    with open(data_dir / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def _gamma_mags(data_dir: Path, split: str) -> np.ndarray:
    with h5py.File(str(data_dir / f"{split}.h5"), "r") as f:
        return np.abs(f["c_high"]["gamma"][:])


def _require_artifacts() -> None:
    """L3 前置：数据与结果产物必须已生成（未生成视为 M2 未交付）。"""
    for path in (
        DATA_DEV1 / "manifest.json",
        DATA_V1 / "manifest.json",
        DATA_V1 / "train.h5",
        M2_RESULTS / "g0_report.json",
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"M2 产物缺失：{path}——请先运行 build_dataset 与 probe 再验收"
            )


# ---------------------------------------------------------------------------
# 规模与工件契约（60 [S8]/[S14]）
# ---------------------------------------------------------------------------


def test_counts_dev1_and_v1():
    """调试与标准规模计数正确（dev1: 2000/500/250/250；v1: 20000/2000/1000/1000）。"""
    _require_artifacts()
    for scale in ("dev1", "v1"):
        manifest = _manifest(PROJECT_ROOT / "data" / scale)
        for split, expected in EXPECTED_COUNTS[scale].items():
            assert manifest["splits"][split]["count"] == expected, (scale, split)
            assert len(manifest["splits"][split]["sample_ids"]) == expected
        # HDF5 行数一致
        for split in ("train", "val", "test_id", "test_pb", "test_ood"):
            with h5py.File(str(PROJECT_ROOT / "data" / scale / f"{split}.h5"), "r") as f:
                assert f["H"].shape[0] == EXPECTED_COUNTS[scale][split]


def test_manifest_triple():
    """manifest 登记 code_version/data_version/spec_version 三元组（60 [S14] C1）。"""
    _require_artifacts()
    for scale in ("dev1", "v1"):
        manifest = _manifest(PROJECT_ROOT / "data" / scale)
        for key in ("code_version", "data_version", "spec_version"):
            assert isinstance(manifest.get(key), str) and manifest[key], (scale, key)
        assert manifest["data_version"] == scale
        # N4 版本对账（99 行 163 登记）：spec_version 格式 v1.0 或 v1.0+YYYY-MM-DD
        # （dev1 为历史格式 v1.0，v1 起为 v1.0+批次日期，90 [S5] N8 对账清单）
        assert re.match(r"^v1\.0(\+\d{4}-\d{2}-\d{2})?$", manifest["spec_version"]), (scale, manifest["spec_version"])
        assert manifest["master_seed"] == 20260825


def test_gamma_block_info_and_1to1():
    """manifest 含 γ 块信息（维度/区间/分位派生/各子集样本数）且 test 1:1。"""
    _require_artifacts()
    manifest = _manifest(DATA_V1)
    block = manifest["gamma_block"]
    assert block["dimension"] == "abs(gamma)"
    assert block["interval"] == [0.3, 0.4]
    assert block["population_quantile"]["ranks"] == [0.4, 0.6]
    assert block["mode_per_split"]["train"] == "outside"
    assert block["mode_per_split"]["test_pb"] == "inside"
    assert (
        manifest["splits"]["test_id"]["count"]
        == manifest["splits"]["test_pb"]["count"]
        == 1000
    )


def test_gamma_block_on_real_data():
    """v1 实数据：test_pb 全部 |γ|∈[0.3,0.4]；train/val/test_id 无一样本落块内。"""
    _require_artifacts()
    lo, hi = GAMMA_BLOCK
    assert (lo, hi) == (0.3, 0.4)
    for split, expect_inside in (
        ("test_pb", True), ("train", False), ("val", False), ("test_id", False),
    ):
        mag = _gamma_mags(DATA_V1, split)
        inside = (mag >= lo) & (mag <= hi)
        assert inside.all() if expect_inside else not inside.any(), split


def test_ood_500_and_disjoint():
    """test_ood 500 样本、sample_id 与其余划分不相交（80 [S7] + 60 [S14] C2）。"""
    _require_artifacts()
    manifest = _manifest(DATA_V1)
    ood_ids = set(manifest["splits"]["test_ood"]["sample_ids"])
    assert len(ood_ids) == 500
    for split in ("train", "val", "test_id", "test_pb"):
        others = set(manifest["splits"][split]["sample_ids"])
        assert ood_ids.isdisjoint(others), split


def test_seed_derivation_recorded():
    """manifest 种子登记：master_seed 与 HDF5 seed_i 按派生规则一致（60 [S14] C4/C8）。"""
    _require_artifacts()
    manifest = _manifest(DATA_V1)
    assert manifest["master_seed"] == 20260825
    with h5py.File(str(DATA_V1 / "train.h5"), "r") as f:
        for idx in (0, 1, 9999):
            assert f["seed_i"][idx] == derive_sample_seed(
                20260825, "train", idx, 20000
            )


# ---------------------------------------------------------------------------
# G0 三判据（80 [S9] C3）
# ---------------------------------------------------------------------------


def test_g0_a_w8_coverage():
    """G0(a)：候选（过 W1–W7）中 W8 比例 ≥ 60%（20 [S9] C9 生成期统计口径）。"""
    _require_artifacts()
    manifest = _manifest(DATA_V1)
    stats = manifest["mask_stats"]["train"]
    fraction = float(stats["w8_fraction_among_w1_w7_passers"])
    assert fraction >= 0.6, f"W8 覆盖 {fraction:.3f} < 0.6"


@pytest.mark.slow
def test_g0_b_probe_method():
    """G0(b)：探针法 3 参数 × 200 样本，min(s_x) < 0.5（slow，CPU 并行可用）。

    s_x 为探针样本中 ρ ≥ 0.1 的占比；探针集 SHALL NOT 进入训练数据集。
    """
    _require_artifacts()
    manifest = _manifest(DATA_V1)
    sigma_K = float(manifest["calibration"]["sigma_K_px"])
    train_ids = set(manifest["splits"]["train"]["sample_ids"])

    s_x_values: dict[str, float] = {}
    probe_ids: list[str] = []
    for param in sorted(PROBE_PARAMS):
        records, summary = generate_probe_set(
            param, 20260825, sigma_K, n=PROBE_N
        )
        s_x_values[param] = summary["s_x"]
        probe_ids.extend(r["sample_id"] for r in records)

    min_s = min(s_x_values.values())
    assert min_s < 0.5, f"min(s_x) = {min_s:.3f} ≥ 0.5（{s_x_values}）"
    assert set(probe_ids).isdisjoint(train_ids), "探针集与训练集 sample_id 相交"


def test_g0_c_snr_hf():
    """G0(c)：SNR_hf 批量中位数 < 0.1（30 [S6] C8；dev1 实算 + v1 报告值）。"""
    _require_artifacts()
    from src.generators.f_deg import snr_hf
    from src.generators.probe import evaluate_snr_hf_median

    # dev1 训练集实算（2000 样本，CPU 并行）
    median_dev1, n = evaluate_snr_hf_median(
        DATA_DEV1 / "train.h5", 2000, workers=16
    )
    assert median_dev1 < 0.1, f"dev1 SNR_hf 中位数 {median_dev1:.4f} ≥ 0.1"
    assert n == 2000

    # v1 训练集（g0_report.json 记录的评估值）
    with open(M2_RESULTS / "g0_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    crit_c = report["criteria"]["c_snr_hf"]
    assert crit_c["value"] < 0.1, f"v1 SNR_hf 中位数 {crit_c['value']:.4f} ≥ 0.1"
    assert crit_c["n_samples"] == 20000


def test_g0_report_pass_and_artifacts():
    """g0_report.json 存在、三项判据全过、探针证据已归档（80 [S9] C10）。"""
    _require_artifacts()
    with open(M2_RESULTS / "g0_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["gate"] == "G0"
    assert report["verdict"] == "pass"
    for key in ("a_w8_coverage", "b_probe_survival", "c_snr_hf"):
        assert report["criteria"][key]["passed"] is True, key
    assert (M2_RESULTS / "probe_sets.h5").exists()
    assert (M2_RESULTS / "probe_report.json").exists()
    # 探针集与训练集不相交的记录
    assert report["criteria"]["b_probe_survival"]["probe_set_not_in_training"] is True
