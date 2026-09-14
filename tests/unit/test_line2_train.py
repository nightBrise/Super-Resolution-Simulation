"""L3b [S3]/[S5] 验收测试（spec line2/30）：训练冒烟、确定性、checkpoint、推理。

小配置（3 束 × 100,001 粒子数据集、base=24、depth=3、30 步）保证单元测试
时间预算；生产规模运行由管线 CLI 承担。
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest
import torch

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.training.data import Line2Pairs
from src.line2.training.infer import infer
from src.line2.training.train import train

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    from tests.unit.test_line2_dataset import _synthetic_table

    out = tmp_path_factory.mktemp("ds_train")
    anchor = _synthetic_table(100_001, seed=99)
    # 10 束：保证 val 切分非空（70/15/15 → 7/2/1）；构建 ~1-2 min
    return build_dataset(out, anchor, n_beams=10, data_version="test_train")


@pytest.fixture(scope="module")
def train_out(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """训练冒烟的输出目录（模块级共享，供 infer 测试消费 best_val.ckpt）。"""
    out = tmp_path_factory.mktemp("train_runs") / "run_a"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg(data_dir: pathlib.Path, out_dir: pathlib.Path, seed: int) -> dict:
    return {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "max_steps": 30,
        "eval_every": 10,
        "patience": 10,
        "batch": 2,
        "base": 24,
        "num_levels": 3,
        "blocks": 1,
        "lr": 3e-4,
        "n_low": 10_000,
        "n_high": 200_000.0,
        "w_psd": 0.1,
        "device": "cpu",
        "torch_threads": 2,
    }


def test_train_smoke_and_determinism(data_dir: pathlib.Path, train_out: pathlib.Path) -> None:
    """[S3] C1/C3：训练可运行、日志/ckpt 落盘；同种子 val N_eff 轨迹逐位一致。"""
    out_a = train_out
    out_b = train_out.parent / "run_b"
    summary_a = train(_cfg(data_dir, out_a, seed=20260913))
    summary_b = train(_cfg(data_dir, out_b, seed=20260913))
    assert summary_a["steps"] == 30 and summary_b["steps"] == 30
    assert summary_a["val_n_eff_history"] == summary_b["val_n_eff_history"]
    assert (out_a / "best_val.ckpt").is_file() and (out_a / "last.ckpt").is_file()
    entries = [
        json.loads(line)
        for line in (out_a / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 3  # eval_every=10 × 30 步
    assert all("val_n_eff" in e and "train_nll" in e for e in entries)


def test_infer_roundtrip(data_dir: pathlib.Path, train_out: pathlib.Path, tmp_path: pathlib.Path) -> None:
    best_ckpt = train_out / "best_val.ckpt"
    if not best_ckpt.is_file():
        # 自足守卫：字母序下本测试可能先于训练冒烟运行——先跑一次 30 步训练
        train(_cfg(data_dir, train_out, seed=20260913))
    """[S5] C1–C3：推理确定性、率图总计数 = N_high。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    out = tmp_path / "infer_out"
    args = [
        "--ckpt", str(data_dir / ".." / "ckpt_placeholder" / "best_val.ckpt"),
        "--input", str(data_dir / "train.h5"),
        "--beam-id", str(beam_id),
        "--station", "0",
        "--n-low", "10000",
        "--sub", "0",
        "--n-high", "200000",
        "--out", str(out),
    ]
    # 训练冒烟产出的 best_val.ckpt 作为 ckpt（占位路径替换）
    best_ckpt = train_out / "best_val.ckpt"
    if not best_ckpt.is_file():  # 依赖顺序保护：smoke 先行时已存在
        pytest.skip("训练冒烟未先行")
    args[args.index("--ckpt") + 1] = str(best_ckpt)
    rc = infer_main(args)
    assert rc == 0
    data = np.load(out / "group_n.npz")
    assert data["prob"].shape == (6, 128, 128) and data["rate"].shape == (6, 128, 128)
    per_proj = data["rate"].sum(axis=(1, 2))
    np.testing.assert_allclose(per_proj, 200_000.0, rtol=1e-4)  # C3 率语义
    # 确定性
    out2 = tmp_path / "infer_out2"
    args[args.index("--out") + 1] = str(out2)
    assert infer_main(args) == 0
    a = np.load(out2 / "group_n.npz")
    np.testing.assert_array_equal(a["prob"], data["prob"])


def infer_main(argv: list[str]) -> int:
    from src.line2.training import infer as infer_mod

    return infer_mod.main(argv)
