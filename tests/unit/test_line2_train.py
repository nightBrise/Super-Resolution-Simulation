"""L3b [S3]–[S5] 验收测试（spec line2/30）：训练冒烟、确定性、日志契约、test 束隔离、推理。

小配置（10 束 × 100,001 粒子合成数据集、joint base=24 / independent ind_base=16、
depth=3、20–30 步）保证单元测试时间预算；生产规模运行由管线 CLI 承担。
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import h5py
import numpy as np
import pytest
import torch

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.training.common import config_hash
from src.line2.training.infer import infer, load_ckpt_payload
from src.line2.training.infer import main as infer_main
from src.line2.training.train import _forward_variant, build_model, should_update_best, train

pytestmark = pytest.mark.unit

#: `30` [S3] C3 要求逐评估周期落盘的字段。
LOG_KEYS = ("step", "train_nll", "val_nll", "val_n_eff", "lr", "grad_norm")

SEED = 20260913


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    from tests.unit.test_line2_dataset import _synthetic_table

    out = tmp_path_factory.mktemp("ds_train")
    anchor = _synthetic_table(100_001, seed=99)
    # 10 束：保证 val 切分非空（70/15/15 → 7/2/1）；构建 ~40 s
    return build_dataset(out, anchor, n_beams=10, data_version="test_train")


@pytest.fixture(scope="module")
def joint_run(tmp_path_factory: pytest.TempPathFactory, data_dir: pathlib.Path) -> dict:
    """联合版 30 步冒烟（模块级共享）：run A 供日志/ckpt 消费，run B 供确定性对比。"""
    root = tmp_path_factory.mktemp("train_runs")
    cfg = _cfg(data_dir, root / "run_a", SEED)
    return {
        "cfg": cfg,
        "out_a": root / "run_a",
        "out_b": root / "run_b",
        "a": train(cfg),
        "b": train(dict(cfg, out_dir=str(root / "run_b"))),
    }


@pytest.fixture(scope="module")
def ind_run(tmp_path_factory: pytest.TempPathFactory, data_dir: pathlib.Path) -> dict:
    """独立版 20 步冒烟（`30` [S4] C2 训练路径）：同种子两次，供确定性与推理消费。"""
    root = tmp_path_factory.mktemp("ind_runs")
    cfg = _cfg(
        data_dir, root / "run_a", SEED,
        variant="independent", ind_base=16, ind_blocks=3, max_steps=20, eval_every=10,
    )
    return {
        "cfg": cfg,
        "out_a": root / "run_a",
        "out_b": root / "run_b",
        "a": train(cfg),
        "b": train(dict(cfg, out_dir=str(root / "run_b"))),
    }


def _cfg(data_dir: pathlib.Path, out_dir: pathlib.Path, seed: int, **over) -> dict:
    cfg = {
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
    cfg.update(over)
    return cfg


def _read_log(run_dir: pathlib.Path) -> list[dict]:
    text = (run_dir / "train_log.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_ckpt(run_dir: pathlib.Path, name: str = "best_val.ckpt") -> dict:
    return torch.load(run_dir / name, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- [S3] 训练循环
def test_should_update_best_strict() -> None:
    """[S3] C2：best_val 选择规则 = 严格更优才更新（人工退化/持平序列）。"""
    best = float("-inf")
    updated_at: list[int] = []
    for i, val in enumerate([1.0, 0.9, 0.95, 1.1, 1.1]):
        if should_update_best(val, best):
            best = val
            updated_at.append(i)
    assert updated_at == [0, 3]  # 0.9/0.95 劣于 1.0；末尾 1.1 与 best 持平不再更新
    assert best == 1.1
    assert not should_update_best(1.1, 1.1)
    assert not should_update_best(0.5, 1.1)
    assert should_update_best(-2.0, float("-inf"))  # 初值哨兵下首次评估必更新


def test_train_smoke_and_determinism(joint_run: dict) -> None:
    """[S3] C1：训练可运行、ckpt 落盘；同种子同 config 的 val N_eff 轨迹逐位一致。"""
    assert joint_run["a"]["steps"] == 30 and joint_run["b"]["steps"] == 30
    assert joint_run["a"]["val_n_eff_history"] == joint_run["b"]["val_n_eff_history"]
    for key in ("out_a", "out_b"):
        assert (joint_run[key] / "best_val.ckpt").is_file()
        assert (joint_run[key] / "last.ckpt").is_file()


def test_best_ckpt_matches_argmax(joint_run: dict) -> None:
    """[S3] C2：best_val.ckpt 对应 val N_eff 最优的评估周期，且携带 config 哈希与种子。"""
    summary = joint_run["a"]
    history = summary["val_n_eff_history"]
    assert summary["best_val_n_eff"] == max(history)
    # eval_every=10 且严格更优才更新 → best_step = 首个最大值所在周期
    assert summary["best_step"] == 10 * (int(np.argmax(history)) + 1)
    payload = _load_ckpt(joint_run["out_a"])
    assert payload["step"] == summary["best_step"]
    assert payload["best_val_n_eff"] == summary["best_val_n_eff"]
    assert payload["master_seed"] == SEED
    assert payload["cfg_hash"] == config_hash(payload["config"])


def test_train_log_contract(joint_run: dict) -> None:
    """[S3] C3：逐评估周期落盘 step/train NLL/val NLL/val N_eff/lr/grad_norm，可 json 解析。"""
    entries = _read_log(joint_run["out_a"])
    assert [e["step"] for e in entries] == [10, 20, 30]  # eval_every=10 × max_steps=30
    for entry in entries:
        assert set(LOG_KEYS) <= set(entry)
        for key in LOG_KEYS:
            assert isinstance(entry[key], int | float) and math.isfinite(entry[key])
        assert entry["grad_norm"] >= 0.0  # clip_grad_norm_ 的总范数不变量
        assert entry["lr"] == pytest.approx(3e-4)  # 记录的是优化器实际 lr


def test_training_never_reads_test_split(
    data_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[S3] C5 / GC3：训练全程不读 test 束——数据目录里根本没有 test.h5 也能跑完。

    可证伪性：``train`` 只经 ``h5py.File`` 触达数据集；若实现（含 ``val_curve_for``
    的曲线口径）打开 test.h5，此处会因文件缺失抛 FileNotFoundError 而失败；spy 同时
    记录全部实际打开路径，任何指向 test 束的读取都会被第二条断言直接捕获。
    """
    ds = tmp_path / "ds_no_test"
    ds.mkdir()
    for name in ("train.h5", "val.h5", "manifest.json"):
        (ds / name).symlink_to(data_dir / name)
    assert not (ds / "test.h5").exists()

    opened: list[str] = []
    real_file = h5py.File

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", spy)
    summary = train(_cfg(ds, tmp_path / "out_no_test", SEED))

    assert summary["steps"] == 30
    assert any(p.endswith("train.h5") for p in opened)  # spy 生效（否则下条形同虚设）
    assert any(p.endswith("val.h5") for p in opened)
    assert not [p for p in opened if "test.h5" in p]


# ---------------------------------------------------------------- [S4] 独立版对照
def test_independent_variant_training(ind_run: dict) -> None:
    """[S4] C2：独立版训练路径可跑完并落盘 summary；同种子 val N_eff 轨迹逐位一致。"""
    assert ind_run["a"]["variant"] == "independent" == ind_run["b"]["variant"]
    assert ind_run["a"]["steps"] == 20 == ind_run["b"]["steps"]
    assert ind_run["a"]["val_n_eff_history"] == ind_run["b"]["val_n_eff_history"]
    for key in ("out_a", "out_b"):
        run_dir = ind_run[key]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["variant"] == "independent" and summary["steps"] == 20
        assert len(_read_log(run_dir)) == 2  # eval_every=10 × max_steps=20
        assert all("val_n_eff" in e for e in _read_log(run_dir))

    # 结构契约（`30` [S4] / D4）：6 个互不共享权重的单投影子网，无共享骨干
    state = _load_ckpt(ind_run["out_a"])["state_dict"]
    assert {k.split(".")[0] for k in state} == {"nets"}
    assert {k.split(".")[1] for k in state} == {str(i) for i in range(6)}


# ---------------------------------------------------------------- [S5] 推理
def _run_infer(ckpt: pathlib.Path, data_dir: pathlib.Path, out: pathlib.Path,
               beam_id: int, station: int = 0, sub: int = 0) -> dict:
    """跑一次推理 CLI 并返回 npz 全部数组（返回码在此断言，调用方只看内容）。"""
    rc = infer_main([
        "--ckpt", str(ckpt),
        "--input", str(data_dir / "train.h5"),
        "--beam-id", str(beam_id),
        "--station", str(station),
        "--n-low", "10000",
        "--sub", str(sub),
        "--n-high", "200000",
        "--out", str(out),
    ])
    assert rc == 0
    with np.load(out / "group_n.npz") as data:
        return {k: data[k] for k in data.files}


def _assert_group_n(arrays: dict, ckpt: pathlib.Path, beam_id: int, sub: int,
                    station: int = 0) -> None:
    """[S5] C2/C3：双套数组、形状、严格正、逐投影率和 = N_high、来源四元组 + ckpt 哈希。"""
    prob, rate = arrays["prob"], arrays["rate"]
    assert prob.shape == (6, 128, 128) and rate.shape == (6, 128, 128)
    assert prob.dtype == np.float64 and rate.dtype == np.float64
    assert (prob > 0).all()
    np.testing.assert_allclose(prob.sum(axis=(1, 2)), 1.0, rtol=1e-5)
    np.testing.assert_allclose(rate.sum(axis=(1, 2)), 200_000.0, rtol=1e-4)
    np.testing.assert_array_equal(rate, prob * 200_000.0)  # D7 率语义：rate = p·N_high
    assert int(arrays["beam_id"]) == beam_id
    assert int(arrays["station"]) == station
    assert int(arrays["n_low"]) == 10_000
    assert int(arrays["sub_idx"]) == sub
    assert str(arrays["ckpt_md5"]) == hashlib.md5(ckpt.read_bytes()).hexdigest()


def test_infer_roundtrip(joint_run: dict, data_dir: pathlib.Path,
                         tmp_path: pathlib.Path) -> None:
    """[S5] C1–C3：联合版 ckpt 推理确定性、概率/率双套输出、率图总计数 = N_high。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    ckpt = joint_run["out_a"] / "best_val.ckpt"
    first = _run_infer(ckpt, data_dir, tmp_path / "infer_out", beam_id)
    _assert_group_n(first, ckpt, beam_id, sub=0)
    again = _run_infer(ckpt, data_dir, tmp_path / "infer_out2", beam_id)
    np.testing.assert_array_equal(again["prob"], first["prob"])  # C1 确定性
    np.testing.assert_array_equal(again["rate"], first["rate"])


def test_infer_independent_ckpt(ind_run: dict, data_dir: pathlib.Path,
                                tmp_path: pathlib.Path) -> None:
    """[S5] C1–C3 + `40` [S2] E3 前提：变体由 ckpt config 决定，独立版 ckpt 走同一入口。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    ckpt = ind_run["out_a"] / "best_val.ckpt"
    assert _load_ckpt(ind_run["out_a"])["config"]["variant"] == "independent"
    first = _run_infer(ckpt, data_dir, tmp_path / "infer_ind", beam_id, sub=1)
    _assert_group_n(first, ckpt, beam_id, sub=1)
    again = _run_infer(ckpt, data_dir, tmp_path / "infer_ind2", beam_id, sub=1)
    np.testing.assert_array_equal(again["prob"], first["prob"])


def test_infer_matches_train_forward_variant(joint_run: dict, ind_run: dict,
                                            data_dir: pathlib.Path) -> None:
    """`30` [S5] C1 + `40` [S2] E3：`infer` 的前向与 `train._forward_variant` 逐位一致。

    变体前向有三份实现（训练循环、门禁评估、交付推理）。前两份已有逐位对拍；本用例把
    第三份也钉上：同一 ckpt 在两条路径上必须给出同一数值，否则 E3 跨站推理与训练期
    验证/交付评估之间没有数值可比性。两变体各一例（CPU，float32 主干）。
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    counts, edges = _group_counts(data_dir, beam_id)
    for run in (joint_run, ind_run):
        ckpt = run["out_a"] / "best_val.ckpt"
        payload = _load_ckpt(run["out_a"])
        variant = str(payload["config"]["variant"])
        prob, rate = infer(ckpt, counts, edges, 200_000.0)

        net = build_model(payload["config"])
        net.load_state_dict(payload["state_dict"])
        net.eval()
        batch = {
            "counts": torch.from_numpy(counts.astype(np.float32)).unsqueeze(0),
            "target": torch.zeros(1, 6, 128, 128),
            "edges": torch.from_numpy(edges.astype(np.float32)).unsqueeze(0),
        }
        with torch.no_grad():
            train_prob, _target, _edges = _forward_variant(
                net, variant, batch, torch.device("cpu")
            )
        np.testing.assert_array_equal(
            prob, train_prob[0].numpy().astype(np.float64),
            err_msg=f"{variant}：infer 与 train._forward_variant 前向不一致",
        )
        np.testing.assert_array_equal(rate, prob * 200_000.0)


# ------------------------------------------- [S1] C3：ckpt 载入的 config 哈希/种子校验
def _group_counts(data_dir: pathlib.Path, beam_id: int, sub: int = 0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """train.h5 站 0 的 1e4 档子样本图组（推理入口的最小输入）。"""
    with h5py.File(data_dir / "train.h5", "r") as h5:
        dset = h5[f"beam_{beam_id:04d}/station0/tier_010000_sub_{sub:02d}"]
        return dset[...], np.asarray(dset.attrs["bin_edges"], dtype=np.float64)


def _resave(ckpt: pathlib.Path, dst: pathlib.Path, mutate) -> pathlib.Path:
    """把 ckpt 改一处后另存（篡改/残缺 ckpt 的构造入口，不改动原产物）。"""
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, dst)
    return dst


def test_infer_validates_cfg_hash_and_master_seed(
    joint_run: dict, data_dir: pathlib.Path,
) -> None:
    """[S1] C3：config 哈希恒被校验（内建）；期望值入口正确放行、错误抛 ValueError。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    ckpt = joint_run["out_a"] / "best_val.ckpt"
    counts, edges = _group_counts(data_dir, beam_id)
    payload = _load_ckpt(joint_run["out_a"])
    expected = config_hash(payload["config"])
    assert payload["cfg_hash"] == expected  # 训练产物与重算哈希同源（前置条件）

    prob, rate = infer(ckpt, counts, edges, 200_000.0, expected, SEED)
    assert prob.shape == (6, 128, 128) and (prob > 0).all()
    np.testing.assert_allclose(rate, prob * 200_000.0)

    with pytest.raises(ValueError, match="config 哈希与期望值不符"):
        infer(ckpt, counts, edges, 200_000.0, "0" * 16, SEED)
    with pytest.raises(ValueError, match="master_seed 与期望值不符"):
        infer(ckpt, counts, edges, 200_000.0, expected, SEED + 1)


def test_infer_rejects_tampered_or_incomplete_ckpt(
    joint_run: dict, data_dir: pathlib.Path, tmp_path: pathlib.Path,
) -> None:
    """[S1] C3：config 与 cfg_hash 必须同源——篡改任一侧、缺段落均拒绝加载。"""
    ckpt = joint_run["out_a"] / "best_val.ckpt"
    actual = str(_load_ckpt(joint_run["out_a"])["cfg_hash"])
    counts, edges = _group_counts(
        data_dir, int(json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
                      ["splits"]["train"][0])
    )

    def bump_config(payload: dict) -> None:
        payload["config"]["base"] = int(payload["config"]["base"]) + 1

    def break_hash(payload: dict) -> None:
        payload["cfg_hash"] = "1" * 16 if actual != "1" * 16 else "0" * 16

    cases = (
        (bump_config, "config 哈希不符"),
        (break_hash, "config 哈希不符"),
        (lambda p: p.pop("cfg_hash"), "缺 cfg_hash"),
        (lambda p: p.pop("config"), "缺 config"),
        (lambda p: p.pop("state_dict"), "state_dict"),
    )
    for index, (mutate, match) in enumerate(cases):
        bad = _resave(ckpt, tmp_path / f"bad_{index}.ckpt", mutate)
        with pytest.raises(ValueError, match=match):
            load_ckpt_payload(bad)
    # 推理入口同样拒绝（校验发生在建网与前向之前）
    tampered = _resave(ckpt, tmp_path / "tampered.ckpt", bump_config)
    with pytest.raises(ValueError, match="config 哈希不符"):
        infer(tampered, counts, edges, 200_000.0)


def test_infer_meta_separates_file_and_config_hashes(
    joint_run: dict, data_dir: pathlib.Path, tmp_path: pathlib.Path,
) -> None:
    """[S5] C2 + [S1] C3：meta.json 把文件字节哈希与 ckpt 内 config 哈希分别记在两个字段。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    beam_id = int(manifest["splits"]["train"][0])
    ckpt = joint_run["out_a"] / "best_val.ckpt"
    out = tmp_path / "infer_meta"
    _run_infer(ckpt, data_dir, out, beam_id)

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    payload = _load_ckpt(joint_run["out_a"])
    assert meta["ckpt_sha256_16"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    assert meta["cfg_hash"] == payload["cfg_hash"] == config_hash(payload["config"])
    assert meta["ckpt_md5"] == hashlib.md5(ckpt.read_bytes()).hexdigest()
    assert meta["master_seed"] == SEED
    assert meta["ckpt_sha256_16"] != meta["ckpt_md5"]  # 两种哈希口径互不替代

    # CLI 期望值入口：不匹配时在推理前拒绝，不留半成品产物
    bad_out = tmp_path / "infer_bad_seed"
    with pytest.raises(ValueError, match="master_seed 与期望值不符"):
        infer_main([
            "--ckpt", str(ckpt), "--input", str(data_dir / "train.h5"),
            "--beam-id", str(beam_id), "--station", "0", "--n-low", "10000",
            "--sub", "0", "--n-high", "200000", "--out", str(bad_out),
            "--expect-master-seed", str(SEED + 1),
        ])
    assert not bad_out.exists()
