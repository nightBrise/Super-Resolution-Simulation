"""L4 [S2]/[S3] 判据层测试（spec line2/40）：test 端评估接口、H7/H8 判定、门禁落盘消费。

小配置：3 束数据集（test 切分 1 束）+ 缩配模型（joint base=24 / independent base=16，
num_levels=3、blocks=1），全程 CPU。本文件不训练——ckpt 由 `build_model` 的权重直接落盘，
覆盖的是接口契约与判定表达式的机械执行（`00` GC6：不断言研究方向）。
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest
import torch

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.experiments.evaluate import (
    curve_from_json,
    evaluate_run,
    load_model,
    main as evaluate_main,
    parse_subs,
    predict_group,
)
from src.line2.experiments.gates import (
    OBSERVABLE_NAMES,
    evaluate_gates_from_disk,
    evaluate_h7_cross_station,
    evaluate_h8,
    h8_deviation_matrix,
)
from src.line2.metrics.n_eff import error_n_curve, n_eff_from_err
from src.line2.metrics.tolerance import N_OBSERVABLES
from src.line2.training.train import _forward_variant, build_model
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit

#: `40` [S3] 的合法门禁枚举（字面量，避免与实现常量同源自证）
LEGAL_VERDICTS = {"成立", "方向成立", "不成立", "不可判定"}
N_LOW = 10_000
CURVE_TIERS = (1_000, 10_000, 100_000)
JOINT_CFG = {
    "variant": "joint", "base": 24, "num_levels": 3, "blocks": 1, "seed": 100, "n_low": N_LOW,
}
INDEP_CFG = {
    "variant": "independent", "ind_base": 16, "num_levels": 3, "ind_blocks": 1,
    "seed": 100, "n_low": N_LOW,
}


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """3 束小数据集（切分 2/0/1：test 恰 1 束）；构建 ~13 s。"""
    from tests.unit.test_line2_dataset import _synthetic_table

    out = tmp_path_factory.mktemp("ds_eval")
    anchor = _synthetic_table(100_001, seed=derive_seed(40, 1))
    return build_dataset(out, anchor, n_beams=3, data_version="test_eval")


@pytest.fixture(scope="module")
def test_beam(data_dir: pathlib.Path) -> int:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["splits"]["test"][0])


@pytest.fixture(scope="module")
def curve_path(data_dir: pathlib.Path, test_beam: int,
               tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """E0 曲线 dump（缩配口径：1 束 × 站 0 × 3 档 × k=2）——`curve_from_json` 的输入。"""
    out = tmp_path_factory.mktemp("curve") / "e0_curve.json"
    error_n_curve(
        data_dir, beam_ids=[test_beam], stations=(0,), tiers=CURVE_TIERS, k=2, dump=out
    )
    return out


@pytest.fixture(scope="module")
def group_sample(data_dir: pathlib.Path, test_beam: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(counts, edges, ref) = test 束站 0 的 1e4 档 sub 0 与其 2e5 参照。"""
    with h5py.File(data_dir / "test.h5", "r") as h5:
        stg = h5[f"beam_{test_beam:04d}/station0"]
        counts = stg[f"tier_{N_LOW:06d}_sub_00"][...]
        edges = np.asarray(stg[f"tier_{N_LOW:06d}_sub_00"].attrs["bin_edges"])
        ref = stg["target_full"][...]
    return counts, edges, ref


def _save_ckpt(run_dir: pathlib.Path, cfg: dict, seed: int) -> pathlib.Path:
    """落盘未训练 ckpt（结构由 cfg.variant 决定，与 `train.py` 同一构建入口）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(derive_seed(40, seed))
    net = build_model(cfg)
    ckpt = run_dir / "best_val.ckpt"
    torch.save(
        {"state_dict": net.state_dict(), "config": dict(cfg, seed=seed), "master_seed": seed, "step": 0},
        ckpt,
    )
    return ckpt


def _new_model(cfg: dict, seed: int):
    """按派生种子构建 eval 模式模型（与 `_save_ckpt` 同一种子链，权重可复现）。"""
    torch.manual_seed(derive_seed(40, seed))
    return build_model(cfg).eval()


# --------------------------------------------------- predict_group / load_model
def test_predict_group_both_variants_probability_and_determinism(
    group_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`30` [S1] C1/C3 + `40` [S3] C1：两变体前向跑通、逐投影 Σp=1、重复调用逐位一致。"""
    counts, edges, _ref = group_sample
    for cfg, seed in ((JOINT_CFG, 11), (INDEP_CFG, 12)):
        model = _new_model(cfg, seed)
        prob = predict_group(model, cfg, counts, edges, N_LOW, "cpu")
        assert prob.shape == (6, 128, 128) and prob.dtype == np.float64
        assert np.isfinite(prob).all() and (prob >= 0).all()
        np.testing.assert_allclose(prob.sum(axis=(1, 2)), np.ones(6), rtol=1e-5)
        np.testing.assert_array_equal(prob, predict_group(model, cfg, counts, edges, N_LOW, "cpu"))


def test_predict_group_restores_training_mode(
    group_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """推理走 eval 模式（`40` [S3] C1 确定性），且不给调用方留下训练态副作用。"""
    counts, edges, _ref = group_sample
    model = _new_model(JOINT_CFG, 13)
    model.train()
    predict_group(model, JOINT_CFG, counts, edges, N_LOW, "cpu")
    assert model.training is True
    model.eval()
    predict_group(model, JOINT_CFG, counts, edges, N_LOW, "cpu")
    assert model.training is False


def test_predict_group_matches_training_forward(
    group_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """test 端前向 = 训练端 `_forward_variant`（同权重同输入逐位一致，κ 仅由输入派生）。"""
    counts, edges, _ref = group_sample
    for cfg, seed in ((JOINT_CFG, 14), (INDEP_CFG, 15)):
        model = _new_model(cfg, seed)
        batch = {
            "counts": torch.from_numpy(counts.astype(np.float32)).unsqueeze(0),
            "target": torch.zeros(1, 6, 128, 128),
            "edges": torch.from_numpy(edges.astype(np.float32)).unsqueeze(0),
        }
        expected, _target, _edges = _forward_variant(model, cfg["variant"], batch, torch.device("cpu"))
        got = predict_group(model, cfg, counts, edges, N_LOW, "cpu")
        np.testing.assert_array_equal(got, expected[0].detach().numpy().astype(np.float64))


def test_predict_group_rejects_unknown_variant(
    group_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """未知变体显式报错（`30` [S4] F1 只有 joint / independent 两臂），不静默走 joint。"""
    counts, edges, _ref = group_sample
    model = _new_model(JOINT_CFG, 16)
    with pytest.raises(ValueError, match="variant"):
        predict_group(model, {"variant": "hybrid"}, counts, edges, N_LOW, "cpu")


def test_load_model_roundtrip(tmp_path: pathlib.Path,
                              group_sample: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """`40` [S3] C2：模型只从落盘 ckpt 复原（eval 模式 + config 决定变体）。"""
    counts, edges, _ref = group_sample
    ckpt = _save_ckpt(tmp_path / "e1_indep_s0", INDEP_CFG, seed=101)
    model, cfg = load_model(ckpt)
    assert model.training is False and cfg["variant"] == "independent"
    fresh = _new_model(INDEP_CFG, 101)  # 同派生种子 → 同权重
    np.testing.assert_array_equal(
        predict_group(model, cfg, counts, edges, N_LOW, "cpu"),
        predict_group(fresh, INDEP_CFG, counts, edges, N_LOW, "cpu"),
    )
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.ckpt")
    bad = tmp_path / "bad.ckpt"
    torch.save({"config": INDEP_CFG}, bad)
    with pytest.raises(ValueError, match="state_dict"):
        load_model(bad)


# ------------------------------------------------------------------ evaluate_run
def test_evaluate_run_payload_schema_and_determinism(
    data_dir: pathlib.Path, curve_path: pathlib.Path, tmp_path: pathlib.Path, test_beam: int
) -> None:
    """`40` [S3] C1/C2 + `20` [S3] C1/C2 + `20` [S4]：冻结字段、N_eff 域与标记、逐位可复算。"""
    run_dir = tmp_path / "e1_joint_s0"
    ckpt = _save_ckpt(run_dir, JOINT_CFG, seed=100)
    subs = [0, 1]
    payload = evaluate_run(ckpt, data_dir, "test", 0, N_LOW, subs, curve_path)

    assert set(payload) == {
        "run_id", "variant", "seed", "split", "station", "n_low", "subs", "curve",
        "n_beams", "n_groups", "err_median", "n_eff", "n_eff_flag", "per_beam", "code_version",
    }
    assert payload["run_id"] == "e1_joint_s0" and payload["variant"] == "joint"
    assert payload["seed"] == 100 and payload["split"] == "test"
    assert payload["station"] == 0 and payload["n_low"] == N_LOW and payload["subs"] == subs
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["n_beams"] == len(manifest["splits"]["test"]) == 1
    assert payload["n_groups"] == payload["n_beams"] * len(subs)
    assert len(payload["code_version"]) == 40

    curve_dump = json.loads(curve_path.read_text(encoding="utf-8"))
    assert payload["curve"]["path"] == str(curve_path)
    assert payload["curve"]["beams_scope"] == "test"  # 无 meta 侧文件 → 由 manifest 推断
    assert payload["curve"]["reference_noise_floor"] == curve_dump["reference_noise_floor"]

    beam = payload["per_beam"][0]
    assert set(beam) == {"beam_id", "err", "n_eff", "n_eff_flag", "obs", "n_eff_median"}
    assert beam["beam_id"] == test_beam
    assert len(beam["obs"]) == N_OBSERVABLES and np.isfinite(beam["obs"]).all()
    assert beam["n_eff_flag"] in (None, "underflow", "overflow")
    assert 1e3 <= beam["n_eff"] <= 1e5 and 1e3 <= beam["n_eff_median"] <= 1e5

    # N_eff 由 err 唯一决定（`20` [S3]）；run 级 = 束级 err 的中位数再映射
    errs = np.asarray([b["err"] for b in payload["per_beam"]], dtype=np.float64)
    assert payload["err_median"] == float(np.median(errs))
    curve = curve_from_json(curve_path)
    expected_n_eff, expected_flag = n_eff_from_err(payload["err_median"], curve)
    assert payload["n_eff"] == expected_n_eff and payload["n_eff_flag"] == expected_flag

    # 落盘 + 同输入两次调用逐位一致（`40` [S3] C1）
    out_file = run_dir / "test_eval.json"
    assert out_file.is_file()
    assert json.loads(out_file.read_text(encoding="utf-8")) == payload
    first = out_file.read_bytes()
    again = evaluate_run(ckpt, data_dir, "test", 0, N_LOW, subs, curve_path)
    assert out_file.read_bytes() == first and again == payload


def test_evaluate_run_uses_requested_station_data(
    data_dir: pathlib.Path, curve_path: pathlib.Path, tmp_path: pathlib.Path, test_beam: int
) -> None:
    """`40` [S2] C1/C3：站 1 评估用站 1 自己的 bin 边界与参照（不跨站套用站 0）。

    以 h5 中该站的 counts/edges/target_full 独立复算 M1，与产物逐位比对——若实现误用
    站 0 的边界或参照，两站传输后的图组不同，复算值必然对不上。
    """
    from src.line2.metrics.metrics import group_rel_rmse

    ckpt = _save_ckpt(tmp_path / "e3_joint_s0", JOINT_CFG, seed=100)
    payload = evaluate_run(ckpt, data_dir, "test", 1, N_LOW, [0], curve_path)
    assert payload["station"] == 1
    with h5py.File(data_dir / "test.h5", "r") as h5:
        stg = h5[f"beam_{test_beam:04d}/station1"]
        counts = stg[f"tier_{N_LOW:06d}_sub_00"][...]
        edges = np.asarray(stg[f"tier_{N_LOW:06d}_sub_00"].attrs["bin_edges"])
        ref = stg["target_full"][...].astype(np.float64)
        ref_s0 = h5[f"beam_{test_beam:04d}/station0"]["target_full"][...].astype(np.float64)
    model, cfg = load_model(ckpt)
    prob = predict_group(model, cfg, counts, edges, N_LOW, "cpu")
    assert payload["per_beam"][0]["err"] == group_rel_rmse(prob, ref)
    assert group_rel_rmse(prob, ref) != group_rel_rmse(prob, ref_s0)  # 两站参照可区分


def test_evaluate_run_rejects_unknown_and_empty_split(
    data_dir: pathlib.Path, curve_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """切分不存在 / 切分为空（3 束时 val 取整为 0）均报错，不静默产出空评估。"""
    ckpt = _save_ckpt(tmp_path / "e1_joint_s0", JOINT_CFG, seed=100)
    with pytest.raises(ValueError, match="不在 manifest"):
        evaluate_run(ckpt, data_dir, "nope", 0, N_LOW, [0], curve_path)
    assert json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["splits"]["val"] == []
    with pytest.raises(ValueError, match="为空"):
        evaluate_run(ckpt, data_dir, "val", 0, N_LOW, [0], curve_path)


def test_evaluate_run_cli_writes_out(tmp_path: pathlib.Path, data_dir: pathlib.Path,
                                     curve_path: pathlib.Path) -> None:
    """`40` [S1] 调度入口（CLI）：--subs a-b 解析；--out 重定向落盘（E3 跨站不相覆）。"""
    ckpt = _save_ckpt(tmp_path / "e1_joint_s0", JOINT_CFG, seed=100)
    out = tmp_path / "copy" / "test_eval_station1.json"
    rc = evaluate_main([
        "--ckpt", str(ckpt), "--data-dir", str(data_dir), "--split", "test",
        "--station", "0", "--n-low", str(N_LOW), "--subs", "0-1",
        "--curve", str(curve_path), "--out", str(out),
    ])
    assert rc == 0 and out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["subs"] == [0, 1] and payload["n_groups"] == 2
    # 给了 --out 就只写 --out：run 目录默认名留给站 0 产物，不被站 1 评估覆盖
    assert not (ckpt.parent / "test_eval.json").exists()


def test_parse_subs_forms() -> None:
    """--subs 的三种写法与默认全档（`20` [S2] k=20）。"""
    assert parse_subs("0-19") == list(range(20))
    assert parse_subs(None) == list(range(20))
    assert parse_subs(3) == [3]
    assert parse_subs("3,1,3") == [1, 3]
    assert parse_subs([2, 0]) == [2, 0]
    with pytest.raises(ValueError):
        parse_subs("5-3")


# ------------------------------------------------------------------ curve 往返
def test_curve_from_json_roundtrip(data_dir: pathlib.Path, curve_path: pathlib.Path,
                                   test_beam: int) -> None:
    """`20` [S2] C1 + [S3] C1：dump → 读回 → N_eff 反解逐位一致（含上/下溢分支）。"""
    original = error_n_curve(
        data_dir, beam_ids=[test_beam], stations=(0,), tiers=CURVE_TIERS, k=2
    )
    loaded = curve_from_json(curve_path)
    assert loaded.points == original.points
    assert loaded.mutual_floor == original.mutual_floor
    assert loaded.reference_noise_floor == original.reference_noise_floor
    assert loaded.beam_ids == (test_beam,) and loaded.tiers == CURVE_TIERS and loaded.k == 2
    medians = [p.median for p in original.points]
    probes = [
        medians[0] + 1.0,                     # 高于最低档 → 下溢
        medians[-1] - 1e-6,                   # 低于最高档 → 上溢
        medians[1],                           # 恰在节点上
        0.5 * (medians[0] + medians[1]),      # 档间插值
        0.5 * (medians[1] + medians[2]),
    ]
    assert medians[0] > medians[1] > medians[2]  # 曲线单调（插值前提）
    for err in probes:
        assert n_eff_from_err(err, loaded) == n_eff_from_err(err, original)
    flags = {n_eff_from_err(err, loaded)[1] for err in probes}
    assert {"underflow", "overflow", None} <= flags


def test_curve_from_json_rejects_bad_input(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        curve_from_json(tmp_path / "missing.json")
    bad = tmp_path / "not_a_curve.json"
    bad.write_text(json.dumps({"tiers": [1000]}), encoding="utf-8")
    with pytest.raises(ValueError, match="points"):
        curve_from_json(bad)


# ------------------------------------------------------------------------- H8
def test_evaluate_h8_threshold_boundaries() -> None:
    """`40` [S3] H8：达标率恰 0.8 → 成立、恰 0.5 → 不成立、之间 → 不可判定；枚举合法。"""
    tol = np.full((10, 13), 0.6)
    cases = {104: "成立", 65: "不成立", 100: "不可判定", 105: "成立", 64: "不成立"}
    for n_pass, expected in cases.items():
        dev = np.ones((10, 13))  # 1.0 > 0.6 → 不达标
        dev.ravel()[:n_pass] = 0.5
        result = evaluate_h8(dev, tol)
        assert result["pass_rate"] == pytest.approx(n_pass / 130.0, rel=1e-12)
        assert result["verdict"] == expected
        assert result["verdict"] in LEGAL_VERDICTS
        assert len(result["per_observable"]) == N_OBSERVABLES
        assert sum(o["n_pass"] for o in result["per_observable"]) == n_pass
    # 闭区间：偏差恰等于容差线算达标（容差 = 95% 分位）
    assert evaluate_h8(np.full((4, 13), 0.6), np.full((4, 13), 0.6))["pass_rate"] == 1.0


def test_evaluate_h8_input_contracts() -> None:
    """容差可广播为 (13,)；形状/空输入按契约报错，不给悬空判定。"""
    dev = np.zeros((3, N_OBSERVABLES))
    assert evaluate_h8(dev, np.ones(N_OBSERVABLES))["verdict"] == "成立"
    with pytest.raises(ValueError):
        evaluate_h8(np.zeros((3, 12)), np.ones((3, 12)))
    with pytest.raises(ValueError):
        evaluate_h8(np.zeros((3, N_OBSERVABLES)), np.ones((4, N_OBSERVABLES)))
    with pytest.raises(ValueError):
        evaluate_h8(np.zeros((0, N_OBSERVABLES)), np.ones((0, N_OBSERVABLES)))
    assert len(OBSERVABLE_NAMES) == N_OBSERVABLES


def test_h8_deviation_matrix_median_over_seeds() -> None:
    """`20` [S4]：逐观测量**先对 5 种子取中位数**再判达标（不是逐种子计票）。

    反例：3/5 种子各自的达标率 ≥0.8，但种子中位数矩阵的达标率 <0.8 →
    中位数口径判"不可判定"，逐种子计票口径会判"成立"。
    """
    tol_vec = np.full(N_OBSERVABLES, 0.5)
    fail_entries = {0: (0, 3, 4), 1: (1, 3, 4), 2: (2, 3, 4)}  # 条目 → 不达标的种子
    runs = []
    for seed in range(5):
        obs = np.full(N_OBSERVABLES, 0.1)
        for entry, seeds in fail_entries.items():
            if seed in seeds:
                obs[entry] = 0.9
        runs.append({"run_id": f"r{seed}", "per_beam": [{"beam_id": 7, "obs": obs.tolist()}]})

    matrix = h8_deviation_matrix(runs, [7])
    assert matrix.shape == (1, N_OBSERVABLES)
    stacked = np.stack([np.asarray(r["per_beam"][0]["obs"])[None, :] for r in runs])
    np.testing.assert_array_equal(matrix, np.median(stacked, axis=0))

    per_seed_verdicts = [
        evaluate_h8(np.asarray(runs[s]["per_beam"][0]["obs"])[None, :], tol_vec)["verdict"]
        for s in range(5)
    ]
    assert per_seed_verdicts.count("成立") == 3          # 3/5 种子逐种子口径达标
    median_result = evaluate_h8(matrix, tol_vec)
    assert median_result["pass_rate"] == pytest.approx(10.0 / 13.0)
    assert median_result["verdict"] == "不可判定"          # 中位数口径结论不同


def test_h8_deviation_matrix_is_median_not_mean_and_follows_beam_order() -> None:
    """中位数语义（离群种子不拉动结果）+ 行序 = beam_ids 给定顺序。"""
    runs = [
        {"run_id": "a", "per_beam": [
            {"beam_id": 3, "obs": [0.1] * N_OBSERVABLES},
            {"beam_id": 7, "obs": [100.0] * N_OBSERVABLES}]},
        {"run_id": "b", "per_beam": [
            {"beam_id": 3, "obs": [0.1] * N_OBSERVABLES},
            {"beam_id": 7, "obs": [100.0] * N_OBSERVABLES}]},
        {"run_id": "c", "per_beam": [
            {"beam_id": 3, "obs": [0.1] * N_OBSERVABLES},
            {"beam_id": 7, "obs": [9.0] * N_OBSERVABLES}]},
    ]
    matrix = h8_deviation_matrix(runs, [3, 7])
    assert matrix.shape == (2, N_OBSERVABLES)
    np.testing.assert_allclose(matrix[0], 0.1)
    np.testing.assert_allclose(matrix[1], 100.0)          # 中位数，非均值 (209/3)
    swapped = h8_deviation_matrix(runs, [7, 3])
    np.testing.assert_allclose(swapped[0], 100.0)
    np.testing.assert_allclose(swapped[1], 0.1)


def test_h8_deviation_matrix_rejects_incomplete_runs() -> None:
    with pytest.raises(ValueError, match="无 run"):
        h8_deviation_matrix([], [1])
    with pytest.raises(ValueError, match="beam_ids"):
        h8_deviation_matrix([{"run_id": "a", "per_beam": []}], [])
    with pytest.raises(ValueError, match="缺 beam"):
        h8_deviation_matrix(
            [{"run_id": "a", "per_beam": [{"beam_id": 1, "obs": [0.1] * N_OBSERVABLES}]}], [1, 2]
        )
    with pytest.raises(ValueError, match="观测量维度"):
        h8_deviation_matrix(
            [{"run_id": "a", "per_beam": [{"beam_id": 1, "obs": [0.1] * 12}]}], [1]
        )


# ------------------------------------------------------------------------- H7
def test_evaluate_h7_cross_station_all_branches() -> None:
    """`40` [S2] C2：ρ 对边界 {1.0, 0.5} 的三分类（含"成立（弱）"的衰减系数）。"""
    strong = evaluate_h7_cross_station(1e4, 2e4)
    assert strong["verdict"] == "成立" and strong["weak"] is False and strong["rho"] == 2.0
    boundary = evaluate_h7_cross_station(2e4, 2e4)          # ρ = 1 恰在强边界
    assert boundary["verdict"] == "成立" and boundary["weak"] is False and boundary["rho"] == 1.0
    weak = evaluate_h7_cross_station(1e4, 5e3)              # ρ = 0.5 恰在弱边界
    assert weak["verdict"] == "成立" and weak["weak"] is True and weak["decay"] == 0.5
    assert evaluate_h7_cross_station(1e4, 4999.0)["verdict"] == "不成立"
    for result in (strong, boundary, weak):
        assert result["verdict"] in LEGAL_VERDICTS


def test_evaluate_h7_flags_do_not_block_ratio() -> None:
    """上/下溢只表示 N_eff 触及报告域端点（1e3/1e5），端点值有限 → ρ 仍可算。

    `40` [S2] C2 要求结论由 ρ 与边界唯一决定，故标记不增设不可判定分支，只随结论
    输出供报告披露饱和情形（双上溢 = 双域顶 → ρ = 1）。
    """
    both_over = evaluate_h7_cross_station((1e5, "overflow"), (1e5, "overflow"))
    assert both_over["verdict"] == "成立" and both_over["rho"] == 1.0
    assert both_over["flag_station0"] == "overflow" and both_over["flag_station1"] == "overflow"
    both_under = evaluate_h7_cross_station((1e3, "underflow"), (1e3, "underflow"))
    assert both_under["verdict"] == "成立" and both_under["rho"] == 1.0
    asymmetric = evaluate_h7_cross_station((1e5, "overflow"), (3e4, None))
    assert asymmetric["verdict"] == "不成立" and asymmetric["rho"] == pytest.approx(0.3)
    dict_form = evaluate_h7_cross_station(
        {"n_eff": 1e4, "n_eff_flag": None}, {"n_eff": 1e5, "n_eff_flag": "overflow"}
    )
    assert dict_form["verdict"] == "成立" and dict_form["flag_station1"] == "overflow"


def test_evaluate_h7_undecidable_when_ratio_undefined() -> None:
    """ρ 真的无法计算（缺失 / 非有限 / 除零）→ 不可判定，且不伪造 ρ 数值。"""
    for s0, s1 in ((None, 1e4), (1e4, None), (float("nan"), 1e4), (1e4, float("nan")),
                   (0.0, 1e4), (-1.0, 1e4)):
        result = evaluate_h7_cross_station(s0, s1)
        assert result["verdict"] == "不可判定", (s0, s1)
        assert result["rho"] is None and result["weak"] is False


# ------------------------------------------------- evaluate_gates_from_disk
def _write_run(results_dir: pathlib.Path, run_id: str, variant: str, seed: int,
               n_eff: float, val_n_eff: float, station: int = 0, n_low: int = N_LOW,
               split: str = "test", obs: float = 0.4,
               beam_ids: tuple[int, ...] = (1, 2, 3), eval_name: str = "test_eval.json") -> None:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_beam = [
        {"beam_id": b, "err": 1.0, "n_eff": n_eff, "n_eff_flag": None,
         "obs": [obs] * N_OBSERVABLES, "n_eff_median": n_eff}
        for b in beam_ids
    ]
    payload = {
        "run_id": run_id, "variant": variant, "seed": seed, "split": split, "station": station,
        "n_low": n_low, "subs": [0, 1],
        "curve": {"path": "e0_curve.json", "beams_scope": "valtest",
                  "reference_noise_floor": 0.28},
        "n_beams": len(beam_ids), "n_groups": 2 * len(beam_ids),
        "err_median": 1.0, "n_eff": n_eff, "n_eff_flag": None,
        "per_beam": per_beam, "code_version": "0" * 40,
    }
    (run_dir / eval_name).write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"variant": variant, "seed": seed, "best_val_n_eff": val_n_eff}),
        encoding="utf-8",
    )


def _write_curve_and_tolerance(results: pathlib.Path) -> None:
    """写门禁所需的曲线 dump（+ meta 侧文件）与 `20` [S4] 容差线产物。"""
    curve = {
        "tiers": list(CURVE_TIERS), "k": 2, "beam_ids": [1, 2, 3],
        "points": [
            {"n_low": 1_000, "median": 4.0, "iqr_lo": 3.9, "iqr_hi": 4.1, "n_samples": 4},
            {"n_low": 10_000, "median": 1.2, "iqr_lo": 1.1, "iqr_hi": 1.3, "n_samples": 4},
            {"n_low": 100_000, "median": 0.1, "iqr_lo": 0.09, "iqr_hi": 0.11, "n_samples": 4},
        ],
        "mutual_floor": {"1000": 5.0, "10000": 1.5, "100000": 0.2},
        "reference_noise_floor": 0.14,
    }
    (results / "e0_curve.json").write_text(json.dumps(curve), encoding="utf-8")
    (results / "e0_curve.meta.json").write_text(
        json.dumps({"beams_scope": "valtest"}), encoding="utf-8"
    )
    (results / "tolerance.json").write_text(
        json.dumps({"quantile": 0.95, "scope": "valtest_pooled",
                    "dev_table": {str(t): [0.5] * N_OBSERVABLES for t in CURVE_TIERS}}),
        encoding="utf-8",
    )


@pytest.fixture()
def fake_results(tmp_path: pathlib.Path) -> pathlib.Path:
    """假 run 目录：5 种子 × 2 变体（站 0）+ 站 1 跨站评估 + 曲线/容差产物。

    val 与 test 段刻意给出**相反的变体优劣**（val 独立更强、test 联合更强），
    以检验 [S3] 两段式消费边界：路由只吃 val，H1/H3/H8 只吃 test。
    """
    results = tmp_path / "results"
    results.mkdir(parents=True)
    _write_curve_and_tolerance(results)
    joint_test = [40_000.0, 41_000.0, 39_000.0, 42_000.0, 38_000.0]
    indep_test = [30_000.0, 31_000.0, 29_000.0, 32_000.0, 28_000.0]
    for slot in range(5):
        _write_run(results, f"e1_joint_s{slot}", "joint", slot, joint_test[slot], 5_000.0)
        _write_run(results, f"e1_indep_s{slot}", "independent", 100 + slot,
                   indep_test[slot], 20_000.0)
    # E3：获胜变体（val 段裁定 = independent）与 joint 各一份站 1 评估
    _write_run(results, "e3_indep_s0_st1", "independent", 100, 20_000.0, 20_000.0,
               station=1, eval_name="test_eval_station1.json")
    _write_run(results, "e3_joint_s0_st1", "joint", 0, 12_000.0, 5_000.0,
               station=1, eval_name="test_eval_station1.json")
    return results


def test_evaluate_gates_from_disk_verdicts_and_two_segment_boundary(
    fake_results: pathlib.Path,
) -> None:
    """`40` [S3] C1–C4：只读落盘产物出四条门禁、枚举合法、val/test 两段不交叉。"""
    out = evaluate_gates_from_disk(fake_results)
    assert (fake_results / "e1_gates.json").is_file()
    assert json.loads((fake_results / "e1_gates.json").read_text(encoding="utf-8")) == out

    # 路由只吃 val 段：val 上独立更强 → 获胜变体 = independent（尽管 test 上联合更强）
    assert out["winner_variant"] == "independent"
    assert out["routing_basis"] == "val"
    assert out["val_segment"]["n_pairs"] == 5
    assert out["val_segment"]["per_seed_n_eff"]["joint"]["0"] == 5_000.0
    # H1 只吃获胜变体的 test 段（median 30000 ≥ 2×1e4 且 5/5 > 1e4）
    assert out["H1"]["verdict"] == "成立" and out["H1"]["variant"] == "independent"
    assert out["H1"]["median_n_eff"] == 30_000.0 and out["H1"]["n_seeds"] == 5
    assert set(out["H1"]["inputs"]) == {"100", "101", "102", "103", "104"}
    # H3 = test 段同槽位成对 5/5 → 成立；配对表落盘可审计（两臂种子数值互不重叠）
    assert out["H3"]["verdict"] == "成立" and out["H3"]["joint_wins"] == 5
    assert out["H3"]["pairing"]["pairs"] == [[s, 100 + s] for s in range(5)]
    # H7 = 获胜变体的站 1/站 0 比：20000/30000 → 成立（弱），衰减系数 = ρ
    assert out["H7"]["variant"] == "independent"
    assert out["H7"]["verdict"] == "成立" and out["H7"]["weak"] is True
    assert out["H7"]["rho"] == pytest.approx(2.0 / 3.0)
    # H8 = 获胜变体的种子中位数偏差 (0.4) 对容差 (0.5) → 全达标
    assert out["H8"]["verdict"] == "成立" and out["H8"]["pass_rate"] == 1.0
    assert out["H8"]["n_beams"] == 3 and out["H8"]["beam_ids"] == [1, 2, 3]
    assert out["H8"]["variant"] == "independent" and out["H8"]["n_seeds"] == 5
    assert len(out["H8"]["per_observable"]) == N_OBSERVABLES

    for key in ("H1", "H3", "H7", "H8"):
        assert out[key]["verdict"] in LEGAL_VERDICTS
    # 快照与溯源
    assert set(out["beam_n_eff_snapshot"]) == {"joint", "independent"}
    snapshot = out["beam_n_eff_snapshot"]["independent"][0]
    assert snapshot["beam_id"] == 1 and snapshot["n_eff_median"] == 30_000.0
    assert set(snapshot["n_eff_per_seed"]) == {"100", "101", "102", "103", "104"}
    assert out["gate_dims"] == {"split": "test", "station": 0, "n_low": N_LOW, "n_seeds_expected": 5}
    assert out["curve"]["beams_scope"] == "valtest"
    assert out["curve"]["reference_noise_floor"] == 0.14
    assert len(out["code_version"]) == 40
    # test 段消费面只列 test 产物（站 0 进门禁、站 1 进 H7）
    assert len(out["test_segment"]["gate_inputs"]) == 10
    assert len(out["test_segment"]["cross_station_inputs"]) == 2


def test_evaluate_gates_from_disk_bitwise_reproducible(fake_results: pathlib.Path) -> None:
    """`40` [S3] C1：同输入两次调用产物逐位一致。"""
    evaluate_gates_from_disk(fake_results)
    first = (fake_results / "e1_gates.json").read_bytes()
    evaluate_gates_from_disk(fake_results)
    assert (fake_results / "e1_gates.json").read_bytes() == first


def test_evaluate_gates_rejects_non_test_split(fake_results: pathlib.Path) -> None:
    """`40` [S3] C2 / `00` GC3：门禁消费的 eval 文件 split 必须为 test。"""
    victim = fake_results / "e1_joint_s0" / "test_eval.json"
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["split"] = "val"
    victim.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="只消费 test 段"):
        evaluate_gates_from_disk(fake_results)


def test_evaluate_gates_rejects_duplicate_test_eval(fake_results: pathlib.Path) -> None:
    """同一（变体, 种子, 站 0, 1e4）出现两份 test 评估 → 报错（test 只评一次）。"""
    _write_run(fake_results, "e1_joint_s0_dup", "joint", 0, 40_000.0, 5_000.0)
    with pytest.raises(ValueError, match="test 只评一次"):
        evaluate_gates_from_disk(fake_results)


def test_evaluate_gates_missing_artifacts(tmp_path: pathlib.Path) -> None:
    """缺产物即报错，不出"空门禁"：results 目录 / test_eval / e0_curve 三档。"""
    with pytest.raises(FileNotFoundError):
        evaluate_gates_from_disk(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="test_eval"):
        evaluate_gates_from_disk(empty)
    no_curve = tmp_path / "no_curve"
    _write_run(no_curve, "e1_joint_s0", "joint", 0, 40_000.0, 5_000.0)
    with pytest.raises(FileNotFoundError, match="e0_curve.json"):
        evaluate_gates_from_disk(no_curve)
    assert not (no_curve / "e1_gates.json").exists()  # 报错时不留半成品产物


def test_evaluate_gates_h8_degrades_without_tolerance(fake_results: pathlib.Path) -> None:
    """容差线产物缺失 → H8 判不可判定并记 reason_code（不现场猜容差）。"""
    (fake_results / "tolerance.json").unlink()
    out = evaluate_gates_from_disk(fake_results)
    assert out["H8"]["verdict"] == "不可判定"
    assert out["H8"]["reason_code"] == "tolerance_missing" and out["H8"]["pass_rate"] is None
    assert out["H1"]["verdict"] == "成立"  # 其余门禁不受影响


def test_evaluate_gates_partial_seeds_are_undecidable(fake_results: pathlib.Path) -> None:
    """种子数不足 5（E1 未跑完）→ H1/H3 判不可判定并记 n_seeds/n_pairs，路由不成立。"""
    for slot in (3, 4):
        for name in (f"e1_joint_s{slot}", f"e1_indep_s{slot}"):
            (fake_results / name / "test_eval.json").unlink()
            (fake_results / name / "summary.json").unlink()
    out = evaluate_gates_from_disk(fake_results)
    assert out["winner_variant"] == "indeterminate" and out["routing_basis"] == "val_insufficient"
    assert out["H1"]["verdict"] == "不可判定" and out["H1"]["n_seeds"] == 0
    assert out["H3"]["verdict"] == "不可判定" and out["H3"]["n_pairs"] == 3
    assert out["H8"]["verdict"] == "不可判定" and out["H8"]["reason_code"] == "no_runs"
    for key in ("H1", "H3", "H7", "H8"):
        assert out[key]["verdict"] in LEGAL_VERDICTS


@pytest.mark.parametrize(
    "joint_val,indep_val,expected",
    [
        ([30_000.0] * 5, [10_000.0] * 5, "joint"),                            # 5/5
        ([30_000.0] * 4 + [1_000.0], [10_000.0] * 5, "joint"),                # 4/5（弱）
        ([5_000.0, 5_000.0, 200_000.0, 1_000.0, 1_000.0],
         [5_000.0, 5_000.0, 200_000.0, 2_000.0, 2_000.0], "joint"),           # 3/5 → 中位数同分取联合
        ([5_000.0, 5_000.0, 200_000.0, 1_000.0, 1_000.0],
         [5_000.0, 5_000.0, 200_000.0, 300_000.0, 300_000.0], "independent"),  # 3/5 → 中位数翻转
        ([1_000.0] * 3 + [30_000.0] * 2, [20_000.0] * 5, "independent"),      # ≤2/5
    ],
)
def test_evaluate_gates_variant_routing_all_branches(
    tmp_path: pathlib.Path, joint_val: list[float], indep_val: list[float], expected: str
) -> None:
    """`40` [S3] C4：路由覆盖 H3 计数的全部四种分支（含 3/5 的 val 中位数裁定）。

    test 段恒为联合更强，故 H3/H1 的结论与路由结果解耦——两段互不干涉即在此断言。
    """
    results = tmp_path / "results"
    results.mkdir(parents=True)
    _write_curve_and_tolerance(results)
    for slot in range(5):
        _write_run(results, f"e1_joint_s{slot}", "joint", slot, 40_000.0, joint_val[slot])
        _write_run(results, f"e1_indep_s{slot}", "independent", 100 + slot,
                   30_000.0, indep_val[slot])
    out = evaluate_gates_from_disk(results)
    assert out["winner_variant"] == expected
    assert out["routing_basis"] == "val" and out["val_segment"]["n_pairs"] == 5
    assert out["H3"]["verdict"] == "成立" and out["H3"]["joint_wins"] == 5
    assert out["H1"]["variant"] == expected and out["H1"]["verdict"] == "成立"
    assert out["H8"]["variant"] == expected


def test_gates_consume_real_evaluate_run_output(
    data_dir: pathlib.Path, curve_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """端到端交接口径：门禁直接吃 `evaluate_run` 的真实产物（不训练、不重推理）。

    假 JSON 只能证明门禁自洽；本用例证明 test_eval.json 的字段与类型确实可被门禁消费
    （`40` [S3] C1/C2）。结论不做方向断言（`00` GC6），只断言枚举合法与结构完整。
    """
    results = tmp_path / "results"
    results.mkdir(parents=True)
    (results / "e0_curve.json").write_text(
        curve_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (results / "tolerance.json").write_text(
        json.dumps({"quantile": 0.95, "scope": "valtest_pooled",
                    "dev_table": {str(t): [0.5] * N_OBSERVABLES for t in CURVE_TIERS}}),
        encoding="utf-8",
    )
    arms = (("joint", "joint", JOINT_CFG, 5_000.0),
            ("indep", "independent", INDEP_CFG, 20_000.0))
    for slot in range(5):
        for tag, variant, cfg, val in arms:
            # 种子编号与 E1 冻结配置同构：joint 0–4 / independent 100–104（槽位配对）
            seed = slot if variant == "joint" else 100 + slot
            run_dir = results / f"e1_{tag}_s{slot}"
            evaluate_run(_save_ckpt(run_dir, cfg, seed), data_dir, "test", 0, N_LOW,
                         [0, 1], curve_path)
            (run_dir / "summary.json").write_text(
                json.dumps({"variant": variant, "seed": seed, "best_val_n_eff": val}),
                encoding="utf-8",
            )

    out = evaluate_gates_from_disk(results)
    assert out["winner_variant"] == "independent"  # 路由只吃 val 段
    assert out["H1"]["n_seeds"] == 5 and out["H3"]["n_pairs"] == 5
    assert out["H3"]["pairing"]["pairs"] == [[s, 100 + s] for s in range(5)]
    assert out["H8"]["n_beams"] == 1 and out["H8"]["n_seeds"] == 5
    assert out["H8"]["pass_rate"] is not None and 0.0 <= out["H8"]["pass_rate"] <= 1.0
    assert len(out["H8"]["per_observable"]) == N_OBSERVABLES
    assert out["H7"]["verdict"] == "不可判定" and out["H7"]["reason_code"] == "station1_missing"
    for key in ("H1", "H3", "H7", "H8"):
        assert out[key]["verdict"] in LEGAL_VERDICTS
    assert len(out["beam_n_eff_snapshot"]["joint"]) == 1
    first = (results / "e1_gates.json").read_bytes()
    evaluate_gates_from_disk(results)
    assert (results / "e1_gates.json").read_bytes() == first
