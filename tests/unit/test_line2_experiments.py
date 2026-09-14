"""L4 E1 门禁评估与 [S4] 产出物脚本的测试（spec line2/40）。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.line2.experiments import evaluate, run_e1
from src.line2.experiments.gates import (
    evaluate_e1_gates,
    evaluate_h1,
    evaluate_h3,
    pair_by_seed,
    winner_variant,
)
from src.line2.experiments.map_report import (
    EVAL_GLOB,
    TABLE_COLUMNS,
    check_e0_precedes_runs,
    collect_n_eff_table,
    generate,
    run_dir_triple,
    write_n_eff_table,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- H1
def test_h1_positive_strong() -> None:
    """5 种子全部 N_eff > 1e4 且中位数 ≥ 2×1e4 → 成立。"""
    values = [22000.0, 25000.0, 18000.0, 30000.0, 27000.0]
    assert evaluate_h1(values) == "成立"


def test_h1_negative_median_low() -> None:
    """中位数 ≤ 1e4 → 不成立。"""
    values = [1500.0, 800.0, 2000.0, 500.0, 1000.0]
    assert evaluate_h1(values) == "不成立"


def test_h1_indeterminate() -> None:
    """5/5 > 1e4 但中位数 < 2e4 → 不可判定。"""
    values = [12000.0, 11000.0, 13000.0, 15000.0, 10000.0]
    assert evaluate_h1(values) == "不可判定"


# ---------------------------------------------------------------- H3
def test_h3_5_5() -> None:
    """5/5 对联合 ≥ 独立 → 成立。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "成立"


def test_h3_4_5_weak() -> None:
    """4/5 对联合 ≥ 独立 → 方向成立。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "方向成立"


def test_h3_3_5_indeterminate() -> None:
    """3/5 → 不可判定。"""
    j = [20000.0, 8000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert evaluate_h3(j, i) == "不可判定"


def test_h3_2_5_negative() -> None:
    """≤ 2/5 → 不成立。"""
    j = [8000.0, 9000.0, 7000.0, 12000.0, 6000.0]
    i = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    assert evaluate_h3(j, i) == "不成立"


# ---------------------------------------------------------------- 路由推算
def test_winner_variant_full_joint() -> None:
    j = [22000.0, 25000.0, 18000.0, 30000.0, 27000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert winner_variant(j, i) == "joint"


def test_winner_variant_four_5_joint() -> None:
    """4/5 → joint（方向成立）。"""
    j = [20000.0, 22000.0, 18000.0, 25000.0, 9000.0]
    i = [12000.0, 13000.0, 11000.0, 14000.0, 12000.0]
    assert winner_variant(j, i) == "joint"


def test_winner_variant_three_5() -> None:
    """3/5 且独立中位数较高 → independent（[S3] 3/5 裁定的独立出口）。"""
    j = [14000.0, 11000.0, 19000.0, 5000.0, 6000.0]     # 3 胜（14000/11000/19000）；中位数 11000
    i = [13000.0, 10000.0, 18000.0, 20000.0, 19000.0]   # 中位数 18000
    assert winner_variant(j, i) == "independent"


def test_winner_variant_three_5_joint_median_higher() -> None:
    """`40` [S3] C4：3/5 且联合中位数较高 → joint（3/5 裁定的联合出口）。"""
    j = [20000.0, 20000.0, 20000.0, 5000.0, 6000.0]     # 3 胜；中位数 20000
    i = [13000.0, 10000.0, 18000.0, 20000.0, 19000.0]   # 中位数 18000
    assert winner_variant(j, i) == "joint"


@pytest.mark.parametrize("joint_wins", [0, 1, 2, 3, 4, 5])
def test_winner_variant_no_dangling_branch(joint_wins: int) -> None:
    """`40` [S3] C4：成对计数 0–5 全部落到合法出口，无悬空分支。

    3/5 档的期望值即"中位数较高者"：此处 3 胜位于高位，联合中位数 30000 > 独立中位数
    10000，故 3/5 落 joint——与 :func:`test_winner_variant_three_5` 的独立出口成对覆盖。
    """
    j = [30000.0 if k < joint_wins else 1000.0 for k in range(5)]
    i = [10000.0] * 5
    expected = {5: "joint", 4: "joint", 3: "joint",
                2: "independent", 1: "independent", 0: "independent"}[joint_wins]
    assert winner_variant(j, i) in ("joint", "independent")
    assert winner_variant(j, i) == expected


def test_winner_variant_two_5_independent() -> None:
    j = [8000.0, 9000.0, 7000.0, 12000.0, 6000.0]
    i = [20000.0, 22000.0, 18000.0, 25000.0, 21000.0]
    assert winner_variant(j, i) == "independent"


# ---------------------------------------------------------------- H3 配对口径
def test_pair_by_seed_pairs_identical_seed_values() -> None:
    """`40` [S3] 同种子成对 + `30` [S4] C2 同种子派生链：集合相同 → 按种子值字面配对。"""
    joint = {100: 1.0, 101: 2.0, 102: 3.0, 103: 4.0, 104: 5.0}
    indep = {104: 1.0, 100: 2.0, 102: 3.0, 101: 4.0, 103: 5.0}  # 插入序不影响配对
    assert pair_by_seed(joint, indep) == [
        (100, 100), (101, 101), (102, 102), (103, 103), (104, 104)
    ]


def test_pair_by_seed_rejects_mismatched_seed_sets() -> None:
    """种子集合不同（不相交或部分重叠）→ 报错，不退化升序槽位配对。"""
    with pytest.raises(ValueError, match="同种子配对不成立"):
        pair_by_seed({0: 1.0, 1: 2.0}, {100: 1.0, 101: 2.0})
    with pytest.raises(ValueError, match="同种子配对不成立"):
        pair_by_seed({1: 1.0, 2: 2.0}, {2: 1.0, 3: 2.0})  # 槽位配对会错配 (1,2)/(2,3)


# ---------------------------------------------------------------- 端到端
def test_evaluate_e1_gates_endtoend() -> None:
    """5 种子配对端到端：H1 + H3 + 路由。"""
    j = [25000.0, 22000.0, 19000.0, 24000.0, 26000.0]
    i = [11000.0, 12000.0, 10000.0, 13000.0, 11000.0]
    result = evaluate_e1_gates(j, i)
    assert result["H1"] == "成立"        # 5/5 > 1e4, 中位 24000 ≥ 2×1e4
    assert result["H3"] == "成立"        # 5/5 对联合 ≥ 独立
    assert result["winner_variant"] == "joint"


# --------------------------------------------- `40` [S1] C3：E1 公平性 config diff
E1_CONFIG_DIR = PROJECT_ROOT / "docs" / "specs" / "line2" / "e1_configs"

#: `40` [S1] C3 / `00` GC4：10 份 E1 config 必须逐字相同的键（数据集/步数/物理档/优化器）
E1_FAIRNESS_KEYS = (
    "data_dir", "n_low", "n_high", "max_steps", "eval_every", "patience",
    "batch", "lr", "w_psd", "device", "torch_threads",
)
#: `30` [S4] C2：变体间允许的差异只在网络拓扑（"差异仅在网络拓扑与损失罚项"）
E1_VARIANT_KEYS = ("base", "num_levels", "blocks", "ind_base", "ind_blocks")


def _load_e1_configs() -> dict[str, dict]:
    paths = sorted(E1_CONFIG_DIR.glob("e1_*.json"))
    if len(paths) != 10:
        raise AssertionError(f"{E1_CONFIG_DIR}：E1 config 应恰 10 份，实得 {len(paths)} 份")
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}


def test_e1_configs_differ_only_in_variant_and_seed() -> None:
    """`40` [S1] C3：10 份 E1 config 仅在 (variant, seed) 上不同（GC4 公平性自动断言）。"""
    configs = _load_e1_configs()
    by_variant: dict[str, list[dict]] = {}
    for cfg in configs.values():
        by_variant.setdefault(str(cfg["variant"]), []).append(cfg)
    assert set(by_variant) == {"joint", "independent"}
    for sub in by_variant.values():
        assert len(sub) == 5
        assert len({int(cfg["seed"]) for cfg in sub}) == 5
    # H3 是**同种子成对**比较（`40` [S3]「同种子成对比较（5 对）」+ `30` [S4] C2
    # 「相同种子派生链」）：两变体的种子集合必须逐一对应，否则配对被种子混淆。
    seeds_joint = sorted(int(c["seed"]) for c in by_variant["joint"])
    seeds_indep = sorted(int(c["seed"]) for c in by_variant["independent"])
    assert seeds_joint == seeds_indep, (
        f"H3 要求同种子成对：joint {seeds_joint} vs independent {seeds_indep}"
    )
    # 同种子的两份 config 必须落在同序号 run_id 上（s0..s4 ↔ 种子一一对应）
    for cfg in configs.values():
        suffix = Path(cfg["out_dir"]).name.rsplit("_s", 1)[-1]
        assert int(cfg["seed"]) == 100 + int(suffix), (
            f"{Path(cfg['out_dir']).name} 的种子 {cfg['seed']} 与序号 {suffix} 不对应"
        )
    for key in E1_FAIRNESS_KEYS:
        values = {json.dumps(cfg[key], sort_keys=True) for cfg in configs.values()}
        assert len(values) == 1, f"公平性键 {key} 在 10 份 config 间不一致：{values}"
    for cfg in configs.values():
        assert cfg["max_steps"] == 50000      # `30` [S3] 预注册步数预算（GC4）
        assert cfg["n_low"] == 10000          # `40` [S1] E1 深挖工况
        assert cfg["n_high"] == 200000.0      # `00` [S1] 目标端粒子数
    # 键集差异只允许落在变体拓扑键上（其余键的有无也必须一致）
    assert len({frozenset(cfg) - set(E1_VARIANT_KEYS) for cfg in configs.values()}) == 1
    # 拓扑键在变体内恒定：差异只跨变体、不跨种子
    for sub in by_variant.values():
        for key in E1_VARIANT_KEYS:
            assert len({json.dumps(cfg.get(key), sort_keys=True) for cfg in sub}) == 1, key
    # 每 run 独立目录（`40` [S1] C2）：out_dir 基名与 config 文件名一一对应
    assert {Path(cfg["out_dir"]).name for cfg in configs.values()} == set(configs)


# ------------------------------------- `40` [S1] C1：eval 档位按 run 自己的 n_low
def _write_fake_run(results: Path, run_id: str, n_low: int, *, with_meta: bool = True) -> Path:
    """假 run 目录：best_val.ckpt（+ run_meta.json）——eval 档位测试的最小输入。"""
    run_dir = results / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_val.ckpt").write_bytes(b"fake-ckpt")
    if with_meta:
        (run_dir / "run_meta.json").write_text(
            json.dumps({"n_low": n_low, "variant": "joint", "seed": 0}), encoding="utf-8"
        )
    return run_dir


def _eval_namespace(results: Path, data_dir: Path, n_low: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        results_dir=results, data_dir=data_dir, runs=None, split="test", station=0,
        n_low=n_low, subs="0-1", device="cpu",
    )


def test_cmd_eval_takes_n_low_from_each_run_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`40` [S1] E2：`--n-low` 缺省 = 各 run 自己的档位（不是统一 1e4）。

    E1（1e4）与 E2（3e3 / 1e5）共用这条 eval 路径，而数据集五个档位都在——"缺省 1e4"
    不会报错，只会把 E2 的 run 按训练期未见的档位评估并把结果标成该 run 的 N_eff。
    """
    results = tmp_path / "results"
    results.mkdir(parents=True)
    for run_id, n_low in (("e2_joint_s0_3e3", 3000), ("e2_joint_s0_1e5", 100000)):
        _write_fake_run(results, run_id, n_low)

    captured: list[list[str]] = []
    monkeypatch.setattr(evaluate, "main", lambda argv: captured.append(list(argv)) or 0)
    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data")) == 0

    by_ckpt = {
        argv[argv.index("--ckpt") + 1]: argv[argv.index("--n-low") + 1] for argv in captured
    }
    assert by_ckpt == {
        str(results / "e2_joint_s0_3e3" / "best_val.ckpt"): "3000",
        str(results / "e2_joint_s0_1e5" / "best_val.ckpt"): "100000",
    }
    # 站 0 写规范产物名：门禁与产出物按 test_eval*.json 的 glob 消费（`40` [S2]/[S4]）
    assert all(argv[argv.index("--out") + 1].endswith("test_eval.json") for argv in captured)


def test_cmd_eval_explicit_n_low_wins_and_missing_meta_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """显式 `--n-low` 覆盖各 run 档位；缺 run_meta.json 的 run 跳过而非套用别的档位。"""
    results = tmp_path / "results"
    results.mkdir(parents=True)
    _write_fake_run(results, "e1_joint_s0", 10000)
    _write_fake_run(results, "e1_indep_s0", 10000, with_meta=False)

    captured: list[list[str]] = []
    monkeypatch.setattr(evaluate, "main", lambda argv: captured.append(list(argv)) or 0)
    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data")) == 0
    assert [argv[argv.index("--ckpt") + 1] for argv in captured] == [
        str(results / "e1_joint_s0" / "best_val.ckpt")
    ]
    assert captured[0][captured[0].index("--n-low") + 1] == "10000"
    assert "缺 run_meta.json" in capsys.readouterr().out

    captured.clear()
    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data", n_low=50000)) == 0
    assert captured[0][captured[0].index("--n-low") + 1] == "50000"


# --------------------------------------------- `40` [S1] C5：尺子先行的顺序约束
def _write_run_meta(
    run_dir: Path, started_at: str, completed_at: str, seed: int = 0,
    variant: str = "joint",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps({
            "started_at": started_at, "completed_at": completed_at, "code_version": "a" * 40,
            "data_version": "line2_ds_v1", "spec_version": "line2-spec-v1.0",
            "variant": variant, "seed": seed, "n_low": 10000, "max_steps": 50000,
        }),
        encoding="utf-8",
    )


def _write_e0_meta(results_dir: Path, completed_at: str) -> None:
    (results_dir / "e0_curve.meta.json").write_text(
        json.dumps({"started_at": "2026-09-14T04:09:42+00:00", "completed_at": completed_at}),
        encoding="utf-8",
    )


def test_check_e0_precedes_runs_passes_and_lists_checked(tmp_path: Path) -> None:
    """`40` [S1] C5：E0 完成早于全部 run 启动 → 可验证、无违例。"""
    res = tmp_path / "results"
    res.mkdir(parents=True)
    _write_e0_meta(res, "2026-09-14T04:11:22+00:00")
    _write_run_meta(res / "e1_joint_s0", "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    _write_run_meta(res / "e1_joint_s1", "2026-09-14T05:00:01+00:00", "2026-09-14T06:00:00+00:00",
                    seed=1)
    out = check_e0_precedes_runs(res)
    assert out["verifiable"] is True
    assert out["violations"] == []
    assert set(out["checked"]) == {"e1_joint_s0", "e1_joint_s1"}


def test_check_e0_precedes_runs_raises_on_violation(tmp_path: Path) -> None:
    """`40` [S1] C5 反例：任一 run 启动不晚于 E0 完成即报错（含跨时区写法）。"""
    res = tmp_path / "results"
    res.mkdir(parents=True)
    _write_e0_meta(res, "2026-09-14T04:11:22+00:00")
    _write_run_meta(res / "e1_joint_s0", "2026-09-14T04:00:00+00:00", "2026-09-14T04:30:00+00:00")
    # 12:00+08:00 = 04:00 UTC，同样早于 E0 完成——时区归一后不得漏判
    _write_run_meta(res / "e1_joint_s1", "2026-09-14T12:00:00+08:00", "2026-09-14T13:00:00+08:00",
                    seed=1)
    _write_run_meta(res / "e1_joint_s2", "2026-09-14T09:00:00+00:00", "2026-09-14T10:00:00+00:00",
                    seed=2)
    with pytest.raises(ValueError, match="C5"):
        check_e0_precedes_runs(res)


def test_check_e0_precedes_runs_unverifiable_without_evidence(tmp_path: Path) -> None:
    """`40` [S1] C5：证据缺失判"不可验证"并列出缺失项，不静默判通过。"""
    res = tmp_path / "results"
    res.mkdir(parents=True)
    out = check_e0_precedes_runs(res)
    assert out["verifiable"] is False
    assert out["checked"] == {}
    assert len(out["missing"]) == 2          # E0 侧文件 + run_meta 两侧都缺
    _write_e0_meta(res, "2026-09-14T04:11:22+00:00")
    out = check_e0_precedes_runs(res)
    assert out["verifiable"] is False
    assert len(out["missing"]) == 1          # 仍缺 run 侧时间戳


# --------------------------------------------- `40` [S1] C2：run 目录契约与 GC7 三元组
def test_run_dir_triple_reads_triple_and_registers_presence(tmp_path: Path) -> None:
    """`40` [S1] C2：假 run 目录读全三元组；未落盘的契约文件显式登记。

    契约文件名取 spec 原文 ``config.yaml``（`40` [S1] C2）；该文件只登记存在性
    （YAML，无生产者，不做内容校验）。
    """
    run = tmp_path / "e1_joint_s0"
    _write_run_meta(run, "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    (run / "summary.json").write_text(
        json.dumps({"variant": "joint", "seed": 0, "best_val_n_eff": 15000.0}), encoding="utf-8")
    (run / "config.yaml").write_text("variant: joint\nseed: 0\n", encoding="utf-8")
    record = run_dir_triple(run)
    assert (record["data_version"], record["code_version"], record["spec_version"]) == (
        "line2_ds_v1", "a" * 40, "line2-spec-v1.0")
    assert record["run_id"] == "e1_joint_s0"
    assert record["best_val_n_eff"] == 15000.0
    assert "config.yaml" in record["present"]
    assert record["missing"] == ["train_log.jsonl", "best_val.ckpt", "last.ckpt"]


def test_run_dir_triple_registers_missing_config_yaml_by_spec_name(tmp_path: Path) -> None:
    """`40` [S1] C2：config 副本未落盘时按 spec 名 ``config.yaml`` 登记（实况缺口）。"""
    run = tmp_path / "e1_joint_s0"
    _write_run_meta(run, "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    (run / "summary.json").write_text(
        json.dumps({"variant": "joint", "seed": 0, "best_val_n_eff": 15000.0}), encoding="utf-8")
    record = run_dir_triple(run)
    assert record["missing"] == [
        "config.yaml", "train_log.jsonl", "best_val.ckpt", "last.ckpt"]
    assert "config.yaml" not in record["present"]


def test_run_dir_triple_raises_on_missing_field(tmp_path: Path) -> None:
    """`40` [S1] C2：run_meta 缺字段报错（缺哪个字段在消息里点名）。"""
    run = tmp_path / "e1_joint_s0"
    _write_run_meta(run, "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    payload = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    del payload["spec_version"]
    (run / "run_meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="spec_version"):
        run_dir_triple(run)


def test_run_dir_triple_raises_on_short_code_version(tmp_path: Path) -> None:
    """`00` GC7：code_version 非完整 40 位 git hash 报错。"""
    run = tmp_path / "e1_joint_s0"
    _write_run_meta(run, "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    payload = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    payload["code_version"] = "abc123"
    (run / "run_meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="40 位"):
        run_dir_triple(run)


def test_run_dir_triple_raises_on_conflicting_summary(tmp_path: Path) -> None:
    """`40` [S1] C2：summary.json 与 run_meta 的种子冲突报错（同一 run 两份溯源必须同一次训练）。"""
    run = tmp_path / "e1_joint_s0"
    _write_run_meta(run, "2026-09-14T05:00:00+00:00", "2026-09-14T06:00:00+00:00")
    (run / "summary.json").write_text(
        json.dumps({"variant": "joint", "seed": 7, "best_val_n_eff": 15000.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        run_dir_triple(run)


def test_run_dir_triple_missing_run_meta_is_file_not_found(tmp_path: Path) -> None:
    run = tmp_path / "e1_joint_s0"
    run.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        run_dir_triple(run)


# --------------------------------------------- `40` [S4]：三类产出物一键脚本
def _fake_curve() -> dict:
    return {
        "tiers": [1000, 3000, 10000, 30000, 100000], "k": 20, "beam_ids": [1, 2, 3],
        "points": [
            {"n_low": 1000, "median": 4.0, "iqr_lo": 3.9, "iqr_hi": 4.1, "n_samples": 60},
            {"n_low": 3000, "median": 2.3, "iqr_lo": 2.2, "iqr_hi": 2.4, "n_samples": 60},
            {"n_low": 10000, "median": 1.2, "iqr_lo": 1.1, "iqr_hi": 1.3, "n_samples": 60},
            {"n_low": 30000, "median": 0.68, "iqr_lo": 0.66, "iqr_hi": 0.70, "n_samples": 60},
            {"n_low": 100000, "median": 0.30, "iqr_lo": 0.28, "iqr_hi": 0.33, "n_samples": 60},
        ],
        "mutual_floor": {"1000": 5.0, "3000": 2.9, "10000": 1.5, "30000": 0.9, "100000": 0.4},
        "reference_noise_floor": 0.2856,
    }


def _fake_test_eval(run_id: str, n_low: int, n_eff: float, flag: str | None = None,
                    err: float = 0.5, station: int = 0) -> dict:
    return {
        "run_id": run_id, "variant": "joint", "seed": 0, "split": "test", "station": station,
        "n_low": n_low, "subs": [0, 1],
        "curve": {"path": "e0_curve.json", "beams_scope": "valtest",
                  "reference_noise_floor": 0.2856},
        "n_beams": 3, "n_groups": 6, "err_median": err, "n_eff": n_eff, "n_eff_flag": flag,
        "per_beam": [{"beam_id": b, "err": err, "n_eff": n_eff, "n_eff_flag": flag,
                      "obs": [0.4] * 13, "n_eff_median": n_eff} for b in (1, 2, 3)],
        "code_version": "a" * 40,
    }


def _fake_gates() -> dict:
    """门禁产物：四条门禁全部走"降级"分支（种子/容差/站 1 不足），快照键齐备。"""
    return {
        "results_dir": "fake", "runs": ["e1_joint_s0", "e2_joint_s0_1e5"],
        "code_version": "a" * 40,
        "gate_dims": {"split": "test", "station": 0, "n_low": 10000, "n_seeds_expected": 5},
        "curve": {"path": "e0_curve.json", "beams_scope": "valtest",
                  "reference_noise_floor": 0.2856,
                  "nodes": [1000, 3000, 10000, 30000, 100000]},
        "val_segment": {"source": "summary.json:best_val_n_eff",
                        "consumed_by": "variant_routing", "n_pairs": 1,
                        "per_seed_n_eff": {"joint": {"0": 15000.0}}},
        "test_segment": {"consumed_by": "H1/H3/H7/H8", "n_eval_files": 2, "gate_inputs": [],
                         "cross_station_inputs": [], "per_seed_n_eff": {}},
        "winner_variant": "joint", "routing_basis": "val_insufficient",
        "H1": {"verdict": "不可判定", "variant": "joint", "n_seeds": 1, "median_n_eff": 40000.0,
               "n_above_n_low": 1, "thresholds": {"n_low": 10000.0, "median_min": 20000.0},
               "inputs": {"0": 40000.0}, "reason_code": "winner_or_seeds_insufficient"},
        "H3": {"verdict": "不可判定", "n_pairs": 1, "joint_wins": 1,
               "pairing": {"basis": "seed_value", "pairs": [[100, 100]]},
               "inputs": {"joint": {"100": 40000.0}, "independent": {"100": 30000.0}},
               "reason_code": "pairs_insufficient"},
        "H7": {"verdict": "不可判定", "rho": None, "weak": False, "decay": None,
               "variant": "joint", "n_seeds": 0, "aggregation": "seed_median_then_ratio",
               "thresholds": {"strong": 1.0, "weak": 0.5}, "per_seed": {},
               "reason_code": "station1_missing"},
        "H8": {"seeds": [0], "n_seeds": 1, "tolerance": None, "verdict": "不可判定",
               "pass_rate": None, "per_observable": [], "n_beams": 0, "beam_ids": [],
               "tolerance_tier_per_beam": {}, "tolerance_tier_counts": {},
               "tolerance_tier_mode": None,
               "variant": "joint", "reason_code": "tolerance_missing"},
        "beam_n_eff_snapshot": {"joint": [{"beam_id": 1, "n_eff_median": 40000.0,
                                           "n_eff_per_seed": {"0": 40000.0},
                                           "flag_per_seed": {"0": None}}]},
    }


@pytest.fixture()
def fake_products(tmp_path: Path) -> Path:
    """`40` [S4] C1 的最小假产物：e0_curve.json + 两份 test_eval.json + e1_gates.json。"""
    res = tmp_path / "results"
    res.mkdir(parents=True)
    (res / "e0_curve.json").write_text(json.dumps(_fake_curve()), encoding="utf-8")
    for run_id, n_low, n_eff, flag, err in (
        ("e1_joint_s0", 10000, 40000.0, None, 0.55),
        ("e2_joint_s0_1e5", 100000, 100000.0, "overflow", 0.25),
    ):
        run_dir = res / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "test_eval.json").write_text(
            json.dumps(_fake_test_eval(run_id, n_low, n_eff, flag, err=err)), encoding="utf-8")
    (res / "e1_gates.json").write_text(json.dumps(_fake_gates()), encoding="utf-8")
    return res


def _digest(paths) -> dict[str, str]:
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def test_map_report_three_products_bitwise_reproducible(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S4] C1：同输入两次生成，CSV/报告/PNG 逐位一致；PNG 非空；图落在 reports 侧。"""
    out_dir, reports_dir = tmp_path / "out", tmp_path / "reports"
    first = generate(fake_products, out_dir=out_dir, reports_dir=reports_dir)
    digest1 = _digest((first["figure"], first["table"], first["report"]))
    second = generate(fake_products, out_dir=out_dir, reports_dir=reports_dir)
    digest2 = _digest((second["figure"], second["table"], second["report"]))
    assert digest1 == digest2
    assert first["figure"].name == "figure_avail_domain.png"
    assert first["table"].name == "n_eff_table.csv"
    assert first["report"].name == "phase1_report.md"
    # 地图与报告同目录（交付物随 reports/ 入库）；N_eff 表留在 results 侧的输出目录
    assert first["figure"] == reports_dir / "figure_avail_domain.png"
    assert first["report"] == reports_dir / "phase1_report.md"
    assert first["table"] == out_dir / "n_eff_table.csv"
    assert first["figure"].stat().st_size > 0
    assert first["n_rows"] == 2


def test_map_report_writes_only_its_three_outputs(fake_products: Path, tmp_path: Path) -> None:
    """`40` Global Constraints：产出物脚本只读——输入目录的每个文件内容不变、无新增。

    写入面恰为三个路径：reports 侧只落图与报告，输出目录只落 N_eff 表（不多建 assets/ 之类）。
    """
    before = {
        p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(fake_products.rglob("*")) if p.is_file()
    }
    out_dir, reports_dir = tmp_path / "out", tmp_path / "reports"
    generate(fake_products, out_dir=out_dir, reports_dir=reports_dir)
    after = {
        p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(fake_products.rglob("*")) if p.is_file()
    }
    assert before == after
    assert sorted(p.name for p in reports_dir.rglob("*")) == [
        "figure_avail_domain.png", "phase1_report.md"]
    assert sorted(p.name for p in out_dir.rglob("*")) == ["n_eff_table.csv"]


def test_map_report_figure_dir_overrides_reports_dir(fake_products: Path, tmp_path: Path) -> None:
    """`--figure-dir` 覆盖图件落盘目录；报告里的配套产出物路径随实际图路径走。"""
    reports_dir = tmp_path / "reports"
    out = generate(fake_products, out_dir=tmp_path / "out", reports_dir=reports_dir,
                   figure_dir=tmp_path / "figs")
    assert out["figure"] == tmp_path / "figs" / "figure_avail_domain.png"
    assert out["figure"].is_file()
    assert sorted(p.name for p in reports_dir.rglob("*")) == ["phase1_report.md"]
    assert str(out["figure"].resolve()) in out["report"].read_text(encoding="utf-8")


@pytest.mark.parametrize("victim", ["e0_curve.json", "e1_gates.json", "all_test_evals"])
def test_map_report_missing_inputs_raise(fake_products: Path, tmp_path: Path, victim: str) -> None:
    """`40` [S4]：缺必需产物明确报错，不静默跳过。"""
    if victim == "all_test_evals":
        for path in fake_products.glob(f"*/{EVAL_GLOB}"):
            path.unlink()
    else:
        (fake_products / victim).unlink()
    with pytest.raises(FileNotFoundError):
        generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")


def test_collect_n_eff_table_rejects_bad_rows(fake_products: Path) -> None:
    """`20` [S3] C2 / `00` GC3：缺字段、非 test 段、无标记越界、标记与端点不符均报错。"""
    path = fake_products / "e1_joint_s0" / "test_eval.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["n_eff"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="n_eff"):
        collect_n_eff_table(fake_products)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["n_eff"] = 40000.0
    payload["split"] = "val"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test 段"):
        collect_n_eff_table(fake_products)

    payload["split"] = "test"
    payload["n_eff"] = 500000.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="报告域"):
        collect_n_eff_table(fake_products)

    payload["n_eff_flag"] = "overflow"      # 标记在但取值不是域端点
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="端点"):
        collect_n_eff_table(fake_products)


def test_n_eff_table_columns_and_deterministic_order(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S4]：CSV 列序冻结；行序按 (站, N_low, 变体, 种子, run_id) 确定。"""
    rows = collect_n_eff_table(fake_products)
    assert all(list(row) == list(TABLE_COLUMNS) for row in rows)
    out = write_n_eff_table(rows, tmp_path / "n_eff_table.csv")
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert header == list(TABLE_COLUMNS)
    assert [row[0] for row in body] == ["e1_joint_s0", "e2_joint_s0_1e5"]
    assert body[0][7] == "" and body[1][7] == "overflow"


def test_report_carries_gate_snapshots_and_why_chapter(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S4] C2：每条门禁结论附判定表达式的输入数值快照；报告含第 7 章设计理由。"""
    out = generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")
    text = out["report"].read_text(encoding="utf-8")
    assert "## 5. 门禁结论" in text
    assert "## 7. 设计理由（Why）" in text
    for heading in ("### 5.1 H1（放大存在性）", "### 5.2 H3（联合优于独立）",
                    "### 5.3 H7（跨站泛化）", "### 5.4 H8（细结构可恢复性）"):
        assert heading in text
    for row in ("| median_n_eff | 40000 |", "| n_above_n_low | 1 |",
                "| joint_wins | 1 |", "| pairing.basis | seed_value |",
                "| rho | — |", "| aggregation | seed_median_then_ratio |",
                "| pass_rate | — |", "| tolerance | — |",
                # H8 的容差档映射：未映射时按同一行位置显式给出降级原因（不静默省略）
                "| 容差档映射 | —（未映射，原因 tolerance_missing） |"):
        assert row in text
    # 降级原因显式登记（第 4 章），不静默
    for code in ("winner_or_seeds_insufficient", "pairs_insufficient",
                 "station1_missing", "tolerance_missing"):
        assert code in text
    assert "无任一 `run_meta.json`" in text
    assert "触及报告域端点" in text
    # 报告头的"配套产出物"指向实际落盘路径（默认：图与报告同在 reports 目录）
    assert str(out["figure"].resolve()) in text
    assert str(out["table"].resolve()) in text
    # 报告头不把门禁产物的 code_version 冒充"本批实验的代码版本"：`00` GC7 要求的是每个 run
    # 的 manifest 三元组，run 级三元组在第 1.2 节
    assert "门禁产物 code_version" in text
    assert "`00` GC7 要求的每个 run 的三元组见 §1.2" in text


def test_report_discloses_h8_tolerance_tier_mapping(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S4] C2 + `20` [S4]：H8 快照披露容差档映射，容差线路径归一为仓库相对路径。

    容差档（N_eff → 最近曲线档）是 H8 判定表达式的输入，而 H8 的结论恰由该档决定；
    鉴别力：旧实现只列 `tolerance` 路径（且是本机绝对路径），读者必须回读 `gates.py` 才能
    知道用了哪一条容差线——本用例断言的"容差档映射"行在旧实现下不存在。
    """
    path = fake_products / "e1_gates.json"
    gates = json.loads(path.read_text(encoding="utf-8"))
    absolute = str(PROJECT_ROOT / "studies" / "line2_particle_sr" / "results" / "tolerance.json")
    gates["H8"].update({
        "tolerance": absolute, "verdict": "不成立", "pass_rate": 0.25,
        "n_beams": 3, "beam_ids": [1, 2, 3],
        "tolerance_tier_per_beam": {
            "1": {"n_eff_median": 100_000.0, "tier": 100_000},
            "2": {"n_eff_median": 100_000.0, "tier": 100_000},
            "3": {"n_eff_median": 3_000.0, "tier": 3_000},
        },
        "tolerance_tier_counts": {"3000": 1, "100000": 2},
        "tolerance_tier_mode": 100_000,
        "per_observable": [{"index": 0, "name": "sigma_xx_rel", "n_pass": 0, "n_beams": 3,
                            "pass_rate": 0.0}],
    })
    path.write_text(json.dumps(gates), encoding="utf-8")
    text = generate(fake_products, out_dir=tmp_path / "out",
                    reports_dir=tmp_path / "reports")["report"].read_text(encoding="utf-8")
    assert ("| 容差档映射 | 1e5（每束 N_eff 中位数最近档，`20` [S4]；档位分布 3e3×1, 1e5×2） |"
            in text)
    assert "| tolerance | studies/line2_particle_sr/results/tolerance.json |" in text
    assert absolute not in text          # 本机绝对路径不进报告
    # 逐束映射明细（每束一条）留在门禁产物里，不挤进"项/值"两列
    assert "tolerance_tier_per_beam" not in text


def test_report_title_and_anchors_follow_actual_products(fake_products: Path, tmp_path: Path
                                                         ) -> None:
    """§3.2 标题随实际档位自适应；§7 的锚点带 spec 前缀；Global Constraints 归属 `40`。

    鉴别力：旧实现的标题写死"主曲线三点"，而本批产物只有 1e4 / 1e5 两档；旧 §7.6 写裸锚
    "（D8）""（D4）"（四份 `.decisions.md` 都有 D4/D8，读者无法解析），旧 §7.1 把顶层
    `## Global Constraints` 记成 `[S4]` 的子节。
    """
    text = generate(fake_products, out_dir=tmp_path / "out",
                    reports_dir=tmp_path / "reports")["report"].read_text(encoding="utf-8")
    assert "主曲线三点" not in text
    assert "### 3.2 主曲线聚合（N_low = 1e4, 1e5；方法可用域地图面板 A 的数值）" in text
    assert "与 `40` D8" in text and "（`40` D4）" in text
    assert "[S4] Global Constraints" not in text
    assert "`40` Global Constraints" in text


def test_report_discloses_config_yaml_gap_by_spec_name(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S1] C2：run 目录缺 config 副本时按 spec 名 ``config.yaml`` 披露，不补写产物。

    缺口披露还必须写明配置的实际承载（冻结副本 + `run_meta.json`），否则读者会把
    "缺 config" 读成"该 run 无配置可核"。
    """
    for run_id in ("e1_joint_s0", "e2_joint_s0_1e5"):
        _write_run_meta(fake_products / run_id, "2026-09-14T05:00:00+00:00",
                        "2026-09-14T06:00:00+00:00")
    before = sorted(p.name for p in fake_products.rglob("*") if p.is_file())
    out = generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")
    text = out["report"].read_text(encoding="utf-8")
    assert "2/2 个 run 目录缺 `config.yaml`（`40` [S1] C2 契约文件）" in text
    assert "config.json" not in text            # 缺口名与 spec 一致
    assert "`run_meta.json` 的 `config`/`cmd` 字段承载" in text
    # 只披露、不补写：输入目录没有新增文件
    after = sorted(p.name for p in fake_products.rglob("*") if p.is_file())
    assert before == after and "config.yaml" not in after


def test_report_discloses_h7_overflow_indeterminacy(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S2]/[S4] C2：H7 因上/下溢判不可判定时，报告按 spec 原文披露该事实。

    报告的判定表达式必须写 [S2] 的条款（含"标记使 ρ 无法计算 → 不可判定"），而不是
    实现口径；聚合条目的降级原因与中位数比值（`rho_median`）都要进快照表，逐种子表要能
    看到标记来自哪一侧、ρ 是多少（截断值之比仅作披露）。
    """
    path = fake_products / "e1_gates.json"
    gates = json.loads(path.read_text(encoding="utf-8"))
    gates["H7"] = {
        "verdict": "不可判定", "rho": 1.0, "rho_median": 1.0, "weak": False, "decay": None,
        "variant": "joint", "n_seeds": 1, "aggregation": "seed_median_then_ratio",
        "thresholds": {"strong": 1.0, "weak": 0.5}, "reason_code": "n_eff_flag_present",
        "flagged_seeds": {"0": {"station0": "underflow", "station1": "underflow"}},
        "per_seed": {"0": {
            "verdict": "不可判定", "rho": 1.0, "weak": False, "decay": None,
            "reason_code": "n_eff_flag_present", "n_eff_station0": 1000.0,
            "n_eff_station1": 1000.0, "flag_station0": "underflow", "flag_station1": "underflow",
        }},
    }
    path.write_text(json.dumps(gates), encoding="utf-8")
    out = generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")
    text = out["report"].read_text(encoding="utf-8")
    assert "N_eff 触发上/下溢标记使 ρ 无法计算 ⟺ 不可判定" in text
    assert "| reason_code | n_eff_flag_present |" in text      # 聚合条目的降级原因
    assert "| rho_median | 1 |" in text                        # 聚合比值（同值只渲染这一行）
    assert "| rho | 1 |" not in text
    assert "| 0 | 1 | 不可判定 | n_eff_flag_present | underflow | underflow | 1000 | 1000 |" in text
    assert "1/1 个种子任一侧携带标记（0=station0=underflow, station1=underflow）" in text
    assert "聚合条目因此同样判不可判定" in text


def test_report_discloses_h3_domain_edge_ties(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S3]/[S4] C2：H3 的"成立"若来自域顶平局，报告必须写清它只是字面结论。

    平局计入胜是冻结判据（结论照旧输出"成立"），但两侧同触 1e5 的平局是被截断的饱和值、
    不具备区分力：报告要给出平局数/严格获胜数（及其证据 `n_eff_flags`），并明说"不表示联合
    严格更优"。无平局的档位不得出现该提示（防误报）。
    """
    plain = generate(fake_products, out_dir=tmp_path / "out",
                     reports_dir=tmp_path / "reports")["report"].read_text(encoding="utf-8")
    assert "域顶平局提示" not in plain          # 默认假产物：H3 无披露字段 → 无提示

    path = fake_products / "e1_gates.json"
    gates = json.loads(path.read_text(encoding="utf-8"))
    gates["H3"].update({
        "verdict": "成立", "n_pairs": 5, "joint_wins": 5,
        "n_ties": 5, "n_strict_joint_wins": 0, "tied_at_domain_edge": True,
        "n_eff_flags": {
            "joint": {str(100 + s): "overflow" for s in range(5)},
            "independent": {str(100 + s): "overflow" for s in range(5)},
        },
    })
    path.write_text(json.dumps(gates), encoding="utf-8")
    text = generate(fake_products, out_dir=tmp_path / "out",
                    reports_dir=tmp_path / "reports")["report"].read_text(encoding="utf-8")
    assert "| n_ties | 5 |" in text
    assert "| n_strict_joint_wins | 0 |" in text
    assert "| tied_at_domain_edge | 是 |" in text
    assert "| n_eff_flags.joint[100] | overflow |" in text
    assert "| n_eff_flags.independent[104] | overflow |" in text
    assert "**域顶平局提示**：5/5 对为域端点平局" in text
    assert "该档**不具备区分力**" in text
    assert "不表示联合严格更优（严格更优仅 0 对）" in text


def test_generate_raises_when_e0_order_violated(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S1] C5：一键路径同样强制尺子先行——违例时不产出报告。"""
    _write_e0_meta(fake_products, "2026-09-14T04:11:22+00:00")
    _write_run_meta(fake_products / "e1_joint_s0", "2026-09-14T04:00:00+00:00",
                    "2026-09-14T04:30:00+00:00")
    with pytest.raises(ValueError, match="C5"):
        generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")
