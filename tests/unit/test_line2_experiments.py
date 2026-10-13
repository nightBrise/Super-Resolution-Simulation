"""L4 E1 门禁评估与 [S4] 产出物脚本的测试（spec line2/40）。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

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
    # 死键：`train.build_model` 对 joint 只读 `base`/`blocks`、对 independent 只读
    # `ind_base`/`ind_blocks`。另一臂的拓扑键留在 config 里会让 diff 显示一个**不生效的**
    # 拓扑差异（读者会以为独立臂还有一层 base/blocks 结构）。
    dead_keys = {"joint": ("ind_base", "ind_blocks"), "independent": ("base", "blocks")}
    for variant, keys in dead_keys.items():
        for cfg in by_variant[variant]:
            present = [k for k in keys if k in cfg]
            assert not present, f"{variant} 臂 config 携带对该臂无效的拓扑键（死键）：{present}"
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


def _eval_namespace(results: Path, data_dir: Path, n_low: int | None = None,
                    force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        results_dir=results, data_dir=data_dir, runs=None, split="test", station=0,
        n_low=n_low, subs="0-1", device="cpu", force=force,
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


# ----------------------------------- `00` GC3：test 只评一次（派发端强制 + 留痕）
def _history(run_dir: Path) -> list[dict]:
    path = run_dir / run_e1.EVAL_HISTORY_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_cmd_eval_skips_existing_target_and_force_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """目标 `test_eval.json` 已存在 ⇒ 默认跳过（GC3 只评一次）；`--force` 才覆盖并留痕。

    鉴别力：旧实现无条件覆盖写——重复派发会静默用当次代码覆盖交付件，首次产物的版本
    不可查（审查 M2 的实证：同一 (run, 站, 档) 被派发 2–3 次）。
    """
    results = tmp_path / "results"
    results.mkdir(parents=True)
    run_dir = _write_fake_run(results, "e1_joint_s0", 10000)
    target = run_dir / "test_eval.json"
    target.write_text('{"stale": true}', encoding="utf-8")

    dispatched: list[list[str]] = []
    monkeypatch.setattr(evaluate, "main", lambda argv: dispatched.append(list(argv)) or 0)

    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data")) == 0
    assert dispatched == []                                   # 默认不派发评估
    assert target.read_text(encoding="utf-8") == '{"stale": true}'   # 既有产物原样保留
    out = capsys.readouterr().out
    assert "跳过 e1_joint_s0（test_eval.json）" in out and "test 只评一次" in out
    skipped = _history(run_dir)
    assert len(skipped) == 1 and skipped[0]["skipped"] is True
    assert skipped[0]["overwrote_existing"] is False and skipped[0]["n_low"] == 10000

    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data", force=True)) == 0
    assert len(dispatched) == 1 and dispatched[0][dispatched[0].index("--out") + 1] == str(target)
    forced = _history(run_dir)
    assert len(forced) == 2 and forced[-1]["skipped"] is False
    assert forced[-1]["overwrote_existing"] is True and forced[-1]["force"] is True


def test_cmd_eval_records_history_for_fresh_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """产物不存在 ⇒ 正常评估，历史记 `overwrote_existing=False`（无覆盖事实）。"""
    results = tmp_path / "results"
    results.mkdir(parents=True)
    run_dir = _write_fake_run(results, "e1_joint_s0", 10000)
    monkeypatch.setattr(evaluate, "main", lambda argv: 0)

    assert run_e1.cmd_eval(_eval_namespace(results, tmp_path / "data")) == 0
    entry = _history(run_dir)[0]
    assert entry["skipped"] is False and entry["overwrote_existing"] is False
    assert entry["out"] == "test_eval.json" and entry["subs"] == "0-1"


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
               "variant": "joint", "reason_code": "tolerance_missing",
               "dark_bucket_m2": {
                   "aggregation": "per_beam_seed_median_then_beam_median",
                   "n_runs": 1, "n_beams_total": 0, "runs_without_m2": ["e1_joint_s0"],
                   "B0": {"bare_median": 0.0, "model_median": 0.4,
                          "relative_improvement": None, "n_beams": 3},
                   "B1": {"bare_median": 4.6, "model_median": 2.3,
                          "relative_improvement": 0.5, "n_beams": 3},
                   "B0B1_combined": {"bare_median": 2.3, "model_median": 1.35,
                                     "relative_improvement": 0.41304347826086957,
                                     "n_beams": 3},
                   "note": "（假产物）暗桶口径说明。",
               }},
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


def test_report_discloses_h8_dark_bucket_m2(fake_products: Path, tmp_path: Path) -> None:
    """`40` [S3] H8：暗桶（B0/B1）M2 单独报告——与达标率**分开**成段渲染。

    鉴别力：冻结文本要求暗桶报告与达标率并列；旧实现完全没有这一段（报告里
    `暗桶`/`B0B1` 命中 0），读者无法判断 13 项的失败里有多少来自幻觉性暗桶。
    裸输入为 0 的桶（B0 的结构性质）渲染为"无定义"而不是伪造一个比值。
    """
    text = generate(fake_products, out_dir=tmp_path / "out",
                    reports_dir=tmp_path / "reports")["report"].read_text(encoding="utf-8")
    assert "**暗桶（B0/B1）M2 相对改进**（`40` [S3] H8 明文要求**单独报告、不并入上面的达标率**）" in text
    assert "| B0 | 0 | 0.4 | —（无定义：裸输入为 0） | 3 |" in text
    assert "| B1 | 4.6 | 2.3 | 0.5 | 3 |" in text
    assert "| B0+B1（等权） | 2.3 | 1.35 | 0.413043 | 3 |" in text
    assert "口径：（假产物）暗桶口径说明。" in text
    assert "覆盖：1 个 run × 0 束，聚合次序 per_beam_seed_median_then_beam_median" in text
    assert "**缺暗桶数据的 run**（1 个）：e1_joint_s0" in text
    # 达标率一节照旧独立渲染（两段并列，不互相吞并）
    assert "| pass_rate | — |" in text


def test_report_requires_dark_bucket_section(fake_products: Path, tmp_path: Path) -> None:
    """门禁产物缺 `dark_bucket_m2` ⇒ 报告生成报错，不静默省略（`40` [S3] H8 的交付面）。"""
    path = fake_products / "e1_gates.json"
    gates = json.loads(path.read_text(encoding="utf-8"))
    del gates["H8"]["dark_bucket_m2"]
    path.write_text(json.dumps(gates), encoding="utf-8")
    with pytest.raises(ValueError, match="dark_bucket_m2"):
        generate(fake_products, out_dir=tmp_path / "out", reports_dir=tmp_path / "reports")


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


# ------------------------------------------- `50` [S4] Phase 1.5：矩阵配置断言与选型准绳
_PHASE15_TIERS = (3_000, 10_000, 100_000)
_PHASE15_SEEDS = (100, 101, 102, 103, 104)
#: 逐站四轴半宽（m / rad）——站 1 更宽（真实 v2 边界同构：公共域半宽站 1 ≈ 2× 站 0）
_AXIS_HALF_WIDTHS = {
    0: (1.99e-3, 1.756e-4, 1.964e-3, 3.672e-4),
    1: (4.02e-3, 6.04e-4, 9.90e-4, 9.89e-4),
}


def _blob_group(n_particles: int, offset: int) -> np.ndarray:
    """(6,128,128) int32 计数图组：六投影各一个高斯斑，逐投影和 = n_particles。"""
    grid = np.mgrid[0:128, 0:128]
    out = np.zeros((6, 128, 128), dtype=np.int64)
    for gi in range(6):
        cx, cy = 64 + (gi - 2) + offset % 5, 64 - gi + offset % 7
        weight = np.exp(
            -((grid[0] - cy) ** 2 + (grid[1] - cx) ** 2) / (2 * 5.0**2)
        )
        weight /= weight.sum()
        counts = np.floor(weight * n_particles).astype(np.int64)
        counts.flat[int(np.argmax(weight.ravel()))] += n_particles - int(counts.sum())
        out[gi] = counts
    return out


def _station_edges_fixture(station: int) -> np.ndarray:
    return np.stack(
        [np.linspace(-half, half, 129) for half in _AXIS_HALF_WIDTHS[station]]
    ).astype(np.float64)


def _write_tiny_field_dataset(
    root: Path,
    tiers: tuple[int, ...] = _PHASE15_TIERS,
    k: int = 5,
    splits: dict[str, list[int]] | None = None,
) -> Path:
    """手写最小 v2 数据集（含站 0 与站 1），供批源/训练/矩阵断言用例使用。

    站 1 数据**真实存在**：站 1 粒子数为 0 的断言因此可证伪（若批源误配 stations，
    抽批会直接把它们捞出来）。
    """
    root.mkdir(parents=True, exist_ok=True)
    splits = splits if splits is not None else {"train": [0, 1, 2, 3], "val": [4], "test": []}
    for split, beams in splits.items():
        with h5py.File(root / f"{split}.h5", "w") as h5:
            for beam in beams:
                grp = h5.create_group(f"beam_{beam:04d}")
                for station in (0, 1):
                    stg = grp.create_group(f"station{station}")
                    edges = _station_edges_fixture(station)
                    stg.create_dataset(
                        "target_full", data=_blob_group(200_000, beam + station)
                    )
                    for tier in tiers:
                        for sub in range(k):
                            dset = stg.create_dataset(
                                f"tier_{tier:06d}_sub_{sub:02d}",
                                data=_blob_group(tier, beam * 7 + station * 3 + sub),
                            )
                            dset.attrs["bin_edges"] = edges
    manifest = {
        "data_version": "line2_ds_v2",
        "master_seed": 20260913,
        "n_bins": 128,
        "tiers": list(tiers),
        "subsamples_per_tier": k,
        "splits": {split: [int(b) for b in beams] for split, beams in splits.items()},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _phase15_config(
    data_dir: Path, variant: str, seed: int, out_dir: Path, **over
) -> dict:
    """Phase 1.5 run config（字段名 = `src.line2.training.config_check` 记录的契约）。"""
    cfg = {
        "phase": "1.5",
        "variant": variant,
        "seed": int(seed),
        "kappa": "v2",
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "tiers": list(_PHASE15_TIERS),
        "stations": [0],
        "k_train": 5,
        "max_steps": 50_000,
        "eval_every": 1_000,
        "patience": 10,
        "batch": 16,
        "lr": 3e-4,
        "num_levels": 4,
        "w_psd": 0.1,
        "n_high": 200_000.0,
        "device": "cpu",
        "torch_threads": 2,
    }
    if variant == "joint":
        cfg.update({"base": 48, "blocks": 2})
    else:
        cfg.update({"ind_base": 16, "ind_blocks": 3})
    cfg.update(over)
    return cfg


def _phase15_matrix(data_dir: Path, runs_dir: Path) -> list[dict]:
    """2 臂 × 5 种子 = 10 个 run（`50` [S5] 矩阵；两臂种子逐元素相同）。"""
    return [
        _phase15_config(
            data_dir, variant, seed, runs_dir / f"e15_{variant}_s{index}"
        )
        for variant in ("joint", "independent")
        for index, seed in enumerate(_PHASE15_SEEDS)
    ]


def test_phase15_matrix_config_assertions(tmp_path: Path) -> None:
    """`50` [S4] C3：矩阵配置自动校验（种子集合/参数量差/档集/站集/k_train/站 1 粒子数）。

    正例走真实 v2 口径的字段组合；负例逐项违约（含"两臂种子集合不同"与"参数量对齐
    失守"），且站 1 探针本身可证伪——把 stations 改成 [0,1] 后同一批源确实能捞出站 1
    粒子（说明"批次含站 1 粒子数 = 0"不是空断言）。
    """
    from src.line2.training import config_check
    from src.line2.training.config_check import validate_phase15_matrix

    data_dir = _write_tiny_field_dataset(tmp_path / "ds")
    configs = _phase15_matrix(data_dir, tmp_path / "runs")
    report = validate_phase15_matrix(configs, data_dir=data_dir)

    assert report["station_check"] == f"ok:{data_dir}"
    assert report["station_counts"] == {0: 4 * 16}  # batch_probe=4 批 × batch=16
    assert abs(report["param_rel_diff"]) <= 0.30
    assert report["params_total"]["joint"] > 0 and report["params_total"]["independent"] > 0
    assert [run["seed"] for run in report["runs"][:5]] == list(_PHASE15_SEEDS)
    assert all(run["tiers"] == list(_PHASE15_TIERS) for run in report["runs"])
    assert all(run["stations"] == [0] and run["k_train"] == 5 for run in report["runs"])

    # 站 1 探针可证伪：loader 配了站 1 就能在 fixture 里捞到站 1 粒子
    assert config_check._station_counts(dict(configs[0], stations=[0, 1]), data_dir, 4).get(1)

    def _mutate(index: int, **over) -> dict:
        mutated = dict(configs[index])
        mutated.update(over)
        return [*configs[:index], mutated, *configs[index + 1 :]]

    cases = (
        ("种子集合多一个", _mutate(9, seed=105), "种子集合"),
        ("种子集合缺一个", _mutate(9, seed=103), "种子集合"),
        ("档集缺 3×10³", _mutate(0, tiers=[10_000, 100_000]), "训练档集"),
        ("站集含站 1", _mutate(0, stations=[0, 1]), "训练站集"),
        ("k_train 非 5", _mutate(0, k_train=20), "k_train"),
        ("参数量差超 30%", _mutate(0, base=24, blocks=1), "参数量差"),
        ("预算未满", _mutate(0, max_steps=10_000), "max_steps"),
        ("κ 口径非 v2", _mutate(0, kappa="v1"), "kappa"),
    )
    for label, mutated, expected in cases:
        with pytest.raises(ValueError, match=expected):
            validate_phase15_matrix(mutated, data_dir=data_dir)


def _phase15_function():
    """`train.train_phase15` 的 AST 节点（静态断言的作用域：Phase 1.5 选型分支）。"""
    from src.line2.training import train as train_module

    tree = ast.parse(Path(train_module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_phase15"
    )


def test_selection_val_nll_only(tmp_path: Path) -> None:
    """`50` [S4] C4：训练结束的 summary 契约；选型分支不存在 N_eff 比较。

    静态：``train_phase15`` 作用域内无任何含 ``n_eff`` 的标识符、best 更新走
    ``should_update_best(..., minimize=True)``（严格 `<`）、summary 含四项 C4 字段且
    stop_reason 的取值集 = {val_nll_patience, budget_exhausted}。
    功能：小 run 两种停止路径各一次（lr=0 ⇒ 评估轨迹恒定，patience 必触发），
    best_step = val_nll argmin、各档分项落盘、best_val.ckpt 与之同步。
    """
    from src.line2.training.train import train

    fn = _phase15_function()
    identifiers = {
        node.id for node in ast.walk(fn) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)}
    assert not {name for name in identifiers if "n_eff" in name.lower()}, (
        f"选型分支出现 N_eff 标识符：{sorted(n for n in identifiers if 'n_eff' in n.lower())}"
    )
    best_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "should_update_best"
    ]
    assert best_calls, "train_phase15 未调用 should_update_best"
    for call in best_calls:
        assert getattr(call.args[0], "id", None) == "val_nll"
        assert any(
            kw.arg == "minimize" and isinstance(kw.value, ast.Constant) and kw.value.value
            for kw in call.keywords
        ), "should_update_best 未按最小化（严格 <）调用"
    summary_keys = {
        key.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "summary" for t in node.targets)
        and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant)
    }
    assert {"stop_reason", "best_step", "best_val_nll", "best_val_nll_per_tier"} <= summary_keys
    literals = {
        node.value for node in ast.walk(fn) if isinstance(node, ast.Constant)
    }
    assert {"val_nll_patience", "budget_exhausted"} <= literals

    data_dir = _write_tiny_field_dataset(tmp_path / "ds")
    small = {"max_steps": 4, "eval_every": 2, "patience": 10, "batch": 4,
             "base": 24, "num_levels": 3, "blocks": 1, "device": "cpu"}
    out_dir = tmp_path / "run_budget"
    summary = train(_phase15_config(data_dir, "joint", 100, out_dir, **small))

    assert summary["phase"] == "1.5" and summary["stop_reason"] == "budget_exhausted"
    assert summary["steps"] == 4 == summary["max_steps"]
    history = [entry["val_nll"] for entry in summary["val_nll_history"]]
    assert len(history) == 2 and summary["best_val_nll"] == min(history)
    assert summary["best_step"] == 2 * (int(np.argmin(history)) + 1)
    assert set(summary["best_val_nll_per_tier"]) == {str(tier) for tier in _PHASE15_TIERS}
    assert all(np.isfinite(list(summary["best_val_nll_per_tier"].values())))
    # 混合档留痕：逐档抽批数之和 = 步数（机制确实在跑，且档集与 config 一致）
    assert summary["tier_draws"] == summary["val_nll_history"][-1]["tier_draws"]
    assert sum(summary["tier_draws"].values()) == summary["steps"] == 4
    assert set(summary["tier_draws"]) == {str(tier) for tier in _PHASE15_TIERS}
    log = [
        json.loads(line)
        for line in (out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["step"] for entry in log] == [2, 4]
    assert {"step", "train_nll", "val_nll", "val_nll_per_tier", "lr", "grad_norm"} <= set(log[0])
    assert (out_dir / "best_val.ckpt").is_file() and (out_dir / "last.ckpt").is_file()
    best_payload = torch.load(out_dir / "best_val.ckpt", map_location="cpu", weights_only=True)
    assert best_payload["step"] == summary["best_step"]
    assert best_payload["best_val_nll"] == summary["best_val_nll"]
    assert best_payload["master_seed"] == 100
    on_disk = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk == summary

    # patience 路径：lr=0 ⇒ 权重不变 ⇒ 首次评估后不再严格下降，patience=2 在第 3 步触发
    patience_dir = tmp_path / "run_patience"
    stopped = train(
        _phase15_config(
            data_dir, "joint", 100, patience_dir,
            max_steps=10, eval_every=1, patience=2, batch=4, lr=0.0,
            base=24, num_levels=3, blocks=1, device="cpu",
        )
    )
    assert stopped["stop_reason"] == "val_nll_patience"
    assert stopped["steps"] == 3 and len(stopped["val_nll_history"]) == 3
    values = [entry["val_nll"] for entry in stopped["val_nll_history"]]
    assert values[1] == values[2] == stopped["best_val_nll"]  # 持平不更新（严格 <）
    assert stopped["best_step"] == 1

    # 独立臂走同一条 Phase 1.5 路径（κ v2 由 ckpt config 分派）
    indep_dir = tmp_path / "run_indep"
    indep = train(
        _phase15_config(
            data_dir, "independent", 101, indep_dir,
            max_steps=2, eval_every=2, patience=10, batch=4,
            num_levels=3, ind_base=16, ind_blocks=1, device="cpu",
        )
    )
    assert indep["variant"] == "independent" and indep["stop_reason"] == "budget_exhausted"
    assert indep["kappa"] == "v2" and indep["best_step"] == 2
    assert torch.load(indep_dir / "best_val.ckpt", map_location="cpu",
                      weights_only=True)["config"]["kappa"] == "v2"


def test_phase15_training_deterministic_same_seed(tmp_path: Path) -> None:
    """`50` Global Constraints（确定性）：同 config 同种子重训，val_nll 轨迹逐位一致。

    训练侧随机性只有两处——torch 初始化/优化器（``set_deterministic``）与混合档批序
    （``SeedSequence(20260913)`` 派生）；两者都确定，故两次 run 的逐评估周期读数必须
    逐位相同（v1 侧同款断言见 ``test_train_smoke_and_determinism``）。
    """
    from src.line2.training.train import train

    data_dir = _write_tiny_field_dataset(tmp_path / "ds")
    over = {"max_steps": 3, "eval_every": 1, "patience": 10, "batch": 4,
            "base": 24, "num_levels": 3, "blocks": 1, "device": "cpu", "val_beams": 1}
    first = train(_phase15_config(data_dir, "joint", 103, tmp_path / "run_a", **over))
    second = train(_phase15_config(data_dir, "joint", 103, tmp_path / "run_b", **over))

    def trajectory(summary: dict) -> list[dict]:
        return [
            {key: value for key, value in entry.items() if key != "elapsed_s"}
            for entry in summary["val_nll_history"]
        ]

    assert trajectory(first) == trajectory(second)
    assert first["best_step"] == second["best_step"] == 3
    assert first["best_val_nll"] == second["best_val_nll"]
    third = train(_phase15_config(data_dir, "joint", 104, tmp_path / "run_c", **over))
    assert trajectory(third) != trajectory(first)


def test_selection_reading_delegates_to_s3_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`50` [S3] C1/C2 单点化：训练侧把 (束, 子样本) NLL 矩阵转发给 `metrics.decisions`。

    探针替换 [S3] 的两个聚合函数并记录其输入，跑 1 次评估的小 run：每个档位收到的
    矩阵 SHALL 为 (val 束 × k_train) 二维矩阵；summary 的 best_val_nll 与各档分项
    SHALL 等于按 [S3] 口径对该矩阵的重算值——训练侧没有自己的池化读法。
    """
    from src.line2.metrics import decisions
    from src.line2.training.train import train

    data_dir = _write_tiny_field_dataset(
        tmp_path / "ds", splits={"train": [0, 1, 2], "val": [3, 4], "test": []}
    )
    seen: dict[int, list[np.ndarray]] = {}
    real_tier, real_selection = decisions.tier_reading, decisions.selection_reading

    def record(per_tier) -> None:
        for tier, matrix in per_tier.items():
            seen.setdefault(int(tier), []).append(np.asarray(matrix, dtype=np.float64).copy())

    def tier_spy(per_tier, tier: int) -> float:
        record(per_tier)
        return real_tier(per_tier, tier)

    def selection_spy(per_tier) -> float:
        record(per_tier)
        return real_selection(per_tier)

    monkeypatch.setattr(decisions, "tier_reading", tier_spy)
    monkeypatch.setattr(decisions, "selection_reading", selection_spy)
    summary = train(
        _phase15_config(
            data_dir, "joint", 100, tmp_path / "run",
            max_steps=2, eval_every=2, patience=10, batch=3,
            base=24, num_levels=3, blocks=1, device="cpu",
        )
    )

    assert set(seen) == set(_PHASE15_TIERS)
    for tier, matrices in seen.items():
        for matrix in matrices:
            assert matrix.shape == (2, 5)  # 2 val 束 × k_train 5（未池化的逐 (束, 子样本) 读数）
        assert summary["best_val_nll_per_tier"][str(tier)] == pytest.approx(
            real_tier({tier: matrices[-1]}, tier)
        )
    assert summary["best_val_nll"] == pytest.approx(
        float(np.mean([real_tier({t: m[-1]}, t) for t, m in seen.items()]))
    )


# ============================ `50` [S5] Phase 1.5：E0' 门禁 / G1–G5 / 矩阵与编排 =====
from src.line2.experiments import (  # noqa: E402
    phase15_configs,
    phase15_gates,
    phase15_orchestrator,
)

#: 生产口径的 test 束集（`50` [S1]：150 束）；门禁判定域由数据契约冻结
_E15_BEAMS = list(range(150))
#: E0' 三件产物的时间戳（构造用；全部早于训练启动）
_E0_DONE = "2026-10-13T10:00:00+00:00"
_TRAIN_START = "2026-10-13T11:00:00+00:00"


def _e15_eval(
    run_id: str,
    *,
    variant: str,
    seed: int,
    tier: int,
    station: int,
    err_by_beam: dict[int, float],
    obs_by_beam: dict[int, list[float]] | None = None,
    split: str = "test",
) -> dict:
    """Phase 1.5 评估产物（字段 = `evaluate.evaluate_run` 的 v1 schema，`50` [S5] C8）。"""
    per_beam = []
    for beam in sorted(err_by_beam):
        obs = (obs_by_beam or {}).get(beam, [0.0] * 13)
        per_beam.append(
            {
                "beam_id": int(beam),
                "err": float(err_by_beam[beam]),
                "n_eff": 50_000.0,          # 披露字段：判定不得消费（`50` [S3] C1）
                "n_eff_flag": None,
                "obs": [float(v) for v in obs],
                "n_eff_median": 50_000.0,
                "m2_buckets": {
                    name: {
                        "n_cells": 400, "insufficient": False, "model_median": 0.5,
                        "bare_median": 1.0, "relative_improvement": 0.5,
                    }
                    for name in ("B0", "B1", "B2", "B3")
                }
                | {"improvement": {"B0": 0.5, "B1": 0.5}},
            }
        )
    errs = np.asarray([entry["err"] for entry in per_beam], dtype=np.float64)
    return {
        "run_id": run_id, "variant": variant, "seed": int(seed), "split": split,
        "station": int(station), "n_low": int(tier), "subs": list(range(20)),
        "curve": {"path": "e0_curve_v2.json", "beams_scope": "valtest",
                  "reference_noise_floor": 0.5},
        "n_beams": len(per_beam), "n_groups": len(per_beam) * 20,
        "err_median": float(np.median(errs)), "n_eff": 50_000.0, "n_eff_flag": None,
        "per_beam": per_beam, "code_version": "a" * 40,
    }


def _e15_bconst_station(err_by_beam: dict[int, float]) -> dict:
    values = np.asarray(list(err_by_beam.values()), dtype=np.float64)
    return {
        "group_projection_sums": [1.0] * 6,
        "group_sha256": "b" * 64,
        "n_eval_beams": len(err_by_beam),
        "eval_split": "test",
        "per_beam_m1": {str(b): float(v) for b, v in sorted(err_by_beam.items())},
        "err_median": float(np.median(values)),
        "err_min": float(values.min()),
        "err_max": float(values.max()),
    }


def _e15_alignment(
    run_id: str, seed: int, *, cos_ci: tuple[float, float] = (0.5, 0.7),
    drop_frac: float = 0.90, tier: int = 10_000, station: int = 0,
) -> dict:
    return {
        "run_id": run_id, "variant": "joint", "seed": int(seed), "split": "test",
        "station": int(station), "n_low": int(tier), "subs": list(range(5)),
        "n_beams": 150, "n_subs": 5, "cos_true": 0.6, "cos_scr": 0.05,
        "drop_frac": float(drop_frac), "disp_ratio": 1.1,
        "cos_true_ci95": [float(cos_ci[0]), float(cos_ci[1])],
        "cos_scr_ci95": [0.0, 0.1],
        "bootstrap": {"n_boot": 10_000, "root_seed": 20_260_825, "unit": "beam"},
    }


def _write_e0_prime(
    results: Path,
    *,
    data_dir: str,
    completed_at: str = _E0_DONE,
    tolerance_completed_at: str | None = None,
    drop_curve_meta: bool = False,
    tolerance_tiers: tuple[str, ...] = ("3000", "10000", "100000"),
    bconst_stations: tuple[str, ...] = ("0", "1"),
    with_groups: bool = True,
    data_dir_tag: str | None = None,
) -> None:
    """v2 E0' 三件产物（可注入违例：缺 meta、时间戳偏移、缺档、缺站）。"""
    results.mkdir(parents=True, exist_ok=True)
    tag = data_dir if data_dir_tag is None else data_dir_tag
    (results / "e0_curve_v2.json").write_text(
        json.dumps({"points": [{"n_low": 10_000, "median": 1.2}], "beam_ids": [0, 1]}),
        encoding="utf-8",
    )
    if not drop_curve_meta:
        (results / "e0_curve_v2.meta.json").write_text(
            json.dumps({"completed_at": completed_at, "data_dir": tag,
                        "data_version": "line2_ds_v2", "beams_scope": "valtest",
                        "stations": [0, 1], "tiers": [1000, 3000, 10000, 30000, 100000]}),
            encoding="utf-8",
        )
    (results / "tolerance_v2.json").write_text(
        json.dumps({
            "completed_at": tolerance_completed_at or completed_at,
            "data_dir": tag, "data_version": "line2_ds_v2",
            "dev_table": {tier: [0.1] * 13 for tier in tolerance_tiers},
        }),
        encoding="utf-8",
    )
    (results / "bconst_v2.json").write_text(
        json.dumps({
            "completed_at": completed_at, "data_dir": tag, "data_version": "line2_ds_v2",
            "station": {s: _e15_bconst_station({b: 0.5 for b in _E15_BEAMS})
                        for s in bconst_stations},
        }),
        encoding="utf-8",
    )
    if with_groups:
        np.savez_compressed(
            results / "bconst_v2_groups.npz",
            **{f"station{s}": np.zeros((6, 4, 4), dtype=np.float32) for s in bconst_stations},
        )


def _write_e15_summary(
    run_dir: Path, *, variant: str, seed: int, val_nll: float, tier_1e4: float | None = None
) -> None:
    """run 的 val 段读数（`50` [S3] C2：三档等权 val_nll + 各档分项）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({
            "phase": "1.5", "variant": variant, "seed": int(seed),
            "best_val_nll": float(val_nll),
            "best_val_nll_per_tier": {
                "3000": float(val_nll), "10000": float(tier_1e4 if tier_1e4 is not None else val_nll),
                "100000": float(val_nll),
            },
            "tiers": [3000, 10000, 100000], "stations": [0], "k_train": 5,
        }),
        encoding="utf-8",
    )


def _write_phase15_results(
    tmp_root: Path, *, beams: int = 12, model_err: float = 0.2, ref_err: float = 0.5
) -> Path:
    """C7/C9 的最小 Phase 1.5 results 目录：E0' 三件 + 10 run 的 summary + 60 评估 + 5 对齐。

    构造使五条门禁全部落"成立"（模型优于 B-const、观测量达标、站 1 退化比 < 1、对齐落差
    大）；改一个变量即可验证单条门禁（正/反例在各自的 gate 测试里另作构造）。
    """
    results = tmp_root / "results"
    data_dir = tmp_root / "line2_ds_v2"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps({"data_version": "line2_ds_v2", "splits": {},
                    "spec_version": "line2-spec-v1.1"}),
        encoding="utf-8",
    )
    _write_e0_prime(results, data_dir=str(data_dir))
    beam_ids = list(range(beams))
    for tag, variant, val_nll in (("joint", "joint", 1.0), ("indep", "independent", 2.0)):
        for index, seed in enumerate(_PHASE15_SEEDS):
            run_id = f"e15_{tag}_s{index}"
            _write_e15_summary(results / run_id, variant=variant, seed=seed, val_nll=val_nll)
            for tier in (3_000, 10_000, 100_000):
                for station in (0, 1):
                    err = model_err if tag == "joint" else model_err + 0.1
                    err += 0.05 * (station == 1)
                    errs = {b: err for b in beam_ids}
                    payload = _e15_eval(
                        run_id, variant=variant, seed=seed, tier=tier, station=station,
                        err_by_beam=errs,
                    )
                    phase15_gates.eval_path(results, run_id, tier, station).write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
    bconst = {
        "data_version": "line2_ds_v2", "data_dir": str(data_dir),
        "completed_at": _E0_DONE,
        "station": {
            "0": _e15_bconst_station({b: ref_err for b in beam_ids}),
            "1": _e15_bconst_station({b: ref_err - 0.1 for b in beam_ids}),
        },
    }
    (results / "bconst_v2.json").write_text(json.dumps(bconst), encoding="utf-8")
    for index, seed in enumerate(_PHASE15_SEEDS):
        run_id = f"e15_joint_s{index}"
        (results / phase15_gates.align_filename(run_id)).write_text(
            json.dumps(_e15_alignment(run_id, seed)), encoding="utf-8"
        )
    # 路由先行（链序：route → align → gates）；获胜臂由 val 段 val_nll 决定
    phase15_gates.write_routing(
        results,
        {"joint": [f"e15_joint_s{i}" for i in range(5)],
         "independent": [f"e15_indep_s{i}" for i in range(5)]},
    )
    return results


# --------------------------------------------------------------- C1：E0' 时间戳门禁
def test_e0_prime_before_training(tmp_path: Path) -> None:
    """`50` [S5] C1：E0' 三件落盘时间戳须早于 run 启动，违例拒绝启动（含可控 clock）。"""
    results = tmp_path / "results"
    data_dir = tmp_path / "line2_ds_v2"
    data_dir.mkdir(parents=True)
    _write_e0_prime(results, data_dir=str(data_dir))

    out = phase15_orchestrator.check_e0_prime_precedes(
        results, started_at=_TRAIN_START, data_dir=data_dir
    )
    assert set(out["products"]) == {"curve", "tolerance", "bconst"}
    assert out["products"]["bconst"]["stations"] == [0, 1]
    assert out["started_at"] == _TRAIN_START

    # 可控 clock：未显式给 started_at 时取 clock()（编排器每个 run 启动前调用）
    clocked = phase15_orchestrator.check_e0_prime_precedes(
        results, clock=lambda: _TRAIN_START, data_dir=data_dir
    )
    assert clocked["started_at"] == _TRAIN_START

    # 反例 1：容差线完成晚于 run 启动
    late = tmp_path / "late"
    _write_e0_prime(late, data_dir=str(data_dir),
                    tolerance_completed_at="2026-10-13T12:00:00+00:00")
    with pytest.raises(ValueError, match="C1"):
        phase15_orchestrator.check_e0_prime_precedes(
            late, started_at=_TRAIN_START, data_dir=data_dir
        )
    # 反例 2：曲线 meta 缺失（时间戳无来源）
    no_meta = tmp_path / "no_meta"
    _write_e0_prime(no_meta, data_dir=str(data_dir), drop_curve_meta=True)
    with pytest.raises(ValueError, match="completed_at 无来源"):
        phase15_orchestrator.check_e0_prime_precedes(
            no_meta, started_at=_TRAIN_START, data_dir=data_dir
        )
    # 反例 3：容差线缺档 / B-const 缺站 / 尺子量的是别的数据集
    for label, kwargs, pattern in (
        ("缺档", {"tolerance_tiers": ("10000",)}, "容差线缺档"),
        ("缺站", {"bconst_stations": ("0",)}, "两站口径"),
        ("异源", {"data_dir_tag": "somewhere/else"}, "不同源"),
    ):
        broken = tmp_path / f"broken_{label}"
        _write_e0_prime(broken, data_dir=str(data_dir), **kwargs)
        with pytest.raises(ValueError, match=pattern):
            phase15_orchestrator.check_e0_prime_precedes(
                broken, started_at=_TRAIN_START, data_dir=data_dir
            )

    # 编排器入口同款：违例时连 run_meta.json 都不落、不触 GPU
    configs = tmp_path / "configs"
    configs.mkdir()
    out_dir = tmp_path / "run_out"
    (configs / "e15_joint_s0.json").write_text(
        json.dumps(phase15_configs.build_config(
            "joint", 100, data_dir=data_dir, out_dir=out_dir, code_version="a" * 40
        )),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="C1"):
        phase15_orchestrator.run_one_train(
            configs, late, data_dir, "e15_joint_s0", gpu=0,
            clock=lambda: _TRAIN_START, free_mib={0: 24_000},
        )
    assert not (late / "e15_joint_s0" / "run_meta.json").exists()


# ------------------------------------------------------------------- C2：G1 判定
def test_gate_g1() -> None:
    """`50` [S5] C2：G1 四值映射（含正/反例构造）与判定输入快照。"""
    # 纯映射（CI 由构造给定，逐分支覆盖四类）
    assert phase15_gates.g1_verdict(1.00, [0.1, 0.4]) == "成立"
    assert phase15_gates.g1_verdict(0.93, [0.1, 0.4]) == "成立"
    assert phase15_gates.g1_verdict(0.89, [0.1, 0.4]) == "方向成立"
    assert phase15_gates.g1_verdict(0.10, [-0.4, -0.1]) == "不成立"
    assert phase15_gates.g1_verdict(0.99, [-0.1, 0.4]) == "不可判定"
    assert phase15_gates.g1_verdict(1.00, [float("nan"), float("nan")]) == "不可判定"

    reference = _e15_bconst_station({b: 0.5 for b in _E15_BEAMS})

    def _adv(deltas: dict[int, float]) -> list[dict]:
        return [
            _e15_eval(f"e15_joint_s{i}", variant="joint", seed=100 + i, tier=10_000, station=0,
                      err_by_beam={b: 0.5 - d for b, d in deltas.items()})
            for i in range(5)
        ]

    strong = phase15_gates.gate_g1(_adv({b: 0.4 for b in _E15_BEAMS}), reference)
    assert strong["verdict"] == "成立"
    assert strong["win_rate"] == 1.0 and strong["ci95"][0] > 0
    assert strong["n_beams"] == 150 and len(strong["inputs"]["delta"]) == 150
    assert strong["thresholds"]["win_rate_min"] == 0.90

    # 方向成立：赢率 0.8933（=134/150 < 0.90）但优势强正 ⇒ CI 下界 > 0
    mixed = {b: 0.2 for b in _E15_BEAMS[:134]} | {b: -1e-9 for b in _E15_BEAMS[134:]}
    directional = phase15_gates.gate_g1(_adv(mixed), reference)
    assert directional["win_rate"] < 0.90 and directional["ci95"][0] > 0
    assert directional["verdict"] == "方向成立"

    negative = phase15_gates.gate_g1(_adv({b: -0.3 for b in _E15_BEAMS}), reference)
    assert negative["verdict"] == "不成立"
    assert negative["ci95"][1] <= 0

    # 反例：模型几何平均与参照无差别（75 正 75 负）⇒ CI 跨 0 ⇒ 不可判定
    zero = {b: 0.1 for b in _E15_BEAMS[:75]} | {b: -0.1 for b in _E15_BEAMS[75:]}
    indeterminate = phase15_gates.gate_g1(_adv(zero), reference)
    assert indeterminate["verdict"] == "不可判定"
    assert indeterminate["ci95"][0] <= 0 < indeterminate["ci95"][1]

    missing_eval = phase15_gates.gate_g1([], reference)
    assert (missing_eval["verdict"], missing_eval["reason_code"]) == ("不可判定", "eval_missing")
    missing_ref = phase15_gates.gate_g1(_adv({b: 0.4 for b in _E15_BEAMS}), None)
    assert (missing_ref["verdict"], missing_ref["reason_code"]) == ("不可判定", "bconst_missing")
    # 束集不一致：不静默收窄统计域
    with pytest.raises(ValueError, match="束集不一致"):
        phase15_gates.gate_g1(_adv({b: 0.4 for b in _E15_BEAMS[:100]}), reference)


# ------------------------------------------------------------------- C3：G2 判定
def test_gate_g2() -> None:
    """`50` [S5] C3：G2 的 k_a/k_and 四值映射（正/反例）与缺读数降级。"""
    def _seeds(ci_lo: list[float], drop: list[float]) -> dict[int, dict]:
        return {
            seed: _e15_alignment(f"e15_joint_s{i}", seed, cos_ci=(lo, 0.9), drop_frac=d)
            for i, (seed, lo, d) in enumerate(zip(_PHASE15_SEEDS, ci_lo, drop))
        }

    five = [0.5] * 5
    high = [0.9] * 5
    assert phase15_gates.gate_g2(_seeds(five, high))["verdict"] == "成立"
    assert phase15_gates.gate_g2(_seeds(five, [0.9, 0.9, 0.9, 0.2, 0.2]))["verdict"] == "方向成立"
    # k_a = 3 / k_a ≥ 4 且 k_and ≤ 2 → 不可判定（两个不同入口）
    three = phase15_gates.gate_g2(_seeds([0.5, 0.5, 0.5, -0.1, -0.1], high))
    assert three["k_a"] == 3 and three["verdict"] == "不可判定"
    and_low = phase15_gates.gate_g2(_seeds(five, [0.9, 0.9, 0.1, 0.1, 0.1]))
    assert (and_low["k_a"], and_low["k_and"]) == (5, 2) and and_low["verdict"] == "不可判定"
    negative = phase15_gates.gate_g2(_seeds([0.5, 0.5, -0.1, -0.1, -0.1], high))
    assert negative["k_a"] == 2 and negative["verdict"] == "不成立"
    # 四值映射的合法枚举与计数快照
    for payload in (three, and_low, negative):
        assert payload["verdict"] in ("成立", "方向成立", "不成立", "不可判定")
        assert set(payload["per_seed"]) == {str(s) for s in _PHASE15_SEEDS}
    missing = phase15_gates.gate_g2({100: _e15_alignment("e15_joint_s0", 100)})
    assert (missing["verdict"], missing["reason_code"]) == ("不可判定", "alignment_missing")
    with pytest.raises(ValueError, match="档位/站不符"):
        phase15_gates.gate_g2(_seeds(five, high) | {}, tier=3_000)


# ------------------------------------------------------------------- C4：G3 判定
def test_gate_g3() -> None:
    """`50` [S5] C4：G3 按评估档直接索引容差线 + 达标率四值映射（正/反例）。"""
    tolerance = [0.1] * 13

    def _evals(passing_pairs: int) -> list[dict]:
        total = 150 * 13
        obs = {}
        for beam in _E15_BEAMS:
            row = []
            for j in range(13):
                index = beam * 13 + j
                row.append(0.0 if index < passing_pairs else 0.2)
            obs[beam] = row
        assert total == 1950
        return [
            _e15_eval(f"e15_joint_s{i}", variant="joint", seed=100 + i, tier=10_000, station=0,
                      err_by_beam={b: 0.2 for b in _E15_BEAMS}, obs_by_beam=obs)
            for i in range(5)
        ]

    perfect = phase15_gates.gate_g3(_evals(1950), tolerance)
    assert perfect["verdict"] == "成立" and perfect["pass_rate"] == 1.0
    directional = phase15_gates.gate_g3(_evals(1300), tolerance)
    assert 0.50 < directional["pass_rate"] < 0.80 and directional["verdict"] == "方向成立"
    negative = phase15_gates.gate_g3(_evals(0), tolerance)
    assert negative["verdict"] == "不成立" and negative["pass_rate"] == 0.0
    assert len(negative["per_observable"]) == 13
    # 容差线/评估缺失 → 不可判定（不现场猜容差）
    assert phase15_gates.gate_g3(_evals(1950), None)["reason_code"] == "tolerance_missing"
    assert phase15_gates.gate_g3([], tolerance)["reason_code"] == "eval_missing"
    with pytest.raises(ValueError, match="容差线形状"):
        phase15_gates.gate_g3(_evals(1950), [0.1] * 12)


# ------------------------------------------------------------------- C5：G4 判定
def test_gate_g4() -> None:
    """`50` [S5] C5：G4 的 R_diff 三分类 + 原始比并列披露（不进判定）。"""
    reference_s0 = _e15_bconst_station({b: 0.5 for b in _E15_BEAMS})
    reference_s1 = _e15_bconst_station({b: 0.4 for b in _E15_BEAMS})

    def _evals(err: float, station: int) -> list[dict]:
        return [
            _e15_eval(f"e15_joint_s{i}", variant="joint", seed=100 + i, tier=10_000, station=station,
                      err_by_beam={b: err for b in _E15_BEAMS})
            for i in range(5)
        ]

    pass_case = phase15_gates.gate_g4(
        _evals(0.2, 0), _evals(0.25, 1), reference_s0, reference_s1
    )
    assert pass_case["verdict"] == "成立" and pass_case["r_diff"] == pytest.approx(0.5)
    assert pass_case["raw_err_ratio"] == pytest.approx(1.25)

    # 方向成立：R_diff = 1.2（原始比 < 1——难度归一后才看得见的退化）
    directional = phase15_gates.gate_g4(
        _evals(0.5, 0), _evals(0.46, 1),
        _e15_bconst_station({b: 0.2 for b in _E15_BEAMS}),
        _e15_bconst_station({b: 0.1 for b in _E15_BEAMS}),
    )
    assert directional["r_diff"] == pytest.approx(1.2)
    assert directional["verdict"] == "方向成立"

    # 反例：原始比 0.7（"站 1 更优"）但难度归一后 R_diff = 1.5 ⇒ 不成立
    negative = phase15_gates.gate_g4(
        _evals(0.5, 0), _evals(0.35, 1),
        _e15_bconst_station({b: 0.3 for b in _E15_BEAMS}),
        _e15_bconst_station({b: 0.05 for b in _E15_BEAMS}),
    )
    assert negative["raw_err_ratio"] == pytest.approx(0.7)
    assert negative["r_diff"] == pytest.approx(1.5) and negative["verdict"] == "不成立"

    missing = phase15_gates.gate_g4(_evals(0.2, 0), [], reference_s0, reference_s1)
    assert (missing["verdict"], missing["reason_code"]) == ("不可判定", "input_missing")
    assert phase15_gates.gate_g4(
        _evals(0.2, 0), _evals(0.25, 1), reference_s0, None
    )["verdict"] == "不可判定"


# ------------------------------------------------------------------- C6：G5 判定
def test_gate_g5() -> None:
    """`50` [S5] C6：G5 两档分别四值判定 + 组合规则（含 4×4 组合表）。"""
    reference = _e15_bconst_station({b: 0.5 for b in _E15_BEAMS})

    def _tier_evals(deltas: dict[int, float]) -> list[dict]:
        return [
            _e15_eval(f"e15_joint_s{i}", variant="joint", seed=100 + i, tier=1, station=0,
                      err_by_beam={b: 0.5 - d for b, d in deltas.items()})
            for i in range(5)
        ]

    strong = {b: 0.4 for b in _E15_BEAMS}
    weak = {b: 0.2 for b in _E15_BEAMS[:134]} | {b: -1e-9 for b in _E15_BEAMS[134:]}
    zero = {b: 0.1 for b in _E15_BEAMS[:75]} | {b: -0.1 for b in _E15_BEAMS[75:]}
    bad = {b: -0.4 for b in _E15_BEAMS}

    both_pass = phase15_gates.gate_g5(
        {3_000: _tier_evals(strong), 100_000: _tier_evals(strong)}, reference
    )
    assert both_pass["verdict"] == "成立"
    assert both_pass["tier_verdicts"] == {"3000": "成立", "100000": "成立"}
    assert set(both_pass["per_tier"]) == {"3000", "100000"}

    mixed = phase15_gates.gate_g5(
        {3_000: _tier_evals(strong), 100_000: _tier_evals(weak)}, reference
    )
    assert (mixed["tier_verdicts"], mixed["verdict"]) == (
        {"3000": "成立", "100000": "方向成立"}, "方向成立"
    )
    indeterminate = phase15_gates.gate_g5(
        {3_000: _tier_evals(strong), 100_000: _tier_evals(zero)}, reference
    )
    assert indeterminate["verdict"] == "不可判定"
    negative = phase15_gates.gate_g5(
        {3_000: _tier_evals(strong), 100_000: _tier_evals(bad)}, reference
    )
    assert negative["verdict"] == "不成立"
    # ①优先于②：一档不成立 + 一档不可判定 ⇒ 不成立
    priority = phase15_gates.gate_g5(
        {3_000: _tier_evals(bad), 100_000: _tier_evals(zero)}, reference
    )
    assert priority["verdict"] == "不成立"
    # 档缺失 → 该档不可判定（不静默丢弃该档）
    absent = phase15_gates.gate_g5({3_000: _tier_evals(strong)}, reference)
    assert absent["verdict"] == "不可判定"
    assert absent["per_tier"]["100000"]["reason_code"] == "eval_missing"

    # 组合规则表（4×4 全枚举，与 `50` [S5] C6 的四条规则逐格一致）
    rule = {
        ("成立", "成立"): "成立",
        ("不成立", "成立"): "不成立", ("成立", "不成立"): "不成立",
        ("不成立", "不成立"): "不成立",
        ("不可判定", "成立"): "不可判定", ("成立", "不可判定"): "不可判定",
        ("不可判定", "不可判定"): "不可判定", ("不可判定", "不成立"): "不成立",
        ("不成立", "不可判定"): "不成立",
        ("方向成立", "成立"): "方向成立", ("成立", "方向成立"): "方向成立",
        ("方向成立", "方向成立"): "方向成立",
        ("方向成立", "不可判定"): "不可判定", ("方向成立", "不成立"): "不成立",
        ("不可判定", "方向成立"): "不可判定", ("不成立", "方向成立"): "不成立",
    }
    for (a, b), expected in rule.items():
        assert phase15_gates.g5_combine({3_000: a, 100_000: b}) == expected, (a, b)


# ---------------------------------------------------- C7：e15_gates.json 载荷与纪律
def test_e15_gates_payload(tmp_path: Path) -> None:
    """`50` [S5] C7：五门禁结论 + 每条输入快照 + routing + 暗桶，脚本化且逐位可复现。"""
    results = _write_phase15_results(tmp_path)
    payload = phase15_gates.evaluate_e15_gates_from_disk(results)

    # 五条结论 + 每条门禁的判定表达式输入快照
    assert payload["verdicts"] == {
        "G1": "成立", "G2": "成立", "G3": "成立", "G4": "成立", "G5": "成立",
    }
    for name in ("G1", "G2", "G3", "G4", "G5"):
        gate = payload[name]
        assert gate["gate"] == name and gate["verdict"] == "成立"
        assert gate["thresholds"] and "source" in gate
    assert payload["G1"]["win_rate"] == 1.0 and len(payload["G1"]["inputs"]["delta"]) == 12
    assert payload["G3"]["pass_rate"] == 1.0 and len(payload["G3"]["per_observable"]) == 13
    assert payload["G4"]["r_diff"] == pytest.approx(0.5)
    assert set(payload["G5"]["tier_verdicts"]) == {"3000", "100000"}
    assert payload["G2"]["k_a"] == 5 and payload["G2"]["k_and"] == 5

    # routing（获胜臂 + 依据 val_nll）与暗桶 M2 分项
    assert payload["routing"]["basis"] == "val_nll"
    assert payload["routing"]["winner_arm"] == "joint"
    assert payload["routing"]["arm_reading"]["joint"] < payload["routing"]["arm_reading"]["independent"]
    assert payload["routing"]["reading_disagreement"] is False
    dark = payload["dark_bucket_m2"]
    assert set(dark) >= {"B0", "B1", "B0B1_combined", "aggregation", "runs_without_m2"}
    assert dark["B1"]["relative_improvement"] == pytest.approx(0.5)
    assert payload["products"]["bconst"]["stations"] == [0, 1]

    # 落盘 == 返回；同输入两次调用逐位一致（脚本化判定的确定性）
    gates_path = results / "e15_gates.json"
    assert json.loads(gates_path.read_text(encoding="utf-8")) == payload
    first = gates_path.read_bytes()
    again = phase15_gates.evaluate_e15_gates_from_disk(results)
    assert gates_path.read_bytes() == first and again == payload

    # 汇总统计：3 档 × 2 站 × 5 run（获胜臂的 6 评估/run 全被消费）
    snapshot = payload["inputs_snapshot"]
    assert len(snapshot["gate_evals"]) == 5 and len(snapshot["cross_station_evals"]) == 5
    assert sum(len(v) for v in snapshot["g5_evals"].values()) == 10

    # 两段式纪律：门禁只吃 test 段（把一份被消费的评估改成 val 段即报错）
    victim = phase15_gates.eval_path(results, "e15_joint_s0", 10_000, 0)
    original = victim.read_text(encoding="utf-8")
    broken = json.loads(original)
    broken["split"] = "val"
    victim.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="只消费 test 段"):
        phase15_gates.evaluate_e15_gates_from_disk(results)
    victim.write_text(original, encoding="utf-8")

    # 路由只吃 val 段：summary.json 以外的路径一律拒绝
    with pytest.raises(ValueError, match="val 段"):
        phase15_gates.assert_val_segment_paths([victim])
    with pytest.raises(ValueError, match="test 段"):
        phase15_gates.assert_test_segment_paths([results / "e15_joint_s0" / "summary.json"])
    # 缺 routing 时门禁拒绝启动（不猜获胜臂）
    (results / "e15_routing.json").unlink()
    with pytest.raises(FileNotFoundError):
        phase15_gates.evaluate_e15_gates_from_disk(results)

    # 判定侧静态纪律：phase15_gates 不出现 N_eff 类标识符（`50` [S3] C1 / [S5] p4）
    tree = ast.parse(Path(phase15_gates.__file__).read_text(encoding="utf-8"))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    offenders = sorted(name for name in identifiers if "n_eff" in name.lower())
    assert not offenders, f"门禁模块出现 N_eff 标识符：{offenders}"


def test_e15_routing_val_segment_and_reading_disagreement(tmp_path: Path) -> None:
    """`50` [S5] 路由：只读 val 段 val_nll、臂级取 5 种子中位数、两读法不一致时显式登记。"""
    results = tmp_path / "results"
    results.mkdir()
    for tag, variant, val_nll, tier_1e4 in (
        ("joint", "joint", 1.0, 0.9), ("indep", "independent", 1.2, 2.0),
    ):
        for index, seed in enumerate(_PHASE15_SEEDS):
            _write_e15_summary(
                results / f"e15_{tag}_s{index}", variant=variant, seed=seed,
                val_nll=val_nll, tier_1e4=tier_1e4,
            )
    routing = phase15_gates.write_routing(
        results, {"joint": [f"e15_joint_s{i}" for i in range(5)],
                  "independent": [f"e15_indep_s{i}" for i in range(5)]}
    )
    assert routing["winner_arm"] == "joint"          # 三档等权读数（[S3] C2）占优
    assert routing["winner_by_tier_1e4"] == "joint"
    assert routing["reading_disagreement"] is False
    assert routing["arm_reading"]["joint"] == pytest.approx(1.0)
    assert sorted(routing["per_arm"]["joint"]["per_seed_val_nll"]) == [str(s) for s in _PHASE15_SEEDS]

    # 1e4 单档读法的胜者相反 ⇒ 显式登记不一致（不静默择一）
    for index in range(5):
        _write_e15_summary(results / f"e15_joint_s{index}", variant="joint", seed=100 + index,
                           val_nll=1.0, tier_1e4=2.5)
        _write_e15_summary(results / f"e15_indep_s{index}", variant="independent", seed=100 + index,
                           val_nll=1.2, tier_1e4=2.0)
    flipped = phase15_gates.write_routing(
        results, {"joint": [f"e15_joint_s{i}" for i in range(5)],
                  "independent": [f"e15_indep_s{i}" for i in range(5)]}
    )
    assert flipped["winner_arm"] == "joint" and flipped["winner_by_tier_1e4"] == "independent"
    assert flipped["reading_disagreement"] is True
    assert json.loads((results / "e15_routing.json").read_text(encoding="utf-8")) == flipped


# ------------------------------------------------------------------ C9：v2 渲染
def test_map_report_v2(tmp_path: Path) -> None:
    """`50` [S5] C9：map_report 适配 v2 路径并渲染 e15_gates.json（G1–G5、B-const、暗桶）。"""
    from src.line2.experiments.map_report import generate_v2, load_e15_gates

    results = _write_phase15_results(tmp_path)
    phase15_gates.evaluate_e15_gates_from_disk(results)
    reports = tmp_path / "reports"

    # v1 路径缺失不算数：v2 剖面只认 *_v2 产物（`50` [S5] C9 的路径适配）
    v1_only = tmp_path / "v1_only"
    v1_only.mkdir()
    (v1_only / "e0_curve.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="e15_gates.json"):
        generate_v2(v1_only, reports_dir=tmp_path / "r1")

    first = generate_v2(results, reports_dir=reports)
    digest = hashlib.sha256(first["report"].read_bytes()).hexdigest()
    text = first["report"].read_text(encoding="utf-8")
    assert first["report"] == reports / "phase15_gates_report.md"
    assert first["winner_arm"] == "joint" and first["n_evals"] == 20

    # G1–G5 各自成节、结论字面量 + 判定表达式 + 判定表达式的标量输入
    for index, name in enumerate(("G1", "G2", "G3", "G4", "G5"), start=1):
        assert f"### 3.{index} {name}：**成立**" in text, name
    assert text.count("判定表达式（`50` [S5]）：") == 5
    for row in ("| win_rate | 1 |", "| advantage |", "| k_a | 5 |", "| k_and | 5 |",
                "| pass_rate | 1 |", "| r_diff | 0.5 |", "| raw_err_ratio | 1.25 |",
                "| tier_verdicts | 100000=成立, 3000=成立 |"):
        assert row in text, row
    # v2 路径适配：报告引用 v2 产物名而非 v1
    assert "e0_curve_v2.json" in text and "tolerance_v2.json" in text
    assert "bconst_v2.json" in text and "e15_gates.json" in text
    assert "e0_curve.json`" not in text.replace("e0_curve_v2.json", "")
    # B-const（两站）与暗桶 M2 单独成节
    assert "## 4. B-const（零信息常数解，两站）" in text
    assert "| 0 |" in text and "| 1 |" in text
    assert "## 5. 暗桶 M2（单独报告，不并入 G1–G5）" in text
    assert "B0B1_combined" in text and "0.5" in text
    # 路由段披露获胜臂与两读法一致性
    assert "获胜臂：**joint**" in text and "两种读法一致" in text

    # 只读：输入目录逐字不变、只落一份报告
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(results.rglob("*")) if p.is_file()}
    second = generate_v2(results, reports_dir=reports)
    assert hashlib.sha256(second["report"].read_bytes()).hexdigest() == digest
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(results.rglob("*")) if p.is_file()}
    assert before == after
    assert sorted(p.name for p in reports.rglob("*")) == ["phase15_gates_report.md"]

    # 门禁产物缺段 ⇒ 报错，不静默省略
    gates_path = results / "e15_gates.json"
    payload = json.loads(gates_path.read_text(encoding="utf-8"))
    del payload["dark_bucket_m2"]
    gates_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dark_bucket_m2"):
        load_e15_gates(results)
    payload["G3"] = dict(payload["G3"], verdict="也许")
    payload["dark_bucket_m2"] = {"B0": {}}
    gates_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="合法枚举"):
        load_e15_gates(results)


# ------------------------------------ 矩阵 config 生成 + val_beams 守卫（[S5] 矩阵）
def test_phase15_configs_and_val_beams_guard(tmp_path: Path) -> None:
    """`50` [S5]：configs 生成器落 10 份 config + 矩阵记录；`val_beams` 只许缺省或 150。"""
    from src.line2.training.config_check import validate_phase15_matrix

    data_dir = _write_tiny_field_dataset(tmp_path / "ds")
    out = phase15_configs.generate(
        results_dir=tmp_path / "results", data_dir=data_dir, code_version="b" * 40
    )
    assert out["configs"] and len(out["configs"]) == 10
    configs_dir = Path(out["config_dir"])
    assert sorted(p.stem for p in configs_dir.glob("e15_*.json")) == sorted(phase15_configs.run_ids())
    assert out["validation"]["station_check"] == f"ok:{data_dir}"
    assert (tmp_path / "results" / "e15_matrix.json").is_file()
    cfg = json.loads((configs_dir / "e15_joint_s0.json").read_text(encoding="utf-8"))
    for key, value in (
        ("phase", "1.5"), ("kappa", "v2"), ("data_version", "line2_ds_v2"),
        ("spec_version", "line2-spec-v1.1"), ("code_version", "b" * 40),
        ("stations", [0]), ("k_train", 5), ("tiers", [3000, 10000, 100000]),
    ):
        assert cfg[key] == value, key
    assert "val_beams" not in cfg                     # `50` [S3] C2：val 集不截短
    assert cfg["out_dir"].endswith("results/e15_joint_s0")

    # `val_beams` 守卫：缺省 / 恰 150 通过，其余拒绝（`50` [S3] C2 的 val_nll 定义域）
    matrix = phase15_configs.build_matrix(
        results_dir=tmp_path / "results", data_dir=data_dir, code_version="b" * 40
    )
    validate_phase15_matrix(matrix, data_dir=data_dir)
    validate_phase15_matrix(
        [dict(cfg_, val_beams=150) for cfg_ in matrix], data_dir=data_dir
    )
    for bad in (1, 100, 200):
        with pytest.raises(ValueError, match="val_beams"):
            validate_phase15_matrix(
                [dict(cfg_, val_beams=bad) for cfg_ in matrix], data_dir=data_dir
            )


# ------------------------------------------------- 双卡队列（按臂整分）与编排入口
def test_plan_arm_queues_assigns_one_arm_per_gpu() -> None:
    """`50` [S5]：双卡队列按臂整分（joint → GPU0、independent → GPU1），run 不重不漏。"""
    runs = phase15_configs.run_ids()
    queue = phase15_orchestrator.plan_arm_queues(runs, [0, 1])
    assert queue[0] == [f"e15_joint_s{i}" for i in range(5)]
    assert queue[1] == [f"e15_indep_s{i}" for i in range(5)]
    assert sorted(queue[0] + queue[1]) == sorted(runs)
    single = phase15_orchestrator.plan_arm_queues(runs, [0])
    assert single[0] == [f"e15_joint_s{i}" for i in range(5)] + [
        f"e15_indep_s{i}" for i in range(5)
    ]
    with pytest.raises(ValueError, match="臂标记"):
        phase15_orchestrator.plan_arm_queues(["e15_x_s0"], [0])
    with pytest.raises(ValueError, match="重复"):
        phase15_orchestrator.plan_arm_queues(["e15_joint_s0", "e15_joint_s0"], [0])
