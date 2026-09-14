"""L4 [S1] C4：E2 config 生成脚本的测试（spec line2/40）。

全部用 `tmp_path` 造假 run 目录与假 E1 config 模板，不依赖真实实验产物（E1 仍在跑）。
路由分支按 `40` [S3] 的四种计数分支逐条覆盖（3/5 的两个出口各一条）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from src.line2.experiments import make_e2_configs as e2c

pytestmark = pytest.mark.unit

SEEDS = (100, 101, 102, 103, 104)

#: 假 E1 config 的公平性键取值（形状照 `docs/specs/line2/e1_configs/`）
E1_FAIRNESS = {
    "data_dir": "studies/line2_particle_sr/data/line2_ds_v1",
    "n_high": 200000.0,
    "max_steps": 50000,
    "eval_every": 1000,
    "patience": 10,
    "batch": 16,
    "lr": 0.0003,
    "w_psd": 0.1,
    "device": "cuda:0",
    "torch_threads": 4,
}
#: 变体拓扑键（`30` [S4] C2：变体间差异只在网络拓扑）
E1_TOPOLOGY = {
    "joint": {"base": 48, "num_levels": 4, "blocks": 2},
    "independent": {"base": 24, "num_levels": 4, "blocks": 1,
                    "ind_base": 16, "ind_blocks": 3},
}


class Case(NamedTuple):
    results: Path
    e1_configs: Path
    out: Path


def _write_run(
    results: Path,
    run_id: str,
    variant: str,
    seed: int,
    val_n_eff: float,
    n_low: int = 10000,
    *,
    with_summary: bool = True,
    with_run_meta: bool = True,
) -> Path:
    run_dir = results / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if with_run_meta:
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id, "variant": variant, "seed": seed, "n_low": n_low,
                    "n_high": 200000.0, "max_steps": 50000,
                    "code_version": "a" * 40, "data_version": "line2_ds_v1",
                    "spec_version": "line2-spec-v1.0",
                }
            ),
            encoding="utf-8",
        )
    if with_summary:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "variant": variant, "seed": seed, "best_val_n_eff": val_n_eff,
                    "val_n_eff_history": [val_n_eff], "steps": 50000,
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def _write_e1_config(
    configs: Path, run_id: str, variant: str, seed: int, n_low: int = 10000
) -> Path:
    configs.mkdir(parents=True, exist_ok=True)
    cfg = {
        **E1_FAIRNESS,
        **E1_TOPOLOGY[variant],
        "n_low": n_low,
        "seed": seed,
        "variant": variant,
        "out_dir": f"studies/line2_particle_sr/results/{run_id}",
    }
    path = configs / f"{run_id}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_arm(
    results: Path,
    configs: Path,
    variant: str,
    seeds: Sequence[int],
    vals: Sequence[float],
    n_low: int = 10000,
) -> None:
    tag = e2c.VARIANT_RUN_TAG[variant]
    for idx, (seed, val) in enumerate(zip(seeds, vals)):
        run_id = f"e1_{tag}_s{idx}"
        _write_run(results, run_id, variant, seed, val, n_low)
        _write_e1_config(configs, run_id, variant, seed, n_low)


def _case(
    tmp_path: Path,
    joint_vals: Sequence[float],
    indep_vals: Sequence[float],
    *,
    joint_seeds: Sequence[int] = SEEDS,
    indep_seeds: Sequence[int] | None = None,
) -> Case:
    results = tmp_path / "results"
    configs = tmp_path / "e1_configs"
    _write_arm(results, configs, "joint", joint_seeds, joint_vals)
    _write_arm(results, configs, "independent", indep_seeds or joint_seeds, indep_vals)
    return Case(results, configs, tmp_path / "e2_configs")


def _generate(case: Case, **kwargs) -> dict:
    return e2c.generate(case.results, out_dir=case.out, e1_config_dir=case.e1_configs, **kwargs)


def _configs_by_key(metas: Sequence[dict]) -> dict[tuple[str, int, int], dict]:
    return {
        (meta["variant"], meta["n_low"], meta["seed"]): json.loads(
            Path(meta["path"]).read_text(encoding="utf-8")
        )
        for meta in metas
    }


# ------------------------------------------------- `40` [S3] 变体路由：四种计数分支
def test_route_5_of_5_joint(tmp_path: Path) -> None:
    """5/5 对联合 ≥ 独立 → 联合。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    payload = _generate(case, dry_run=True)
    assert (payload["winner_variant"], payload["winner_branch"]) == ("joint", "5/5")
    assert payload["joint_wins"] == 5
    assert payload["h3_val_verdict"] == "成立"


def test_route_4_of_5_joint(tmp_path: Path) -> None:
    """4/5 → 联合（方向成立）。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 9000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    payload = _generate(case, dry_run=True)
    assert (payload["winner_variant"], payload["winner_branch"]) == ("joint", "4/5")
    assert payload["joint_wins"] == 4
    assert payload["h3_val_verdict"] == "方向成立"


def test_route_3_of_5_joint_median_higher(tmp_path: Path) -> None:
    """3/5 且联合 val 中位数较高 → 联合，并标注 H3 不可判定（[S3] 3/5 裁定的联合出口）。"""
    case = _case(
        tmp_path,
        [20000.0, 20000.0, 20000.0, 5000.0, 6000.0],     # 3 胜；中位数 20000
        [13000.0, 10000.0, 18000.0, 20000.0, 19000.0],   # 中位数 18000
    )
    payload = _generate(case, dry_run=True)
    assert (payload["winner_variant"], payload["winner_branch"]) == ("joint", "3/5")
    assert payload["joint_wins"] == 3
    assert payload["h3_val_verdict"] == "不可判定"
    assert "H3 不可判定" in payload["branch_rule"]
    assert payload["val_medians"] == {"joint": 20000.0, "independent": 18000.0}


def test_route_3_of_5_independent_median_higher(tmp_path: Path) -> None:
    """3/5 且独立 val 中位数较高 → 独立（[S3] 3/5 裁定的独立出口）。"""
    case = _case(
        tmp_path,
        [14000.0, 11000.0, 19000.0, 5000.0, 6000.0],     # 3 胜；中位数 11000
        [13000.0, 10000.0, 18000.0, 20000.0, 19000.0],   # 中位数 18000
    )
    payload = _generate(case, dry_run=True)
    assert (payload["winner_variant"], payload["winner_branch"]) == ("independent", "3/5")
    assert payload["joint_wins"] == 3
    assert payload["h3_val_verdict"] == "不可判定"


def test_route_2_of_5_independent(tmp_path: Path) -> None:
    """≤2/5 → 独立。"""
    case = _case(
        tmp_path,
        [8000.0, 9000.0, 7000.0, 12000.0, 6000.0],
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
    )
    payload = _generate(case, dry_run=True)
    assert (payload["winner_variant"], payload["winner_branch"]) == ("independent", "<=2/5")
    assert payload["joint_wins"] == 0
    assert payload["h3_val_verdict"] == "不成立"


# ------------------------------------------------- 路由输入的拒绝条件
def test_route_rejects_mismatched_seed_sets(tmp_path: Path) -> None:
    """两臂种子集合不同 → 报错（不退化为槽位配对；`40` [S3] 同种子成对）。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
        indep_seeds=(200, 201, 202, 203, 204),
    )
    with pytest.raises(ValueError, match="两臂种子集合不同"):
        _generate(case, dry_run=True)
    assert not case.out.exists()


def test_route_rejects_unfrozen_seed_set(tmp_path: Path) -> None:
    """两臂种子集合相同但非冻结档 100–104 → 报错（E2 的 _s<idx> 与种子无对应关系）。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
        joint_seeds=(0, 1, 2, 3, 4),
    )
    with pytest.raises(ValueError, match="种子档"):
        _generate(case, dry_run=True)


def test_route_indeterminate_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """路由结论为 indeterminate → 报错退出，不自行挑选变体（`40` [S1] C4）。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    monkeypatch.setattr(e2c, "winner_variant", lambda joint, independent: "indeterminate")
    with pytest.raises(RuntimeError, match="indeterminate"):
        _generate(case, dry_run=True)
    assert not case.out.exists()


def test_arm_with_four_seeds_raises(tmp_path: Path) -> None:
    """某臂只有 4 个种子 → 报错（`40` [S1] 每变体 5 种子）。"""
    results = tmp_path / "results"
    configs = tmp_path / "e1_configs"
    _write_arm(results, configs, "joint", SEEDS, [20000.0] * 5)
    _write_arm(results, configs, "independent", SEEDS[:4], [12000.0] * 4)
    with pytest.raises(ValueError, match="independent 臂只有 4 个种子"):
        e2c.generate(results, out_dir=tmp_path / "out", e1_config_dir=configs, dry_run=True)


def test_missing_summary_raises(tmp_path: Path) -> None:
    """某 run 缺 summary.json → 报错，不静默跳过。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    (case.results / "e1_indep_s2" / "summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="summary.json"):
        _generate(case, dry_run=True)


def test_missing_run_meta_raises(tmp_path: Path) -> None:
    """某 run 缺 run_meta.json → 报错，不静默跳过。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    (case.results / "e1_joint_s0" / "run_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_meta.json"):
        _generate(case, dry_run=True)


def test_stale_e1_config_copy_raises(tmp_path: Path) -> None:
    """E1 config 模板与 run_meta 的种子不符（陈旧副本）→ 报错。"""
    case = _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )
    path = case.e1_configs / "e1_joint_s3.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["seed"] = 0
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="不是同一次训练"):
        _generate(case, dry_run=True)


# ------------------------------------------------- 10 份 config 的命名与内容
def _joint_case(tmp_path: Path) -> Case:
    return _case(
        tmp_path,
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
        [12000.0, 13000.0, 11000.0, 14000.0, 12000.0],
    )


def test_generate_writes_ten_configs_with_expected_names(tmp_path: Path) -> None:
    """获胜变体 × {3e3, 1e5} × 5 种子 = 10 份 config，命名/档位/out_dir 按 `40` [S1]。"""
    case = _joint_case(tmp_path)
    payload = _generate(case)
    assert len(payload["configs"]) == 10
    assert payload["n_low_tiers"] == [3000, 100000]
    expected = {f"e2_joint_n{n}_s{i}" for n in (3000, 100000) for i in range(5)}
    assert {meta["run_id"] for meta in payload["configs"]} == expected
    written = {path.stem for path in case.out.glob("*.json")}
    assert written == expected
    for meta in payload["configs"]:
        assert Path(meta["path"]).name == f"{meta['run_id']}.json"
        assert Path(meta["path"]).parent == case.out
        out_dir = Path(meta["out_dir"])
        assert out_dir.name == meta["run_id"]                       # run 目录与 E1 同级
        assert out_dir.parent == case.results
        cfg = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))
        assert cfg["out_dir"] == meta["out_dir"]
        assert cfg["variant"] == "joint" == meta["variant"]
        assert cfg["seed"] == meta["seed"] == 100 + int(meta["run_id"].rsplit("_s", 1)[1])
        assert cfg["n_low"] == meta["n_low"] in (3000, 100000)


def test_generated_configs_differ_from_e1_only_in_tier_seed_out_dir(tmp_path: Path) -> None:
    """GC4 公平性（`40` [S1] C3）：10 份 config 与 E1 同变体模板只在 n_low/seed/out_dir 上不同。"""
    case = _joint_case(tmp_path)
    payload = _generate(case)
    template = json.loads((case.e1_configs / "e1_joint_s0.json").read_text(encoding="utf-8"))
    configs = _configs_by_key(payload["configs"])
    assert len(configs) == 10
    for (_, n_low, seed), cfg in configs.items():
        diff = {key for key in set(cfg) | set(template) if cfg.get(key) != template.get(key)}
        assert diff <= {"seed", "n_low", "out_dir"}
        assert {"n_low", "out_dir"} <= diff          # 档位与落盘目录必然不同
        assert ("seed" in diff) == (seed != 100)     # 种子 100 即模板本身
        for key in e2c.E2_FAIRNESS_KEYS:
            assert cfg[key] == template[key], key
    assert payload["fairness_keys"] == list(e2c.E2_FAIRNESS_KEYS)
    # 同档内跨种子只差身份与落盘目录；跨档只多出 n_low
    for n_low in (3000, 100000):
        ref = configs[("joint", n_low, 100)]
        for seed in SEEDS[1:]:
            cfg = configs[("joint", n_low, seed)]
            assert {k for k in cfg if cfg[k] != ref[k]} == {"seed", "out_dir"}
    cross = configs[("joint", 3000, 100)]
    other = configs[("joint", 100000, 100)]
    assert {k for k in other if other[k] != cross[k]} == {"n_low", "out_dir"}


def test_independent_winner_uses_independent_template(tmp_path: Path) -> None:
    """获胜变体为独立版时，E2 config 用独立版模板（含 ind_base/ind_blocks，无 joint 拓扑）。"""
    case = _case(
        tmp_path,
        [8000.0, 9000.0, 7000.0, 12000.0, 6000.0],
        [20000.0, 22000.0, 18000.0, 25000.0, 21000.0],
    )
    payload = _generate(case)
    assert payload["winner_variant"] == "independent"
    assert Path(payload["template"]["config"]).name == "e1_indep_s0.json"
    assert {Path(meta["path"]).name for meta in payload["configs"]} == {
        f"e2_independent_n{n}_s{i}.json" for n in (3000, 100000) for i in range(5)
    }
    one = json.loads(Path(payload["configs"][0]["path"]).read_text(encoding="utf-8"))
    assert one["variant"] == "independent"
    assert "ind_base" in one and "ind_blocks" in one
    assert all(
        one[key] == E1_FAIRNESS[key] for key in e2c.E2_FAIRNESS_KEYS
    )


# ------------------------------------------------- dry-run 与路由记录
def test_dry_run_writes_nothing_but_same_decision(tmp_path: Path) -> None:
    """`--dry-run`：不写任何文件，但路由结论与 config 清单与真跑一致。"""
    case = _joint_case(tmp_path)
    dry = _generate(case, dry_run=True)
    assert not case.out.exists()
    assert not (case.results / e2c.ROUTING_FILENAME).exists()
    real = _generate(case)
    assert (Path(real["routing_path"]).is_file()
            and real["routing_path"] == str(case.results / e2c.ROUTING_FILENAME))
    stripped = {k: v for k, v in real.items() if k != "dry_run"}
    assert {k: v for k, v in dry.items() if k != "dry_run"} == stripped
    assert (dry["dry_run"], real["dry_run"]) == (True, False)


def test_routing_record_fields_and_json_round_trip(tmp_path: Path) -> None:
    """路由记录字段齐全、可 json round-trip（同输入两次生成逐位一致）。"""
    case = _joint_case(tmp_path)
    payload = _generate(case)
    text = (case.results / e2c.ROUTING_FILENAME).read_text(encoding="utf-8")
    record = json.loads(text)
    assert record == payload
    assert json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n" == text
    required = {
        "results_dir", "basis", "source", "pairing", "n_pairs", "seeds", "val_n_eff",
        "val_medians", "joint_wins", "h3_val_verdict", "winner_branch", "branch_rule",
        "winner_variant", "e1_n_low", "e1_runs", "n_low_tiers", "config_dir",
        "routing_path", "configs", "dry_run",
    }
    assert required <= set(record)
    assert record["basis"] == "val"                       # [S3] 两段式：路由只消费 val 段
    assert record["source"]["summary"] == "summary.json:best_val_n_eff"
    assert record["n_pairs"] == 5
    assert record["pairing"] == {
        "basis": "seed_value", "pairs": [[s, s] for s in SEEDS],
        "seed_sets": {"joint": list(SEEDS), "independent": list(SEEDS)},
    }
    assert set(record["val_n_eff"]) == {"joint", "independent"}
    assert set(record["val_n_eff"]["joint"]) == {str(s) for s in SEEDS}
    assert record["val_n_eff"]["joint"]["100"] == 20000.0
    assert set(record["e1_runs"]["independent"]) == {str(s) for s in SEEDS}
    assert record["e1_runs"]["joint"]["104"]["run_id"] == "e1_joint_s4"
    assert {c["variant"] for c in record["configs"]} == {"joint"}
    assert all({"path", "variant", "seed", "n_low", "out_dir"} <= set(c)
               for c in record["configs"])


# ------------------------------------------------- CLI
def test_cli_dry_run_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI 正常路径：`--dry-run` 返回 0、打印路由结论、不落盘。"""
    case = _joint_case(tmp_path)
    rc = e2c.main([
        "--results-dir", str(case.results), "--out-dir", str(case.out),
        "--e1-config-dir", str(case.e1_configs), "--dry-run",
    ])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["winner_variant"] == "joint"
    assert (body["n_configs"], body["n_pairs"]) == (10, 5)
    assert body["dry_run"] is True
    assert not case.out.exists()


def test_cli_missing_products_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI 缺输入：退出码非 0，不静默生成。"""
    results = tmp_path / "results"
    results.mkdir()
    rc = e2c.main(["--results-dir", str(results)])
    assert rc != 0
    assert "make_e2_configs" in capsys.readouterr().err
