"""生成 seed2/3 的 EXP-02 config（基于 seed0 模板扩展 scheme_seeds + 改 seed_index）。

每个 run 独立 config.yaml，目录结构沿用 EXP-02_<arm>_seed<N>_run1_D2/。
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

import yaml

ROOT = Path("/home/zhangny/Super-Resolution-Simulation")

# 新种子的 scheme_seeds（与 seed0/1 一致风格，避开已有值）
EXTRA_SCHEME_SEEDS = {
    "scheme_A_seed_2": 55555, "scheme_A_seed_3": 66666,
    "scheme_B_seed_2": 77777, "scheme_B_seed_3": 88888,
    "scheme_C_seed_2": 99999, "scheme_C_seed_3": 10101,
}

NEW_SEEDS = [2, 3]


def gen_one(arm: str, new_seed: int) -> Path:
    src = ROOT / f"results/EXP-02_{arm}_seed0_run1_D2/config.yaml"
    target_dir = ROOT / f"results/EXP-02_{arm}_seed{new_seed}_run1_D2"
    target_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(src.read_text())
    cfg["seed_index"] = new_seed

    # 扩展 scheme_seeds
    for k, v in EXTRA_SCHEME_SEEDS.items():
        if not k.startswith(f"scheme_{arm}_"):
            continue
        cfg["scheme_seeds"][k] = v

    out_cfg = target_dir / "config.yaml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    # 确保 logs/ 目录存在（train.py 启动时不会自动建）
    (target_dir / "logs").mkdir(exist_ok=True)
    print(f"wrote {out_cfg}")
    return out_cfg


def main() -> None:
    for arm in "ABC":
        for s in NEW_SEEDS:
            gen_one(arm, s)
    # 快速校验
    for arm in "ABC":
        for s in NEW_SEEDS:
            p = ROOT / f"results/EXP-02_{arm}_seed{s}_run1_D2/config.yaml"
            c = yaml.safe_load(p.read_text())
            assert c["seed_index"] == s, f"seed_index mismatch in {p}"
            assert c["scheme_seeds"][f"scheme_{arm}_seed_{s}"] == EXTRA_SCHEME_SEEDS[f"scheme_{arm}_seed_{s}"]
    print("all configs verified: seed_index + scheme_seeds OK")


if __name__ == "__main__":
    main()