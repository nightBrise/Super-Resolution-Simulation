"""EXP-03/04/07/08 测试集生成（80 [S6]，60 [S15] C4b 交叉生成工具归 src/generators/）。

对 ``test_id ∪ test_pb`` 的**同一 H** 生成推理级消融测试集，写出与主划分
同 schema 的 ``test_exp0N.h5``（顶层数组 + c/m/masks/m_L 组，60 [S14] 契约，
H5Dataset 可直接读取评估）：

- EXP-03 强模糊（σ_K=22.0、σ_n=D2）——H 重退化，P2 先验沿用源值；
- EXP-04 高噪声（σ_n=OQ-30-04 采用值 1.133e-3、σ_K=D2）——H 重退化；
- EXP-07 先验信息量消融（D2）——**不重退化**（L/L_clean/L_up/m_L 逐位沿用
  源划分，保证 Gain(P2)−Gain(P1) 只归因于先验信息量），先验槽替换为 P1
  （仅 c_low，40 [S6]）；
- EXP-08 噪声敏感度（D2）——同一 H 以 K=8 个独立噪声实现重退化，样本 id
  加 ``_r{k}`` 后缀（80 [S6] 预测离散度 = 同一样本跨实现的不确定性代理）。

噪声实现经**新 SeedSequence 分支**派生（exp03→5、exp04→6、exp07→7、
exp08→8，均未被 SPLIT_BRANCH 0–4 占用）——60 [S8] C4：同一 c 的噪声实现
不得跨划分复用，EXP-0N 的噪声 SHALL 与 test_id/test_pb 及彼此不相交
（EXP-07 无新噪声：其 L 逐位沿用源划分）。

用法：:

    python -m src.generators.build_exp34 --exp 03 [--version v1] [--workers N]
    python -m src.generators.build_exp34 --exp 04 --sigma-n <采用值>
    python -m src.generators.build_exp34 --exp 07
    python -m src.generators.build_exp34 --exp 08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.generators.dataset_builder import (
    _create_split_file,
    _write_batch,
    DEFAULT_DOWNSAMPLE,
    DELTA_PX,
    SIGMA_SMOOTH_P_MULTIPLE,
    fine_structure_width,
    git_head,
    resolve_spec_version,
)
from src.generators.f_deg import f_deg
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 重退化噪声的 SeedSequence 分支（60 [S8] C4 不相交；SPLIT_BRANCH 已占用 0–4）。
#: exp03/04 沿用 spawn(8)[5/6]（工件已落盘，改动破坏可复现性）；exp08 走
#: 分支 7 的深层子代（_derive_exp_seeds）；exp07 无新噪声（复制路径）不派生。
EXP_SEED_BRANCH = {"exp03": 5, "exp04": 6, "exp07": 7}

#: D2 标定采用值（EXP-01d 登记；EXP-07/08 沿用）。
D2_SIGMA_K = 11.0
D2_SIGMA_N = 1.22e-4

#: 30 [S8] EXP-03 预设（σ_K = 2×D2 标定值、σ_n = D2）。
EXP03_SIGMA_K = 22.0

#: 各 EXP 生成预设：sigma_n=None 表示 CLI 必填；redegrade=False 表示 L 沿用源值。
EXP_PRESETS: dict[str, dict[str, Any]] = {
    "exp03": {"sigma_k": EXP03_SIGMA_K, "sigma_n": D2_SIGMA_N, "prior_level": "P2", "reps": 1, "redegrade": True},
    "exp04": {"sigma_k": D2_SIGMA_K, "sigma_n": None, "prior_level": "P2", "reps": 1, "redegrade": True},
    "exp07": {"sigma_k": D2_SIGMA_K, "sigma_n": D2_SIGMA_N, "prior_level": "P1", "reps": 1, "redegrade": False},
    "exp08": {"sigma_k": D2_SIGMA_K, "sigma_n": D2_SIGMA_N, "prior_level": "P2", "reps": 8, "redegrade": True},
}

#: 源划分（80 [S6]：同一 H、同一划分，逐样本配对）。
SOURCE_SPLITS = ("test_id", "test_pb")


def _decode_if_bytes(value: Any) -> Any:
    """h5py 字符串数据集逐元素读回 bytes → str（60 [S14] 契约，utf-8）。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_source_fields(h5_path: Path) -> dict[str, list[Any]]:
    """读源划分的全部字段（按样本切分的列表，供记录组装）。"""
    fields: dict[str, list[Any]] = {k: [] for k in (
        "H", "H_neg_ch", "P2", "L", "L_clean", "L_up",
        "c_low", "c_mid", "c_high", "C", "m", "masks", "m_L", "seed_i", "sample_id",
    )}
    with h5py.File(str(h5_path), "r") as f:
        n = len(f["sample_id"])
        c_low_keys = list(f["c_low"].keys())
        c_mid_keys = list(f["c_mid"].keys())
        c_high_keys = list(f["c_high"].keys())
        m = f["m"]
        m_keys = list(m.keys())
        m_l = f["m_L"]
        m_l_keys = list(m_l.keys())
        for i in range(n):
            fields["H"].append(f["H"][i])
            fields["H_neg_ch"].append(f["H_neg_ch"][i])
            fields["P2"].append(f["P2"][i])
            fields["L"].append(f["L"][i])
            fields["L_clean"].append(f["L_clean"][i])
            fields["L_up"].append(f["L_up"][i])
            fields["c_low"].append({k: float(f["c_low"][k][i]) for k in c_low_keys})
            fields["c_mid"].append({k: float(f["c_mid"][k][i]) for k in c_mid_keys})
            fields["c_high"].append({k: float(f["c_high"][k][i]) for k in c_high_keys})
            fields["C"].append(float(f["c/C"][i]))
            fields["m"].append({k: _decode_if_bytes(m[k][i]) for k in m_keys})
            fields["masks"].append({k: bool(f["masks"][k][i]) for k in f["masks"].keys()})
            fields["m_L"].append({k: _decode_if_bytes(m_l[k][i]) for k in m_l_keys})
            fields["seed_i"].append(int(f["seed_i"][i]))
            raw_id = f["sample_id"][i]
            fields["sample_id"].append(
                raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            )
    return fields


def _derive_exp_seeds(exp_index: str, master_seed: int, n: int) -> list[int]:
    """60 [S8] C4：EXP 重退化噪声种子（与全部现有划分不相交）。

    exp03/04 沿用 ``SeedSequence(master_seed).spawn(8)[5/6]``（工件已落盘，
    派生改动会破坏可复现性）；exp08 取分支 7 的深层子代
    ``spawn(8)[7].spawn(n)``——分支 7 未被任何划分占用，其子代与全部
    已用种子（原始分支 0–6 的值）不相交。
    """
    if exp_index == "exp08":
        return [
            int(c.generate_state(1, dtype=np.uint32)[0])
            for c in np.random.SeedSequence(master_seed).spawn(8)[7].spawn(n)
        ]
    seq = np.random.SeedSequence(master_seed).spawn(8)[EXP_SEED_BRANCH[exp_index]]
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in seq.spawn(n)]


def _redegrade_sample(args: tuple) -> dict:
    """生成单样本记录（并行 worker 的可 pickle 顶层函数，60 [S14] C4 无全局随机源）。

    args: (H, H_neg_ch, P2, L, L_clean, L_up, m_L, c, m, masks,
           seed_derived, seed_src, sigma_K, sigma_n, prior_level, redegrade)
    """
    (H, H_neg_ch, P2, L, L_clean, L_up, m_L, c, m, masks,
     seed_derived, seed_src, sigma_K, sigma_n, prior_level, redegrade) = args

    if redegrade:
        L, L_clean, _d, m_L = f_deg(
            H, sigma_K=sigma_K, sigma_n=sigma_n,
            r=DEFAULT_DOWNSAMPLE, seed=int(seed_derived),
        )
        L_up = normalize_intensity(upsample_4x_bilinear(L))
        seed_out = int(seed_derived)
    else:
        # EXP-07：不重退化，L/L_clean/L_up/m_L 逐位沿用源划分（80 [S6] 干净归因）；
        # 记录种子沿用源样本的 seed_i（与源 L 一一对应）。
        seed_out = int(seed_src)

    if prior_level == "P2":
        prior = P2
    else:  # P1：仅 c_low（40 [S6]），σ_smooth 与 P2 同因子（15×σ_smooth,H，
        # 保证 Gain(P2)−Gain(P1) 只归因于信息量而非平滑度）
        sigma_smooth_h = 0.125 * float(fine_structure_width(c) / DELTA_PX)
        prior, _meta = f_prior(
            c, level="P1", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
        )

    return {
        "_sample_id": "",
        "H": H,
        "H_neg_ch": H_neg_ch,
        "L": L,
        "L_clean": L_clean,
        "L_up": L_up,
        "P2": prior,
        "c": c,
        "m": m,
        "masks": masks,
        "seed_i": seed_out,
        "deg_level": "EXP",
        "m_L": m_L,
    }


def build_redegraded_exp(
    exp_index: str,
    sigma_K: float,
    sigma_n: float,
    data_version: str = "v1",
    master_seed: int = 20260825,
    workers: int = 16,
    prior_level: str = "P2",
    reps: int = 1,
    redegrade: bool = True,
) -> dict:
    """生成 test_exp0N.h5（H5Dataset 兼容 schema）；返回 manifest 节。"""
    import multiprocessing  # noqa: PLC0415

    if exp_index not in EXP_PRESETS:
        raise ValueError(f"exp_index 必须为 {sorted(EXP_PRESETS)}，实际 {exp_index!r}")
    if prior_level not in ("P1", "P2"):
        raise ValueError(f"prior_level 必须为 P1/P2，实际 {prior_level!r}")

    data_dir = PROJECT_ROOT / "data" / data_version
    fields_all: dict[str, list[Any]] = {}
    for split in SOURCE_SPLITS:
        src = data_dir / f"{split}.h5"
        if not src.exists():
            raise FileNotFoundError(f"源划分不存在：{src}")
        part = _load_source_fields(src)
        for k, v in part.items():
            fields_all.setdefault(k, []).extend(v)

    n_base = len(fields_all["H"])
    split_name = f"exp{exp_index[-2:]}"
    out_path = data_dir / f"test_{split_name}.h5"
    n_total = n_base * reps

    # 60 [S8] C4：噪声经新分支派生（与全部现有划分不相交）；EXP-07 无新噪声。
    sample_seeds = _derive_exp_seeds(exp_index, master_seed, n_total) if redegrade else [0] * n_total

    task_args = []
    k = 0
    for i in range(n_base):
        c_dict = {**fields_all["c_low"][i], **fields_all["c_mid"][i],
                  **fields_all["c_high"][i], "C": fields_all["C"][i]}
        for _r in range(reps):
            task_args.append(
                (
                    fields_all["H"][i],
                    fields_all["H_neg_ch"][i],
                    fields_all["P2"][i],
                    fields_all["L"][i],
                    fields_all["L_clean"][i],
                    fields_all["L_up"][i],
                    fields_all["m_L"][i],
                    c_dict,
                    fields_all["m"][i],
                    fields_all["masks"][i],
                    sample_seeds[k],
                    fields_all["seed_i"][i],
                    sigma_K,
                    sigma_n,
                    prior_level,
                    redegrade,
                )
            )
            k += 1

    if workers > 1 and n_total > 1:
        with multiprocessing.Pool(min(workers, 32)) as pool:
            records = pool.map(_redegrade_sample, task_args, chunksize=16)
    else:
        records = [_redegrade_sample(a) for a in task_args]

    # 样本 id：reps>1 时加 _r{k} 后缀（EXP-08 跨实现分组，80 [S6]）
    out_ids = []
    for i in range(n_base):
        base = fields_all["sample_id"][i]
        if reps > 1:
            out_ids.extend(f"{base}_r{r}" for r in range(reps))
        else:
            out_ids.append(base)
    for rec, sid in zip(records, out_ids):
        rec["_sample_id"] = sid
        rec["deg_level"] = split_name.upper()

    h5 = _create_split_file(
        out_path, split_name, n_total, data_version, master_seed, git_head()
    )
    h5.attrs["degradation"] = (
        f"sigma_K={sigma_K}, sigma_n={sigma_n} ({split_name.upper()}, "
        f"prior={prior_level}, reps={reps}, redegrade={redegrade})"
    )
    h5.attrs["source_splits"] = ",".join(SOURCE_SPLITS)
    h5.attrs["n_samples"] = n_total
    try:
        _write_batch(h5, records, 0)
    finally:
        h5.close()

    return {
        "split": split_name,
        "count": int(n_total),
        "n_base": int(n_base),
        "reps": int(reps),
        "sample_ids": out_ids,
        "sigma_K": float(sigma_K),
        "sigma_n": float(sigma_n),
        "prior_level": prior_level,
        "out_path": str(out_path),
        "code_version": git_head(),
        "spec_version": resolve_spec_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generators.build_exp34",
        description="EXP-03/04/07/08 测试集生成（80 [S6]）",
    )
    parser.add_argument("--exp", required=True, choices=["03", "04", "07", "08"], help="EXP 编号")
    parser.add_argument("--version", default="v1", help="数据版本（默认 v1）")
    parser.add_argument("--workers", type=int, default=16, help="并行 worker（默认 16）")
    parser.add_argument("--sigma-k", type=float, default=None, help="σ_K 覆盖")
    parser.add_argument("--sigma-n", type=float, default=None, help="σ_n 覆盖（exp04 必填）")
    args = parser.parse_args(argv)

    preset = EXP_PRESETS[f"exp{args.exp}"]
    sigma_k = args.sigma_k if args.sigma_k is not None else preset["sigma_k"]
    sigma_n = args.sigma_n if args.sigma_n is not None else preset["sigma_n"]
    if sigma_n is None:
        print(f"--exp {args.exp} 必须显式给定 --sigma-n（OQ-30-04 裁定后的 σ_n 取值）", file=sys.stderr)
        return 1

    manifest = build_redegraded_exp(
        f"exp{args.exp}", sigma_k, sigma_n,
        data_version=args.version, workers=args.workers,
        prior_level=preset["prior_level"], reps=preset["reps"], redegrade=preset["redegrade"],
    )
    print(f"[build_exp34] 生成 {manifest['count']} 样本（{manifest['n_base']} 基 × {manifest['reps']} 实现）"
          f" -> {manifest['out_path']}")
    print(f"  σ_K={manifest['sigma_K']}, σ_n={manifest['sigma_n']}, "
          f"prior={manifest['prior_level']}, code_version={manifest['code_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
