"""EXP-03/04 重退化测试集生成（80 [S6]，60 [S15] C4b 交叉生成工具归 src/generators/）。

对 ``test_id ∪ test_pb`` 的**同一 H** 以更强退化重退化（80 [S6]：EXP-03 强模糊
σ_K=2×D2 标定值；EXP-04 高噪声 σ_n 按 OQ-30-04 裁定取值），写出与主划分
同 schema 的 ``test_exp0N.h5``（顶层数组 + c/m/masks/m_L 组，60 [S14] 契约，
H5Dataset 可直接读取评估）。

噪声实现经**新 SeedSequence 分支**（EXP-03→分支 5、EXP-04→分支 6，均未被
SPLIT_BRANCH 0–4 占用）派生——60 [S8] C4：同一 c 的噪声实现不得跨划分复用，
EXP-03/04 的噪声 SHALL 与 test_id/test_pb 及彼此不相交。

用法：:

    python -m src.generators.build_exp34 --exp 03 [--version v1] [--workers N]
    python -m src.generators.build_exp34 --exp 04 --sigma-n <裁定值> [--sigma-k 11.0]

EXP-03 预设 σ_K=22.0、σ_n=1.22e-4（30 [S8]）；EXP-04 的 σ_n 必须显式给定
（OQ-30-04 二级咨询裁定后执行）。
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
    git_head,
    resolve_spec_version,
)
from src.generators.f_deg import f_deg
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 重退化噪声的 SeedSequence 分支（60 [S8] C4 不相交；SPLIT_BRANCH 已占用 0–4）。
EXP_SEED_BRANCH = {"exp03": 5, "exp04": 6}

#: 30 [S8] EXP-03 预设（σ_K = 2×D2 标定值 11.0、σ_n = D2 采用值）。
EXP03_SIGMA_K = 22.0
D2_SIGMA_N = 1.22e-4

#: 源划分（80 [S6]：同一 H、同一划分，逐样本配对）。
SOURCE_SPLITS = ("test_id", "test_pb")


def _load_source_fields(h5_path: Path) -> dict[str, list[Any]]:
    """读源划分的全部字段（按样本切分的列表，供记录组装）。"""
    fields: dict[str, list[Any]] = {k: [] for k in (
        "H", "H_neg_ch", "P2", "c_low", "c_mid", "c_high", "C", "m", "masks", "sample_id",
    )}
    with h5py.File(str(h5_path), "r") as f:
        n = len(f["sample_id"])
        c_low_keys = list(f["c_low"].keys())
        c_mid_keys = list(f["c_mid"].keys())
        c_high_keys = list(f["c_high"].keys())
        m = f["m"]
        m_keys = list(m.keys())
        for i in range(n):
            fields["H"].append(f["H"][i])
            fields["H_neg_ch"].append(f["H_neg_ch"][i])
            fields["P2"].append(f["P2"][i])
            fields["c_low"].append({k: float(f["c_low"][k][i]) for k in c_low_keys})
            fields["c_mid"].append({k: float(f["c_mid"][k][i]) for k in c_mid_keys})
            fields["c_high"].append({k: float(f["c_high"][k][i]) for k in c_high_keys})
            fields["C"].append(float(f["c/C"][i]))
            fields["m"].append({k: _decode_if_bytes(m[k][i]) for k in m_keys})
            fields["masks"].append({k: bool(f["masks"][k][i]) for k in f["masks"].keys()})
            raw_id = f["sample_id"][i]
            fields["sample_id"].append(
                raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            )
    return fields


def _decode_if_bytes(value: Any) -> Any:
    """h5py 字符串数据集逐元素读回 bytes → str（60 [S14] 契约，utf-8）。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _redegrade_sample(args: tuple) -> dict:
    """重退化单样本（并行 worker 的可 pickle 顶层函数，60 [S14] C4 无全局随机源）。

    args: (H, H_neg_ch, P2, c, m, masks, sample_id, seed_i, sigma_K, sigma_n)
    """
    H, H_neg_ch, P2, c, m, masks, sample_id, seed_i, sigma_K, sigma_n = args
    L, L_clean, _d, m_L = f_deg(
        H, sigma_K=sigma_K, sigma_n=sigma_n,
        r=DEFAULT_DOWNSAMPLE, seed=int(seed_i),
    )
    L_up = normalize_intensity(upsample_4x_bilinear(L))
    return {
        "_sample_id": sample_id,
        "H": H,
        "H_neg_ch": H_neg_ch,
        "L": L,
        "L_clean": L_clean,
        "L_up": L_up,
        "P2": P2,
        "c": c,
        "m": m,
        "masks": masks,
        "seed_i": int(seed_i),
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
) -> dict:
    """对 test_id ∪ test_pb 的 H 重退化并写出 test_exp0N.h5（H5Dataset 兼容 schema）。

    返回 manifest 节（样本数、sample_id 清单、退化配置）。
    """
    import multiprocessing  # noqa: PLC0415

    if exp_index not in EXP_SEED_BRANCH:
        raise ValueError(f"exp_index 必须为 exp03/exp04，实际 {exp_index!r}")

    data_dir = PROJECT_ROOT / "data" / data_version
    fields_all: dict[str, list[Any]] = {}
    for split in SOURCE_SPLITS:
        src = data_dir / f"{split}.h5"
        if not src.exists():
            raise FileNotFoundError(f"源划分不存在：{src}")
        part = _load_source_fields(src)
        for k, v in part.items():
            fields_all.setdefault(k, []).extend(v)

    n = len(fields_all["H"])
    split_name = f"exp{exp_index[-2:]}"
    out_path = data_dir / f"test_{split_name}.h5"

    # 60 [S8] C4：重退化噪声经新分支派生（与全部现有划分不相交）。
    seed_seq = np.random.SeedSequence(master_seed).spawn(8)[EXP_SEED_BRANCH[exp_index]]
    sample_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in seed_seq.spawn(n)]

    task_args = [
        (
            fields_all["H"][i],
            fields_all["H_neg_ch"][i],
            fields_all["P2"][i],
            {**fields_all["c_low"][i], **fields_all["c_mid"][i],
             **fields_all["c_high"][i], "C": fields_all["C"][i]},
            fields_all["m"][i],
            fields_all["masks"][i],
            fields_all["sample_id"][i],
            sample_seeds[i],
            sigma_K,
            sigma_n,
        )
        for i in range(n)
    ]
    if workers > 1 and n > 1:
        with multiprocessing.Pool(min(workers, 32)) as pool:
            records = pool.map(_redegrade_sample, task_args, chunksize=16)
    else:
        records = [_redegrade_sample(a) for a in task_args]
    for rec in records:
        rec["deg_level"] = split_name.upper()

    h5 = _create_split_file(
        out_path, split_name, n, data_version, master_seed, git_head()
    )
    h5.attrs["degradation"] = f"sigma_K={sigma_K}, sigma_n={sigma_n} ({split_name.upper()})"
    h5.attrs["source_splits"] = ",".join(SOURCE_SPLITS)
    h5.attrs["n_samples"] = n
    try:
        _write_batch(h5, records, 0)
    finally:
        h5.close()

    return {
        "split": split_name,
        "count": int(n),
        "sample_ids": [r["_sample_id"] for r in records],
        "sigma_K": float(sigma_K),
        "sigma_n": float(sigma_n),
        "out_path": str(out_path),
        "code_version": git_head(),
        "spec_version": resolve_spec_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generators.build_exp34",
        description="EXP-03/04 重退化测试集生成（80 [S6]）",
    )
    parser.add_argument("--exp", required=True, choices=["03", "04"], help="EXP 编号")
    parser.add_argument("--version", default="v1", help="数据版本（默认 v1）")
    parser.add_argument("--workers", type=int, default=16, help="并行 worker（默认 16）")
    parser.add_argument("--sigma-k", type=float, default=None, help="σ_K 覆盖")
    parser.add_argument("--sigma-n", type=float, default=None, help="σ_n 覆盖（exp04 必填）")
    args = parser.parse_args(argv)

    if args.exp == "03":
        sigma_k = args.sigma_k if args.sigma_k is not None else EXP03_SIGMA_K
        sigma_n = args.sigma_n if args.sigma_n is not None else D2_SIGMA_N
    else:  # exp04
        if args.sigma_n is None:
            print("--exp 04 必须显式给定 --sigma-n（OQ-30-04 裁定后的 σ_n 取值）", file=sys.stderr)
            return 1
        sigma_k = args.sigma_k if args.sigma_k is not None else 11.0
        sigma_n = args.sigma_n

    manifest = build_redegraded_exp(
        f"exp{args.exp}", sigma_k, sigma_n,
        data_version=args.version, workers=args.workers,
    )
    print(f"[build_exp34] 生成 {manifest['count']} 样本 -> {manifest['out_path']}")
    print(f"  σ_K={manifest['sigma_K']}, σ_n={manifest['sigma_n']}, "
          f"code_version={manifest['code_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
