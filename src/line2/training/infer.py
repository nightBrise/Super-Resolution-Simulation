"""L3b [S5] 推理：ckpt + 输入图组 → 组N（npz：概率图 + 率图 + 元数据）。

载入端校验（`30` [S1] C3 后半）见 :func:`load_ckpt_payload`：ckpt 内 ``cfg_hash``
与由 ckpt 内 ``config`` 重算的哈希（``common.config_hash``，训练侧同一函数）必须一致，
并可另传期望值比对 config 哈希与 master_seed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import config_hash
from .train import build_model, kappa_version_of
from .unet import kappa_for, normalize_input_group

__all__ = ["infer", "load_ckpt_payload", "main"]


def _sha_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _sha256_16_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_ckpt_payload(
    ckpt: Path,
    expect_cfg_hash: str | None = None,
    expect_master_seed: int | None = None,
) -> dict:
    """ckpt 载入校验（`30` [S1] C3）并返回 payload。

    - 内建校验：ckpt 的 ``cfg_hash`` SHALL 等于由 ckpt 内 ``config`` 重算的
      :func:`~src.line2.training.common.config_hash`——两者不同源即视为篡改/错配，
      抛 ``ValueError``（config 与哈希任一侧被改动都会命中）；
    - 期望值比对（可选）：``expect_cfg_hash`` 对 config 哈希、``expect_master_seed``
      对 ckpt 的 ``master_seed``（实验矩阵侧据此钉死"跑的是哪份 config、哪个种子"），
      不一致抛 ``ValueError``。
    """
    ckpt = Path(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" not in payload:
        raise ValueError(f"{ckpt}：缺 state_dict，不是本线训练产出的 checkpoint")
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"{ckpt}：缺 config 段，无法校验 config 哈希（`30` [S1] C3）")
    actual = config_hash(cfg)
    stored = payload.get("cfg_hash")
    if stored is None:
        raise ValueError(
            f"{ckpt}：缺 cfg_hash 段（`30` [S3] 要求 checkpoint 含 config 哈希）"
        )
    if str(stored) != actual:
        raise ValueError(
            f"{ckpt}：config 哈希不符——ckpt 内 cfg_hash={stored} ≠ 重算 {actual}"
            "（config 或 cfg_hash 被改动，`30` [S1] C3）"
        )
    if expect_cfg_hash is not None and str(expect_cfg_hash) != actual:
        raise ValueError(
            f"{ckpt}：config 哈希与期望值不符——期望 {expect_cfg_hash} ≠ 实际 {actual}"
        )
    if expect_master_seed is not None:
        seed = payload.get("master_seed")
        if seed is None:
            raise ValueError(f"{ckpt}：缺 master_seed 段（`30` [S3]）")
        if int(seed) != int(expect_master_seed):
            raise ValueError(
                f"{ckpt}：master_seed 与期望值不符——期望 {int(expect_master_seed)} ≠ 实际 {int(seed)}"
            )
    return payload


def _forward(
    payload: dict, counts: np.ndarray, edges: np.ndarray, n_high: float
) -> tuple[np.ndarray, np.ndarray]:
    """已校验 payload 的确定性前向：返回 (概率图, 率图)。"""
    cfg = payload["config"]
    variant = str(cfg.get("variant", "joint"))
    net = build_model(cfg)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    counts_t = torch.from_numpy(counts.astype(np.float32)).unsqueeze(0)
    edges_t = torch.from_numpy(edges.astype(np.float32))
    shape_maps, _totals = normalize_input_group(counts_t)
    # κ（联合版）与独立版条件的 log₁₀N_low 一律从**输入计数和**派生，与训练侧
    # condition_vector(log_n_low=None) / _forward_variant 同口径：名义档位与实际计数和
    # 不符时（跨站套用、子样本缺粒）不会给网络喂一个训练期从未见过的条件值。
    # κ 口径由 ckpt 的 config 决定（`50` [S4] C1：``kappa`` = "v1"/"v2"）。
    kappa_version = kappa_version_of(cfg)
    with torch.no_grad():
        if variant == "joint":
            p = net(shape_maps, kappa_for(counts_t, edges_t, kappa_version))  # (1, 6, 128, 128)
        else:
            log_n = torch.log10(counts_t.sum(dim=(1, 2, 3)).clamp_min(1.0))
            p = net(shape_maps, edges_t, log_n)
    prob = p[0].numpy().astype(np.float64)          # (6, 128, 128)
    rate = (prob * float(n_high)).astype(np.float64)  # 逐格期望计数
    return prob, rate


def infer(
    ckpt: Path,
    counts: np.ndarray,
    edges: np.ndarray,
    n_high: float,
    expect_cfg_hash: str | None = None,
    expect_master_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """单图组推理（确定性，eval 模式）。返回 (概率图, 率图)。

    网络拓扑由 ckpt 的 ``config`` 经 ``train.build_model`` 重建：联合版与独立版
    共用本入口（`40` [S2] E3 对获胜变体 ckpt 直接推理，不重训练）。前向分支与
    ``train._forward_variant`` 同构。载入走 :func:`load_ckpt_payload`——config 哈希
    恒被校验，``expect_*`` 为可选的期望值比对入口（`30` [S1] C3）。
    """
    payload = load_ckpt_payload(ckpt, expect_cfg_hash, expect_master_seed)
    return _forward(payload, counts, edges, n_high)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1 推理（spec 30 [S5]）")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="数据集 HDF5")
    parser.add_argument("--beam-id", type=int, required=True)
    parser.add_argument("--station", type=int, default=0)
    parser.add_argument("--n-low", type=int, default=10_000)
    parser.add_argument("--sub", type=int, default=0)
    parser.add_argument("--n-high", type=float, default=200_000.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-cfg-hash", default=None,
                        help="期望的 config 哈希（`30` [S1] C3；不匹配则拒绝推理）")
    parser.add_argument("--expect-master-seed", type=int, default=None,
                        help="期望的 master_seed（`30` [S1] C3；不匹配则拒绝推理）")
    args = parser.parse_args(argv)

    payload = load_ckpt_payload(args.ckpt, args.expect_cfg_hash, args.expect_master_seed)
    with h5py.File(args.input, "r") as h5:
        stg = h5[f"beam_{args.beam_id:04d}/station{args.station}"]
        counts = stg[f"tier_{args.n_low:06d}_sub_{args.sub:02d}"][...]
        edges = stg[f"tier_{args.n_low:06d}_sub_{args.sub:02d}"].attrs["bin_edges"]

    prob, rate = _forward(payload, counts, edges, args.n_high)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "group_n.npz",
        prob=prob,
        rate=rate,
        beam_id=args.beam_id,
        station=args.station,
        n_low=args.n_low,
        sub_idx=args.sub,
        ckpt_md5=_sha_of(args.ckpt),
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "ckpt_md5": _sha_of(args.ckpt),
                # 两个同名前缀的哈希指代不同对象：文件字节 vs ckpt 内 config（防对账歧义）
                "ckpt_sha256_16": _sha256_16_of(args.ckpt),
                "cfg_hash": payload["cfg_hash"],
                "master_seed": payload.get("master_seed"),
                "beam_id": args.beam_id,
                "station": args.station,
                "n_low": args.n_low,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
