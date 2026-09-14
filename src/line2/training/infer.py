"""L3b [S5] 推理：ckpt + 输入图组 → 组N（npz：概率图 + 率图 + 元数据）。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .train import build_model
from .unet import condition_vector, normalize_input_group


def _sha_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def infer(
    ckpt: Path,
    counts: np.ndarray,
    edges: np.ndarray,
    n_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """单图组推理（确定性，eval 模式）。返回 (概率图, 率图)。

    网络拓扑由 ckpt 的 ``config`` 经 ``train.build_model`` 重建：联合版与独立版
    共用本入口（`40` [S2] E3 对获胜变体 ckpt 直接推理，不重训练）。前向分支与
    ``train._forward_variant`` 同构。
    """
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload.get("config", {})
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
    with torch.no_grad():
        if variant == "joint":
            kappa = condition_vector(counts_t, edges_t)
            p = net(shape_maps, kappa)  # (1, 6, 128, 128)
        else:
            log_n = torch.log10(counts_t.sum(dim=(1, 2, 3)).clamp_min(1.0))
            p = net(shape_maps, edges_t, log_n)
    prob = p[0].numpy().astype(np.float64)          # (6, 128, 128)
    rate = (prob * float(n_high)).astype(np.float64)  # 逐格期望计数
    return prob, rate


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
    args = parser.parse_args(argv)

    cfg_hash = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()[:16]
    with h5py.File(args.input, "r") as h5:
        stg = h5[f"beam_{args.beam_id:04d}/station{args.station}"]
        counts = stg[f"tier_{args.n_low:06d}_sub_{args.sub:02d}"][...]
        edges = stg[f"tier_{args.n_low:06d}_sub_{args.sub:02d}"].attrs["bin_edges"]

    prob, rate = infer(args.ckpt, counts, edges, args.n_high)
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
                "cfg_hash": cfg_hash,
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
