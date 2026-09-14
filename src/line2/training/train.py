"""L3b [S3] 训练循环（spec line2/30 [S3]）。

- 确定性（GC2 + M6 条款）：种子链派生一切；deterministic algorithms 开启；
  不可确定算子存在时记录执行模式并放宽为容差一致。
- 早停/选型准绳 = val N_eff（D6）；max_steps = 50000 预注册（M9/GC4）。
- checkpoint：best_val（val N_eff 最优）+ last；含 config 哈希与种子。
- test 束物理隔离：本模块只接收 train/val 路径（GC3 / [S3] C5）。
- 混合站台 batch：PSD 罚项的 Σ̂ 按**逐样本 bin 边界**重构（edges (B,4,129)）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.line2.metrics.n_eff import n_eff

from .data import Line2Pairs, val_curve_for
from .losses import total_loss
from .unet import Phase1Net, condition_vector, normalize_input_group

__all__ = ["set_deterministic", "train", "main"]


def set_deterministic(seed: int) -> None:
    """GC2 + M6：种子派生与确定性算法（不可确定算子 → warn 并记录模式）。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


@torch.no_grad()
def _val_n_eff_median(
    net: Phase1Net,
    val_pairs: Line2Pairs,
    curve,
    n_high: float,
) -> float:
    """val 固定子集上逐样本 N_eff 的中位数（早停/选型准绳，D6）。"""
    net.eval()
    values: list[float] = []
    for i in range(len(val_pairs)):
        item = val_pairs[i]
        counts = item["counts"].unsqueeze(0)
        edges = item["edges"].unsqueeze(0)
        target = item["target"]  # (6, 128, 128)——n_eff 逐投影索引
        shapes, _totals = normalize_input_group(counts)
        kappa = condition_vector(counts, edges)
        prob = net(shapes, kappa)[0]
        value, _flag = n_eff(prob.numpy(), target.numpy(), curve)
        values.append(value)
    return float(np.median(values))


def train(cfg: dict) -> dict:
    """执行一次训练（确定性）。返回 summary dict。"""
    seed = int(cfg["seed"])
    set_deterministic(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))

    device = torch.device(cfg.get("device", "cpu"))
    data_dir = Path(cfg["data_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    val_beams = [int(b) for b in manifest["splits"]["val"]]

    train_set = Line2Pairs(data_dir / "train.h5", train_beams, n_low=int(cfg["n_low"]))
    # val N_eff 固定子集：前 4 个 val 束 × 两站 × 4 子样本 = 32 样本
    val_set = Line2Pairs(
        data_dir / "val.h5", val_beams[:4], n_low=int(cfg["n_low"]), subs=(0, 1, 2, 3)
    )
    curve = val_curve_for(data_dir)
    n_high = float(cfg["n_high"])

    net = Phase1Net(
        base=int(cfg.get("base", 48)),
        num_levels=int(cfg.get("num_levels", 4)),
        blocks=int(cfg.get("blocks", 2)),
        cond_dim=11,
    ).to(device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=1e-4
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set, batch_size=int(cfg.get("batch", 16)), shuffle=True,
        num_workers=0, generator=gen, drop_last=True,
    )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best = {"val_n_eff": float("-inf"), "step": -1}
    step = 0
    eval_step = int(cfg.get("eval_every", 1000))
    patience = int(cfg.get("patience", 10))
    evals_since_improve = 0
    history: list[float] = []
    t0 = time.time()

    def _eval() -> float:
        return _val_n_eff_median(net, val_set, curve, n_high)

    net.train()
    stop = False
    while step < int(cfg["max_steps"]) and not stop:
        for batch in loader:
            if step >= int(cfg["max_steps"]):
                break
            counts = batch["counts"].to(device)
            target = batch["target"].to(device)
            edges = batch["edges"].to(device)  # (B, 4, 129) 逐样本
            shapes, _totals = normalize_input_group(counts)
            kappa = condition_vector(counts, edges)
            prob = net(shapes, kappa)
            loss, parts = total_loss(
                prob, target, edges, n_high, w_psd=float(cfg.get("w_psd", 0.1))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % eval_step == 0 or step == int(cfg["max_steps"]):
                val_metric = _eval()
                history.append(val_metric)
                entry = {
                    "step": step,
                    "nll": float(parts["nll"]),
                    "val_n_eff": val_metric,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if val_metric > best["val_n_eff"]:
                    best = {"val_n_eff": val_metric, "step": step}
                    evals_since_improve = 0
                    torch.save(
                        {
                            "state_dict": net.state_dict(),
                            "config": cfg,
                            "cfg_hash": _config_hash(cfg),
                            "master_seed": seed,
                            "best_val_n_eff": val_metric,
                            "step": step,
                        },
                        out_dir / "best_val.ckpt",
                    )
                else:
                    evals_since_improve += 1
                    if evals_since_improve >= patience:
                        stop = True
                        break
        if stop or step >= int(cfg["max_steps"]):
            break

    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": cfg,
            "cfg_hash": _config_hash(cfg),
            "master_seed": seed,
            "step": step,
        },
        out_dir / "last.ckpt",
    )
    summary = {
        "steps": step,
        "best_val_n_eff": best["val_n_eff"],
        "best_step": best["step"],
        "val_n_eff_history": history,
        "cfg_hash": _config_hash(cfg),
        "seed": seed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1 训练（spec 30 [S3]）")
    parser.add_argument("--config", type=Path, required=True, help="训练 config JSON")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    summary = train(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
