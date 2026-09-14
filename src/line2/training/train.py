"""L3b [S3] 训练循环（spec line2/30 [S3]）。

- 确定性（GC2 + M6 条款）：种子链派生一切；deterministic algorithms 开启；
  不可确定算子存在时记录执行模式并放宽为容差一致（warn_only + summary 记录）。
- 早停/选型准绳 = val N_eff 中位数（D6）；max_steps = 50000 预注册（M9/GC4）。
- checkpoint：best_val（val N_eff 最优）+ last；含 config 哈希与种子。
- test 束物理隔离：本模块只接收 train/val 路径（GC3 / [S3] C5）。
- 混合站台 batch：PSD 罚项的 Σ̂ 按**逐样本 bin 边界**重构（edges (B,4,129)）。
- 变体开关（`30` [S4]/F1）：``variant="joint"``（Phase 1 主模型，PSD 软一致罚）/
  ``variant="independent"``（H3 对照臂，逐投影独立 NLL、无跨投影罚）。
- D16（偏差登记，2026-09-14）：Phase 1 训练为单卡路径；DDP 双卡冒烟缓期
  （E1/E2 共 10–15 runs 单卡可行，排期已计入）——`30` [S3] C4 的双卡冒烟按
  偏差处理，验收闸口裁决。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.line2.metrics.n_eff import n_eff

from .common import config_hash
from .data import Line2Pairs, val_curve_for
from .losses import poisson_nll, psd_penalty
from .unet import Phase1Net, condition_vector, normalize_input_group

__all__ = ["set_deterministic", "should_update_best", "build_model", "train", "main"]


def set_deterministic(seed: int) -> dict:
    """GC2 + M6：种子派生与确定性算法。返回执行模式记录（含 warn_only 状态）。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "deterministic_algorithms": True,
        "warn_only": True,
        "cudnn_deterministic": True,
    }


def should_update_best(val_metric: float, best: float) -> bool:
    """best_val 选择规则（纯函数，M-2 可测）：严格更优才更新。"""
    return val_metric > best


def build_model(cfg: dict):
    """按 variant 构建模型（joint → Phase1Net / independent → IndependentEnsemble）。"""
    if cfg.get("variant", "joint") == "joint":
        return Phase1Net(
            base=int(cfg.get("base", 48)),
            num_levels=int(cfg.get("num_levels", 4)),
            blocks=int(cfg.get("blocks", 2)),
            cond_dim=11,
        )
    from .independent import IndependentEnsemble

    return IndependentEnsemble(
        base=int(cfg.get("ind_base", 24)),
        num_levels=int(cfg.get("num_levels", 4)),
        blocks=int(cfg.get("ind_blocks", 3)),
    )


def _forward_variant(net, variant: str, batch: dict, device):
    counts = batch["counts"].to(device)
    target = batch["target"].to(device)
    edges = batch["edges"].to(device)
    shapes, _totals = normalize_input_group(counts)
    if variant == "joint":
        kappa = condition_vector(counts, edges)
        prob = net(shapes, kappa)
    else:
        n_low_vec = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
        log_n = torch.log10(n_low_vec)
        prob = net(shapes, edges, log_n)
    return prob, target, edges


def train(cfg: dict) -> dict:
    """执行一次训练（确定性）。返回 summary dict。"""
    seed = int(cfg["seed"])
    exec_mode = set_deterministic(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))

    device = torch.device(cfg.get("device", "cpu"))
    data_dir = Path(cfg["data_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    val_beams = [int(b) for b in manifest["splits"]["val"]]
    variant = str(cfg.get("variant", "joint"))

    train_set = Line2Pairs(data_dir / "train.h5", train_beams, n_low=int(cfg["n_low"]))
    val_set = Line2Pairs(
        data_dir / "val.h5", val_beams[:4], n_low=int(cfg["n_low"]), subs=(0, 1, 2, 3)
    )
    curve = val_curve_for(data_dir)
    n_high = float(cfg["n_high"])

    net = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set,
        batch_size=int(cfg.get("batch", 16)),
        shuffle=True,
        num_workers=0,
        generator=gen,
        drop_last=True,
    )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    run_cfg = dict(cfg)
    run_cfg["variant"] = variant
    run_cfg["params_total"] = int(total_params)
    run_cfg["params_trainable"] = int(trainable_params)
    cfg_hash = config_hash(run_cfg)

    best_step = -1
    best_val_n_eff = float("-inf")
    step = 0
    eval_step = int(cfg.get("eval_every", 1000))
    patience = int(cfg.get("patience", 10))
    evals_since_improve = 0
    history: list[float] = []
    t0 = time.time()

    @torch.no_grad()
    def _eval() -> tuple[float, float]:
        """val 固定子集：N_eff 中位数（选型准绳）与平均 NLL（日志字段）。"""
        net.eval()
        n_effs: list[float] = []
        nlls: list[float] = []
        for i in range(len(val_set)):
            item = val_set[i]
            counts = item["counts"].unsqueeze(0)
            edges = item["edges"].unsqueeze(0)
            target = item["target"].unsqueeze(0)
            shapes, _totals = normalize_input_group(counts)
            kappa = condition_vector(counts, edges)
            prob = net(shapes, kappa)
            n_effs.append(n_eff(prob[0].numpy(), target.numpy()[0], curve)[0])
            nlls.append(float(poisson_nll(prob, target, n_high)))
        net.train()
        return float(np.median(n_effs)), float(np.mean(nlls))

    net.train()
    stop = False
    while step < int(cfg["max_steps"]) and not stop:
        for batch in loader:
            if step >= int(cfg["max_steps"]):
                break
            counts = batch["counts"].to(device)
            target = batch["target"].to(device)
            edges = batch["edges"].to(device)
            shapes, _totals = normalize_input_group(counts)
            kappa = condition_vector(counts, edges)
            prob = net(shapes, kappa)
            nll = poisson_nll(prob, target, n_high)
            pen = psd_penalty(prob, edges)
            loss = nll + float(cfg.get("w_psd", 0.1)) * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            opt.step()
            step += 1
            if step % eval_step == 0 or step == int(cfg["max_steps"]):
                val_metric, val_nll = _eval()
                history.append(val_metric)
                entry = {
                    "step": step,
                    "train_nll": float(nll),
                    "psd": float(pen),
                    "val_nll": val_nll,
                    "val_n_eff": val_metric,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad_norm": grad_norm,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if should_update_best(val_metric, best_val_n_eff):
                    best_step, best_val_n_eff = step, val_metric
                    evals_since_improve = 0
                    torch.save(
                        {
                            "state_dict": net.state_dict(),
                            "config": run_cfg,
                            "cfg_hash": config_hash(run_cfg),
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
            "config": run_cfg,
            "cfg_hash": config_hash(run_cfg),
            "master_seed": seed,
            "step": step,
        },
        out_dir / "last.ckpt",
    )
    summary = {
        "variant": variant,
        "steps": step,
        "best_val_n_eff": best_val_n_eff,
        "best_step": best_step,
        "val_n_eff_history": history,
        "cfg_hash": config_hash(run_cfg),
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "seed": seed,
        "exec_mode": exec_mode,
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
