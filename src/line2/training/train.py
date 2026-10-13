"""L3b [S3] 训练循环（spec line2/30 [S3]）。

- 确定性（GC2 + M6 条款）：种子链派生一切；deterministic algorithms 开启；
  不可确定算子存在时记录执行模式并放宽为容差一致（warn_only + summary 记录）。
- 早停/选型准绳 = val N_eff 中位数（D6）；max_steps = 50000 预注册（M9/GC4）。
- checkpoint：best_val（val N_eff 最优）+ last；含 config 哈希与种子。
- test 束物理隔离：本模块只接收 train/val 路径（GC3 / [S3] C5）。
- 混合站台 batch：PSD 罚项的 Σ̂ 按**逐样本 bin 边界**重构（edges (B,4,129)）。
- 变体开关（`30` [S4]/F1）：``variant="joint"``（Phase 1 主模型，PSD 软一致罚）/
  ``variant="independent"``（H3 对照臂，逐投影独立 NLL、无跨投影罚）。
- Phase 1.5 通道（`50` [S4]）：``phase="1.5"`` 走 :func:`train_phase15`——κ v2
  无量纲、混合档（三档同训、站 0）、训满预算 + val_nll 单一选型准绳；κ 口径由
  config 的 ``kappa`` 字段承载（``kappa_version_of``），缺省保持 Phase 1 语义。
- D12（偏差登记，2026-09-14）：Phase 1 训练为单卡路径；DDP 双卡冒烟缓期
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

from src.line2.metrics import decisions
from src.line2.metrics.n_eff import n_eff

from .common import config_hash
from .data import Line2Pairs, MixedTierBatcher, collate_pairs, val_curve_for
from .losses import poisson_nll, poisson_terms, psd_penalty
from .unet import Phase1Net, condition_vector, kappa_for, normalize_input_group

__all__ = [
    "set_deterministic",
    "should_update_best",
    "build_model",
    "kappa_version_of",
    "train",
    "train_phase15",
    "main",
]


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


def should_update_best(val_metric: float, best: float, minimize: bool = False) -> bool:
    """best_val 选择规则（纯函数，M-2 可测）：严格更优才更新。

    最大化（默认，Phase 1 的 val N_eff）用严格 ``>``；最小化（Phase 1.5 的 val_nll，
    `50` [S4] C4）用严格 ``<``——持平不更新，故 best_step 恒为最优评估周期，
    不会停在并列的后一个。
    """
    return val_metric < best if minimize else val_metric > best


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
        kappa_version=kappa_version_of(cfg),
    )


def kappa_version_of(cfg: dict) -> str:
    """κ 口径（`50` [S4] C1）：config 的 ``kappa`` 字段；缺省 "v1"（Phase 1 语义）。"""
    return str(cfg.get("kappa", "v1"))


def _forward_variant(net, variant: str, batch: dict, device, kappa_version: str = "v1"):
    """变体前向（训练/评估共用）。

    ``kappa_version`` 只作用于联合臂的 κ（在调用点派生）；独立臂的 κ 由
    :class:`~src.line2.training.independent.IndependentEnsemble` 按自身构造时的
    ``kappa_version`` 从 edges 派生（两处同源于 config 的 ``kappa`` 字段）。
    """
    counts = batch["counts"].to(device)
    target = batch["target"].to(device)
    edges = batch["edges"].to(device)
    shapes, _totals = normalize_input_group(counts)
    if variant == "joint":
        kappa = kappa_for(counts, edges, kappa_version)
        prob = net(shapes, kappa)
    else:
        n_low_vec = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
        log_n = torch.log10(n_low_vec)
        prob = net(shapes, edges, log_n)
    return prob, target, edges


def train(cfg: dict) -> dict:
    """执行一次训练（确定性）。返回 summary dict。

    Phase 1.5 的混合档 config（``phase = "1.5"``）走 :func:`train_phase15`；
    无 ``phase`` 字段或缺省 "1" 时保持 Phase 1 单档语义逐位不变。
    """
    phase = str(cfg.get("phase", "1"))
    if phase == "1.5":
        return train_phase15(cfg)
    if phase != "1":
        raise ValueError(f"未知 phase={phase!r}（`1` = Phase 1 单档，`1.5` = 混合档）")
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
            counts = item["counts"].unsqueeze(0).to(device)  # (1, 6, 128, 128)
            edges = item["edges"].unsqueeze(0).to(device)     # (1, 4, 129)
            target = item["target"].unsqueeze(0).to(device)   # (1, 6, 128, 128)
            shapes, _totals = normalize_input_group(counts)
            n_low_vec = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
            log_n = torch.log10(n_low_vec)
            if variant == "joint":
                kappa = condition_vector(counts, edges)
                prob = net(shapes, kappa)
            else:
                prob = net(shapes, edges, log_n)
            n_effs.append(n_eff(prob[0].cpu().numpy(), target[0].cpu().numpy(), curve)[0])
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
            n_low_vec = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
            log_n = torch.log10(n_low_vec)
            if variant == "joint":
                kappa = condition_vector(counts, edges)
                prob = net(shapes, kappa)
                nll = poisson_nll(prob, target, n_high)
                pen = psd_penalty(prob, edges)
                loss = nll + float(cfg.get("w_psd", 0.1)) * pen
            else:
                prob = net(shapes, edges, log_n)
                nll = poisson_nll(prob, target, n_high)
                pen = torch.tensor(0.0)
                loss = nll
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


def _selection_reading(
    net,
    variant: str,
    kappa_version: str,
    val_sets: dict[int, Line2Pairs],
    n_high: float,
    device,
    eval_batch: int,
    val_beams: list[int],
    subs: tuple[int, ...],
) -> tuple[float, dict[int, float]]:
    """val 选型读数（`50` [S3] C2 的 val_nll）：逐 (束, 子样本) NLL 矩阵转发给 [S3] 聚合。

    本函数只负责前向与**排矩阵**——逐 (束, 子样本) 读数按 (val 束 × subs) 排成
    (束, 子样本) 矩阵（"先逐束收子样本"的 [S1] 聚合惯性由行分组承载）；聚合口径的
    唯一实现点是 `50` [S3] 的 :mod:`src.line2.metrics.decisions`
    （``tier_reading`` → ``selection_reading``：逐束中位数 → 全束中位数 → 三档等权平均），
    本函数不做任何自己的池化。

    函数名不含 ``nll``：`30` [S2] C4 的"唯一实现点"测试按函数名扫描 src/line2，
    Poisson 项的唯一实现点必须留在 ``losses.poisson_terms``。
    """
    net.eval()
    per_tier: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for tier, dset in val_sets.items():
            values: dict[tuple[int, int], float] = {}
            for start in range(0, len(dset), eval_batch):
                items = [dset[i] for i in range(start, min(start + eval_batch, len(dset)))]
                prob, target, _edges = _forward_variant(
                    net, variant, collate_pairs(items), device, kappa_version
                )
                sample_terms = poisson_terms(prob, target, n_high).mean(dim=(1, 2, 3))
                for item, value in zip(items, sample_terms.tolist(), strict=True):
                    values[(int(item["beam_id"]), int(item["sub_idx"]))] = float(value)
            missing = [
                (int(beam), int(sub))
                for beam in val_beams
                for sub in subs
                if (int(beam), int(sub)) not in values
            ]
            if missing:
                raise ValueError(
                    f"档 {tier} 的 val 读数缺 {len(missing)} 项（如 {missing[:3]}）——"
                    "选型读数要求 val 束 × subs 全覆盖"
                )
            per_tier[int(tier)] = np.asarray(
                [[values[(int(beam), int(sub))] for sub in subs] for beam in val_beams],
                dtype=np.float64,
            )
    net.train()
    return decisions.selection_reading(per_tier), {
        int(tier): decisions.tier_reading(per_tier, int(tier)) for tier in per_tier
    }


def train_phase15(cfg: dict) -> dict:
    """Phase 1.5 混合档训练（`50` [S4]）：三档同训、训满预算、val_nll 单一选型准绳。

    - 训练/验证集 = {3×10³, 1×10⁴, 1×10⁵} × train/val 束 × subs 0..k_train−1、**站 0**；
      站 1 粒子不进训练/验证（真迁移前提，config 在入口即校验）；
    - 每 step 按档均匀抽批（档位流与批序流独立 spawn，:class:`MixedTierBatcher`）；
    - 早停与选型 = val_nll（patience 用严格 ``<`` 更新 best）；无 N_eff 参与；
    - summary 落 stop_reason/best_step/best_val_nll/各档分项（`50` [S4] C4）+ 逐档抽批数
      （混合档机制的 run 级留痕）。
    """
    from .config_check import validate_run_config

    validate_run_config(cfg)
    seed = int(cfg["seed"])
    exec_mode = set_deterministic(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))

    device = torch.device(cfg.get("device", "cpu"))
    data_dir = Path(cfg["data_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    val_beams = [int(b) for b in manifest["splits"]["val"]]
    val_beams_limit = cfg.get("val_beams")
    if val_beams_limit is not None:  # 评估开销控制：三档同缩，落 config/summary 备核
        val_beams = val_beams[: int(val_beams_limit)]

    variant = str(cfg.get("variant", "joint"))
    kappa_version = kappa_version_of(cfg)
    tiers = tuple(int(t) for t in cfg["tiers"])
    stations = tuple(int(s) for s in cfg["stations"])
    k_train = int(cfg["k_train"])
    subs = tuple(range(k_train))
    n_high = float(cfg["n_high"])
    train_sets = {
        tier: Line2Pairs(
            data_dir / "train.h5", train_beams, n_low=tier, stations=stations, subs=subs
        )
        for tier in tiers
    }
    val_sets = {
        tier: Line2Pairs(
            data_dir / "val.h5", val_beams, n_low=tier, stations=stations, subs=subs
        )
        for tier in tiers
    }

    net = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(
        net.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=1e-4
    )
    batch_size = int(cfg.get("batch", 16))
    batcher = MixedTierBatcher(train_sets, batch_size=batch_size, seed=seed)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    run_cfg = dict(cfg)
    run_cfg["variant"] = variant
    run_cfg["params_total"] = int(total_params)
    run_cfg["params_trainable"] = int(trainable_params)
    run_cfg["n_train_beams"] = len(train_beams)
    run_cfg["n_val_beams"] = len(val_beams)
    cfg_hash = config_hash(run_cfg)

    max_steps = int(cfg["max_steps"])
    eval_step = int(cfg.get("eval_every", 1000))
    patience = int(cfg.get("patience", 10))
    eval_batch = int(cfg.get("eval_batch", 32))
    w_psd = float(cfg.get("w_psd", 0.1))

    best_val_nll = float("inf")
    best_step = -1
    best_per_tier: dict[str, float] = {}
    steps_since_improve = 0
    stop_reason = "budget_exhausted"
    tier_draws = {int(tier): 0 for tier in tiers}  # 混合档机制的 run 级留痕（逐档累计抽批数）
    history: list[dict] = []
    step = 0
    t0 = time.time()
    batches = iter(batcher)
    net.train()
    while step < max_steps:
        batch = next(batches)
        tier_draws[int(batch["tier"])] += 1
        prob, target, edges = _forward_variant(net, variant, batch, device, kappa_version)
        nll = poisson_nll(prob, target, n_high)
        if variant == "joint":
            pen = psd_penalty(prob, edges)
            loss = nll + w_psd * pen
        else:
            pen = torch.tensor(0.0)
            loss = nll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()
        step += 1
        if step % eval_step and step != max_steps:
            continue

        val_nll, per_tier = _selection_reading(
            net, variant, kappa_version, val_sets, n_high, device, eval_batch,
            val_beams, subs,
        )
        entry = {
            "step": step,
            "train_nll": float(nll),
            "psd": float(pen),
            "val_nll": val_nll,
            "val_nll_per_tier": {str(tier): value for tier, value in per_tier.items()},
            "tier_draws": {str(tier): count for tier, count in tier_draws.items()},
            "lr": float(opt.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(entry)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if should_update_best(val_nll, best_val_nll, minimize=True):
            best_val_nll, best_step = val_nll, step
            best_per_tier = {str(tier): value for tier, value in per_tier.items()}
            steps_since_improve = 0
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "config": run_cfg,
                    "cfg_hash": cfg_hash,
                    "master_seed": seed,
                    "best_val_nll": val_nll,
                    "step": step,
                },
                out_dir / "best_val.ckpt",
            )
        else:
            steps_since_improve += 1
            if steps_since_improve >= patience:
                stop_reason = "val_nll_patience"
                break

    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": run_cfg,
            "cfg_hash": cfg_hash,
            "master_seed": seed,
            "step": step,
        },
        out_dir / "last.ckpt",
    )
    summary = {
        "phase": "1.5",
        "variant": variant,
        "seed": seed,
        "steps": step,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_val_nll": best_val_nll,
        "best_val_nll_per_tier": best_per_tier,
        "val_nll_history": history,
        "tier_draws": {str(tier): count for tier, count in tier_draws.items()},
        "tiers": list(tiers),
        "stations": list(stations),
        "k_train": k_train,
        "kappa": kappa_version,
        "n_train_beams": len(train_beams),
        "n_val_beams": len(val_beams),
        "val_beams_limit": val_beams_limit,
        "batch": batch_size,
        "max_steps": max_steps,
        "eval_every": eval_step,
        "patience": patience,
        "cfg_hash": cfg_hash,
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "exec_mode": exec_mode,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 训练（spec 30 [S3]；Phase 1.5 起 phase=1.5 + κ v2 + val_nll 选型，spec 50 [S4]）")
    parser.add_argument("--config", type=Path, required=True, help="训练 config JSON")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    summary = train(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
