"""训练 CLI：``python -m src.training.train --config <path> [--smoke] [--steps N]``。

接口按 60 [S15] 15.7：统一由 ``--config`` 指定方案；输出目录默认
``results/<EXP_id>_<arm>_<seed>_<run_tag>/``（可用 ``--out`` 覆盖）；
启动前先跑 ``scripts/check_env.py``（仅警告，不阻塞）。

训练协议（60）：
- 优化器 AdamW（[S9]：lr / weight decay / beta1 / beta2 从 config 读，
  三方案一致）；损失 HybridLoss（[S2]，λ 冻结 1.0）；
- 数据从 HDF5 读取（[S4][S14]），训练输入为带噪 ``L_up``（已含噪，
  [S6][S13] AC6）；方案 C 参数标准化统计量只从训练集计算（[S5][S13]
  AC5，★ 无测试集泄漏）；
- 早停（[S10]）：每 2,000 步验证一次验证集总损失，保存最优 checkpoint，
  patience=10 无改善停止、不早于最大步数预算的 50%；
- 日志（[S12] C1）与 checkpoint（[S12] C2：best_val + last + 配置 +
  种子 + 数据版本 + 训练曲线）；seeds.json（[S14] C8）；
- DDP（[S15] 15.7）：nccl、DistributedSampler 同数据分片、每卡同 seed、
  rank 0 存 checkpoint；单卡 fallback 使用同一 shuffle seed。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from src.models.schemes import build_scheme_model, forward_scheme
from src.training.loss import HybridLoss
from src.utils.checkpoint import save_checkpoint, write_seeds_json
from src.utils.config_utils import (
    config_digest,
    load_config,
    resolve_data_version,
    resolve_device,
    resolve_precision,
    run_output_dir,
)
from src.utils.h5data import H5Dataset, compute_c_prior_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 默认验证间隔（步，60 [S10] C2）。
DEFAULT_VAL_INTERVAL = 2000
#: 默认日志间隔（步，60 [S12] "每若干 step 记录"）。
DEFAULT_LOG_INTERVAL = 50


# ---------------------------------------------------------------------------
# 早停（60 [S10]）
# ---------------------------------------------------------------------------
class EarlyStopping:
    """基于验证集总损失的早停。

    每 ``val_interval`` 步验证一次；连续 ``patience`` 次无改善且已过
    最大步数预算的 ``min_step_fraction``（默认 50%）时停止（60 [S10]
    C2/C3/C4）。最优验证损失随验证推进更新，不因早停窗口而冻结。
    """

    def __init__(
        self,
        patience: int = 10,
        min_step_fraction: float = 0.5,
        max_steps: int = 100_000,
        val_interval: int = DEFAULT_VAL_INTERVAL,
        improvement_eps: float = 0.0,
    ) -> None:
        self.patience = int(patience)
        self.min_steps = int(max_steps * float(min_step_fraction))
        self.val_interval = int(val_interval)
        self.improvement_eps = float(improvement_eps)
        self.best_val_loss: float | None = None
        self._no_improve = 0
        self._last_val_step: int | None = None

    def should_validate(self, step: int) -> bool:
        """每 ``val_interval`` 步验证一次（60 [S10] C2），step 0 不验证。"""
        return step > 0 and step % self.val_interval == 0

    def on_validation(self, step: int, val_loss: float) -> bool:
        """登记一次验证结果，返回是否应停止训练。

        连续 ``patience`` 次验证无改善且 ``step ≥ min_steps`` 时返回
        ``True``；早停不得早于最大步数预算的 50%（60 [S10] C4）。
        """
        if self.best_val_loss is None or val_loss < self.best_val_loss - self.improvement_eps:
            self.best_val_loss = val_loss
            self._no_improve = 0
        else:
            self._no_improve += 1
        self._last_val_step = step
        return self._no_improve >= self.patience and step >= self.min_steps

    @property
    def no_improve_count(self) -> int:
        """连续无改善的验证次数。"""
        return self._no_improve


# ---------------------------------------------------------------------------
# 日志（60 [S12] C1）
# ---------------------------------------------------------------------------
def format_log_record(record: dict[str, Any]) -> str:
    """日志行格式化：``key=value`` 空格分隔，浮点用 6 位有效数字。"""
    parts = []
    for key, value in record.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class TrainingLogger:
    """训练日志写入器：追加写 ``logs/train.log``（60 [S12] C1 字段）。"""

    def __init__(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        self.path = out_dir / "logs" / "train.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(format_log_record(record) + "\n")


# ---------------------------------------------------------------------------
# 数据 / 分布式
# ---------------------------------------------------------------------------
def _shuffle_seed(config: dict, data_version: str) -> int:
    """单卡 fallback 与 DDP 共用的 shuffle seed（60 [S15] 15.7）。

    由 master_seed + seed_index + 数据版本（确定性摘要）派生，三方案同
    seed_index 下数据顺序一致（60 [S11] C1 公平性），跨进程可复现。
    """
    import hashlib

    version_digest = int(hashlib.md5(data_version.encode("utf-8")).hexdigest(), 16) % (2**32)
    ss = np.random.SeedSequence([int(config["master_seed"]), int(config.get("seed_index", 0)), version_digest])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _init_distributed() -> dict | None:
    """按环境变量初始化 DDP（nccl）；无 ``LOCAL_RANK`` 时返回 None。"""
    if not torch.distributed.is_available() or "LOCAL_RANK" not in os.environ:
        return None
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world)
    return {"rank": rank, "world_size": world}


def _make_loader(dataset, config: dict, shuffle_seed: int, ddp: dict | None, drop_last: bool = True):
    """DataLoader：DDP 用 DistributedSampler（同 seed、不同分片），
    单卡 fallback 用同一 shuffle seed 的确定性打乱（60 [S15] 15.7）。"""
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config.get("training", {}).get("data_loading", {}).get("num_workers", 0))
    pin_memory = bool(config.get("training", {}).get("data_loading", {}).get("pin_memory", True))
    if ddp is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp["world_size"],
            rank=ddp["rank"],
            shuffle=True,
            seed=shuffle_seed,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    generator = torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# 训练步骤
# ---------------------------------------------------------------------------
def _validate(model, loader, loss_fn, device: str) -> float:
    """验证集总损失均值（60 [S10] C2：早停与最优 checkpoint 依据）。"""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            H_hat = forward_scheme(model, batch, device)
            target = batch["H"].to(device) * model.work_scale
            loss = loss_fn(H_hat, target)
            total += float(loss.sum().item())
            count += H_hat.shape[0]
    return total / count if count else float("nan")


def _save_checkpoint(
    path: Path,
    model,
    config: dict,
    step: int,
    val_loss: float | None,
    curve: list[float],
    c_prior_stats: tuple[np.ndarray, np.ndarray] | None,
    meta: dict,
) -> None:
    """按 60 [S12] C2 保存 checkpoint（权重 + 配置 + 种子 + 数据版本 + 曲线）。"""
    state: dict[str, Any] = {
        "model_class": type(model).__name__,
        "network_config": model.network_config,
        "model_state": model.state_dict(),
        "config": config,
        "config_hash": meta["config_hash"],
        "data_version": meta["data_version"],
        "spec_version": meta["spec_version"],
        "master_seed": int(config["master_seed"]),
        "scheme_seed": int(config.get("scheme_seeds", {}).get(f"scheme_{config['scheme']}_seed_{config.get('seed_index', 0)}", 0)),
        "step": int(step),
        "val_loss": val_loss,
        "train_curve": curve,
    }
    if c_prior_stats is not None:
        state["c_prior_mu"], state["c_prior_sigma"] = c_prior_stats
    save_checkpoint(path, state)


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def train(
    config: dict,
    out_dir: str | Path,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """执行一次方案训练，返回统计字典（含逐步损失曲线）。

    参数
    ----
    config: config.yaml 字典（含 scheme / training / network / dataset 等）。
    out_dir: 输出目录（checkpoints/、logs/、seeds.json）。
    train_indices / val_indices: 只用于测试的索引子集钩子（生产为 None）。
    max_steps_override: 覆盖 config 的 max_steps（CLI ``--steps`` / ``--smoke``）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_version = resolve_data_version(config)
    spec_version = str(config.get("spec_version", ""))
    config_hash = config_digest(config)
    scheme = str(config["scheme"]).upper()

    precision = resolve_precision(config)
    device = resolve_device(config)
    ddp = _init_distributed()
    is_main = ddp is None or ddp["rank"] == 0

    max_steps = int(max_steps_override if max_steps_override is not None else config["training"]["max_steps"])
    train_cfg = config["training"]
    batch_size = int(train_cfg["batch_size"])
    val_interval = int(train_cfg.get("early_stopping", {}).get("val_interval", DEFAULT_VAL_INTERVAL))
    log_interval = int(train_cfg.get("log_interval", DEFAULT_LOG_INTERVAL))
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=int(es_cfg.get("patience", 10)),
        min_step_fraction=float(es_cfg.get("min_step_fraction", 0.5)),
        max_steps=max_steps,
        val_interval=val_interval,
    )

    # ---- 数据集（60 [S4][S14]）与方案 C 标准化统计量（60 [S5]）------------
    dataset_dir = PROJECT_ROOT / "data" / str(config["dataset"]["version"])
    train_ds = H5Dataset(dataset_dir / "train.h5", "train")
    if train_indices is not None:
        train_ds = Subset(train_ds, list(train_indices))
    val_ds = H5Dataset(dataset_dir / "val.h5", "val")
    if val_indices is not None:
        val_ds = Subset(val_ds, list(val_indices))

    shuffle_seed = _shuffle_seed(config, data_version)

    # 方案 C：标准化统计量只从训练集计算（60 [S13] AC5，★ 无测试集泄漏）
    c_prior_stats: tuple[np.ndarray, np.ndarray] | None = None
    if scheme == "C":
        raw = np.stack([train_ds[i]["c_prior_raw"].numpy() for i in range(len(train_ds))])
        c_prior_stats = compute_c_prior_stats(raw)

    # ---- 模型 / 优化器 / 损失 ---------------------------------------------
    scheme_seed = int(config.get("scheme_seeds", {}).get(f"scheme_{scheme}_seed_{config.get('seed_index', 0)}", 0))
    torch.manual_seed(scheme_seed)
    model = build_scheme_model(config)
    if scheme == "C":
        assert c_prior_stats is not None
        model.set_c_prior_stats(*c_prior_stats)
    model = model.to(device)
    loss_fn = HybridLoss(image_size=256)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    use_amp = precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = train_cfg.get("grad_clip_max")
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    # ---- 日志 / 种子 / checkpoint 目录 -------------------------------------
    logger = TrainingLogger(out_dir)
    seeds_path = write_seeds_json(config, out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_val.ckpt"
    last_path = ckpt_dir / "last.ckpt"

    header = {
        "scheme": scheme,
        "config_hash": config_hash,
        "data_version": data_version,
        "spec_version": spec_version,
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "shuffle_seed": shuffle_seed,
        "seeds_json": str(seeds_path),
        "checkpoint_dir": str(ckpt_dir),
    }
    logger.log(header)

    train_loader = _make_loader(train_ds, config, shuffle_seed, ddp)
    val_loader = _make_loader(val_ds, config, shuffle_seed, ddp, drop_last=False)

    curve: list[float] = []
    best_val_loss: float | None = None
    best_step = 0
    step = 0
    stopped = False
    epoch = 0
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if ddp is not None:
                train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            H_hat = forward_scheme(model, batch, device)
            # 工作尺度目标：H×S（60 [S2] 实现约定：损失在工作尺度空间计算，
            # L' = S·L 一阶齐次，最优解不变；见 50 [S12] C5 坍缩修复）
            target = batch["H"].to(device) * model.work_scale
            loss = loss_fn(H_hat, target).mean()
            l_space = loss_fn.l_space(H_hat, target)
            l_spec = loss_fn.l_spec(H_hat, target).mean()
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(loss.item())
        curve.append(loss_val)

        if step % log_interval == 0 or step == 1:
            with torch.no_grad():
                out_min = float(H_hat.min().item())
                out_max = float(H_hat.max().item())
                out_sum = float(H_hat.sum().item())
            record = {
                "step": step,
                "train_loss": loss_val,
                "l_space": float(l_space.item()),
                "l_spec": float(l_spec.item()),
                "out_min": out_min,
                "out_max": out_max,
                "out_sum": out_sum,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "config_hash": config_hash,
                "data_version": data_version,
                "spec_version": spec_version,
            }
            logger.log(record)

        # ---- 验证与早停（60 [S10]） ---------------------------------------
        if early_stopping.should_validate(step):
            val_loss = _validate(model, val_loader, loss_fn, device)
            if is_main:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    _save_checkpoint(best_path, model, config, step, val_loss, curve, c_prior_stats, header)
                _save_checkpoint(last_path, model, config, step, val_loss, curve, c_prior_stats, header)
                logger.log({
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "checkpoint_path": str(best_path),
                })
            if early_stopping.on_validation(step, val_loss):
                stopped = True
                break

    # ---- 收尾：最终 checkpoint（60 [S12] C2）-------------------------------
    if is_main:
        # 从未验证（如 smoke 短训练）时 best_val.ckpt = 最终权重
        _save_checkpoint(last_path, model, config, step, best_val_loss, curve, c_prior_stats, header)
        if not (ckpt_dir / "best_val.ckpt").exists():
            _save_checkpoint(best_path, model, config, step, best_val_loss, curve, c_prior_stats, header)
        logger.log({
            "step": step,
            "train_loss": curve[-1] if curve else float("nan"),
            "best_val_loss": best_val_loss,
            "stopped_early": stopped,
            "checkpoint_best": str(best_path),
            "checkpoint_last": str(last_path),
            "config_hash": config_hash,
            "data_version": data_version,
            "spec_version": spec_version,
        })

    if ddp is not None:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return {
        "scheme": scheme,
        "steps_run": step,
        "train_loss_curve": curve,
        "final_train_loss": curve[-1] if curve else None,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "seeds_json": str(seeds_path),
        "config_hash": config_hash,
        "data_version": data_version,
        "spec_version": spec_version,
    }


# ---------------------------------------------------------------------------
# CLI（60 [S15] 15.7）
# ---------------------------------------------------------------------------
def run_check_env() -> None:
    """运行 scripts/check_env.py（60 [S15] 15.5/15.7：仅警告，不阻塞）。"""
    script = PROJECT_ROOT / "scripts" / "check_env.py"
    if not script.exists():
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        if proc.returncode != 0:
            print("[check_env] 校验未通过（仅警告）：\n" + (proc.stdout + proc.stderr), file=sys.stderr)
    except Exception as exc:  # pragma: no cover - 环境异常仅警告
        print(f"[check_env] 无法运行：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.train",
        description="训练 A/B/C 方案（60 [S15] 15.7）",
    )
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--smoke", action="store_true", help="冒烟模式：100 步（可 --steps 覆盖）")
    parser.add_argument("--steps", type=int, default=None, help="覆盖 max_steps")
    parser.add_argument("--out", type=str, default=None, help="输出目录（默认按 15.2 命名）")
    args = parser.parse_args(argv)

    run_check_env()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config 不存在：{config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)
    max_steps = args.steps
    if args.smoke and max_steps is None:
        max_steps = 100
    out_dir = Path(args.out) if args.out else run_output_dir(config)

    try:
        stats = train(config, out_dir, max_steps_override=max_steps)
    except Exception as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 1
    print(f"[train] scheme={stats['scheme']} steps={stats['steps_run']} "
          f"final_train_loss={stats['final_train_loss']:.6g} "
          f"best_val_loss={stats['best_val_loss']} "
          f"checkpoint_best={stats['checkpoint_best']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
