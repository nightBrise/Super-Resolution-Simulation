"""推理 CLI：``python -m src.evaluation.infer --config <path> --split <...> --out <dir>``。

接口按 60 [S15] 15.7：复用 checkpoint（默认 ``best_val.ckpt``，60 [S12]
C3）对指定划分输出预测（``predictions.npz``：sample_id / H_hat / H /
L_up，供可视化与下游分析复用）。启动前先跑 ``scripts/check_env.py``
（仅警告）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.evaluation.evaluate import infer_predictions, load_model
from src.utils.config_utils import data_dir_for, load_config, resolve_device, run_output_dir
from src.utils.h5data import H5Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_check_env() -> None:
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
        prog="python -m src.evaluation.infer",
        description="推理（复用 checkpoint 输出预测，60 [S15] 15.7）",
    )
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--split", required=True, help="测试划分（HDF5 文件名）")
    parser.add_argument("--out", required=True, help="输出目录（predictions.npz）")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认 best_val.ckpt）")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    _run_check_env()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config 不存在：{config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)
    split_file = data_dir_for(config, PROJECT_ROOT) / str(config["dataset"]["version"]) / f"{args.split}.h5"
    if not split_file.exists():
        print(f"数据集不存在：{split_file}", file=sys.stderr)
        return 1
    run_dir = run_output_dir(config)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "best_val.ckpt"
    )
    if not checkpoint_path.exists():
        print(f"checkpoint 不存在：{checkpoint_path}", file=sys.stderr)
        return 1

    device = resolve_device(config)
    model, _ = load_model(config, checkpoint_path, device)
    dataset = H5Dataset(split_file, args.split)
    preds = infer_predictions(model, dataset, device, batch_size=args.batch_size)

    sample_ids = [dataset[i]["sample_id"] for i in range(len(dataset))]
    H = np.stack([dataset[i]["H"].numpy()[0] for i in range(len(dataset))])
    L_up = np.stack([dataset[i]["L_up"].numpy()[0] for i in range(len(dataset))])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{args.split}.npz"
    np.savez_compressed(
        out_path,
        sample_id=np.asarray(sample_ids, dtype=object),
        H_hat=preds.astype(np.float32),
        H=H.astype(np.float32),
        L_up=L_up.astype(np.float32),
    )
    print(f"[infer] scheme={config['scheme']} split={args.split} samples={len(sample_ids)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
