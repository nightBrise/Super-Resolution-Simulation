"""数据集生成 CLI：``python -m src.generators.build_dataset --config <path>``。

接口按 60 [S15] 15.7 表（批次二十补）：``--config`` 必填、``--split`` 可选
（train / val / test_id / test_pb / test_ood；省略时生成全部划分）。命令
启动前先跑 ``scripts/check_env.py`` 校验环境（15.5：不通过仅警告、登记 99，
不视为失败，不阻塞构建）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.generators.dataset_builder import (
    DEFAULT_WORKERS,
    build_dataset,
    git_head,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
            print(
                "[check_env] 校验未通过（仅警告）：\n" + (proc.stdout + proc.stderr),
                file=sys.stderr,
            )
    except Exception as exc:  # pragma: no cover - 环境异常仅警告
        print(f"[check_env] 无法运行：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generators.build_dataset",
        description="数据集生成与划分（60 [S8] + [S14]）",
    )
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test_id", "test_pb", "test_ood"],
        action="append",
        help="只构建指定划分（可重复；省略时构建全部）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"并行进程数（默认配置值或 {DEFAULT_WORKERS}）",
    )
    args = parser.parse_args(argv)

    run_check_env()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config 不存在：{config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)

    manifest = build_dataset(
        config,
        PROJECT_ROOT,
        splits=args.split,
        workers=args.workers,
        code_version=config.get("code_version") or git_head(),
    )

    version = config["dataset"]["version"]
    if args.split:
        print(f"已构建 {len(args.split)} 个划分：{', '.join(args.split)}")
        print(f"位置：{PROJECT_ROOT / 'data' / version}/")
    else:
        counts = {k: v["count"] for k, v in manifest["splits"].items()}
        print(f"数据集 v{version} 构建完成：{counts}")
        print(f"位置：{PROJECT_ROOT / 'data' / version}/")
        print(f"manifest：{PROJECT_ROOT / 'data' / version / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
