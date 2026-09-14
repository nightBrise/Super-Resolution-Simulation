"""L2 [S4] 容差线产物 CLI：`20` [S4] 的 95% 分位偏差表（H8 判定的尺子）。

按 `20` [S4] 预注册规则：观测量 q 的容差 = 该档裸子样本 |q_裸 − q_参照| 分布的
95% 分位（池化口径见 `20_...decisions.md` D15），档位映射由 N_eff 对数最近档决定。

与 E0（`40` [S1]）同属"纯数据尺子"——只消费 `10` [S5] 数据集的裸子样本，与任何
模型输出无关；因此其时间戳不参与 `40` [S1] C5 的 E0 顺序约束，但一并落盘以便复算。

用法：
    python -m src.line2.experiments.tolerance_table [--data-dir <dir>] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.line2.metrics.tolerance import deviation_table

DEFAULT_DATA_DIR = Path("studies/line2_particle_sr/data/line2_ds_v1")
DEFAULT_OUT = Path("studies/line2_particle_sr/results/tolerance.json")
TIERS = (1000, 3000, 10000, 30000, 100000)
K = 20
QUANTILE = 0.95
SCOPE = "valtest_pooled"

__all__ = ["main", "build_payload", "DEFAULT_DATA_DIR", "DEFAULT_OUT"]


def build_payload(
    table: Mapping[int, np.ndarray],
    *,
    beam_ids: Sequence[int],
    tiers: Sequence[int],
    k: int,
    data_dir: str | Path,
    beams_scope: str,
    code_version: str,
    data_version: str | None = None,
    spec_version: str | None = None,
    quantile: float = QUANTILE,
    scope: str = SCOPE,
    started_at: str | None = None,
    completed_at: str | None = None,
    elapsed_s: float | None = None,
) -> dict:
    """`deviation_table` 输出 → 落盘 payload（纯函数：无 IO、无时钟、无随机量）。

    ``dev_table`` 的键按 `20` [S4] 落成**字符串档位**，值 = 13 维分位向量的列表——
    消费侧 :func:`src.line2.experiments.gates._load_tolerance` 与 `map_report` 按此格式
    读回（键 `int()` 还原、值转 ndarray）。
    """
    return {
        "quantile": float(quantile),
        "scope": scope,
        "tiers": [int(t) for t in tiers],
        "k": int(k),
        "beams_scope": beams_scope,
        "n_beams": len(beam_ids),
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_s": None if elapsed_s is None else round(float(elapsed_s), 1),
        "data_dir": str(data_dir),
        "data_version": data_version,
        "spec_version": spec_version,
        "code_version": code_version,
        "dev_table": {
            str(int(t)): [float(v) for v in np.asarray(table[int(t)]).ravel()] for t in tiers
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 容差线表（spec 20 [S4]）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--beams", choices=("valtest", "val", "test"), default="valtest")
    parser.add_argument("--tiers", type=int, nargs="+", default=list(TIERS))
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args(argv)

    from src.line2.experiments.e0 import code_version

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.beams == "valtest":
        beam_ids = [int(b) for s in ("val", "test") for b in manifest["splits"][s]]
    else:
        beam_ids = [int(b) for b in manifest["splits"][args.beams]]

    t0 = time.time()
    table = deviation_table(
        args.data_dir, beam_ids, tuple(args.tiers), args.k, quantile=QUANTILE
    )
    payload = build_payload(
        table, beam_ids=beam_ids, tiers=args.tiers, k=args.k, data_dir=args.data_dir,
        beams_scope=args.beams, code_version=code_version(),
        data_version=manifest.get("data_version"), spec_version=manifest.get("spec_version"),
        started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_s=time.time() - t0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_beams": payload["n_beams"],
                "elapsed_s": payload["elapsed_s"],
                "dev_table_keys": sorted(payload["dev_table"]),
                "dims": {t: len(v) for t, v in payload["dev_table"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
