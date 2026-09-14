"""L4/L5 [S4] 三类产出物一键脚本（spec line2/40 [S4]）。

只读消费 `results/` 落盘产物，生成 Phase 1 交付物（`00` [S3] C7）：

1. **方法可用域地图** ``<figure-dir>/figure_avail_domain.png``（PNG only、300 dpi）；
2. **N_eff 明细表** ``<out-dir>/n_eff_table.csv``（runs × 站 × N_low，含上/下溢标记）；
3. **实验报告** ``<reports-dir>/phase1_report.md``——章节 1 数据 / 2 设置 / 3 结果 /
   4 失败模式与不可验证项 / 5 门禁结论 / 6 结论范围 / 7 设计理由（Why）。

契约：

- **只读**（[S4] Global Constraints）：不训练、不推理、不写 run 目录；写入面 = 上述三个路径。
- **同输入逐位一致**（[S4] C1）：报告与表格不含生成时刻的墙钟时间、git HEAD 或随机量，
  全部数值取自输入产物；PNG 由 matplotlib Agg 后端写出（同版本同输入逐字节一致）。
- **缺输入明确报错**（不静默跳过）：``e0_curve.json``、``e1_gates.json``、至少一份
  ``*/test_eval*.json`` 为必需，缺文件抛 :class:`FileNotFoundError`、缺键抛 :class:`ValueError`；
  可选输入（``e0_curve.meta.json``、``<run_id>/run_meta.json``、门禁产物的非判定段）缺失时
  不中断，但一律登记到报告第 4 章"失败模式与不可验证项"。
- **每条门禁结论附判定表达式的输入数值快照**（[S4] C2）：快照取自 ``e1_gates.json`` 各门禁段的
  ``inputs`` / ``thresholds`` / ``per_observable`` / ``pairing``（由
  :func:`~src.line2.experiments.gates.evaluate_gates_from_disk` 落盘），本模块不重算判定。

地图坐标语义（两面板；[S4]"主曲线 N_eff/N_low vs N_low"）：

- **面板 A（主曲线）**：x = N_low，y = N_eff/N_low（无量纲放大比），双对数。N_eff 恒为
  `20` [S3] 报告域 [1e3, 1e5] 内的值——上/下溢行取域端点，以空心叉号显式标注；灰色带 =
  报告域在比值坐标下的像 [1e3/N_low, 1e5/N_low]，点线 = 无放大（N_eff = N_low）。
- **面板 B（E0 参考线）**：x = N，y = 裸误差 M1（中位数 + IQR 带）与参照噪声地板水平线
  （`20` [S2] C3），并叠加各 run 的 ``err_median`` 散点——网络输出在这把尺上的落点。

两面板分画而非共 x 双 y：面板 A 的 y 是无量纲比值、面板 B 的 y 是相对 RMSE，量纲不同；
叠在同一像素高度会被读作可比（两线的视觉交点无语义）。分画后二者共享 x 的粒子数语义，
参照噪声地板画在它被定义的 M1 面板内。图内标题/轴标签/图例用 ASCII 英文：中文字形依赖
运行机的 CJK 字体，缺字体时降级为方框且跨机渲染不同，会破坏逐位一致。

用法::

    python -m src.line2.experiments.map_report --results-dir <dir> \
        [--out-dir <dir>] [--reports-dir <dir>] [--figure-dir <dir>]

``--out-dir`` 默认 = ``--results-dir``（只托管 N_eff 明细表）；``--reports-dir`` 默认 =
``<results-dir>/../reports``（`00` [S2] 布局：results 与 reports 同为研究线根的子目录）；
``--figure-dir`` 默认 = ``--reports-dir``——图与报告同目录，随 ``reports/`` 一并交付入库
（``results/`` 内的子目录无法白名单化，见 .gitignore）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "amplify_groups",
    "check_e0_precedes_runs",
    "collect_n_eff_table",
    "collect_run_metas",
    "load_gate_results",
    "load_run_meta",
    "render_avail_domain",
    "render_report",
    "run_dir_triple",
    "write_n_eff_table",
    "generate",
    "main",
    "TABLE_COLUMNS",
    "FIGURE_NAME",
    "TABLE_NAME",
    "REPORT_NAME",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CURVE_FILENAME = "e0_curve.json"          # `20` [S2] 曲线 dump（e0.py --out）
E0_META_FILENAME = "e0_curve.meta.json"    # e0.py 侧文件：时间戳与溯源（`40` [S1] C5 证据）
GATES_FILENAME = "e1_gates.json"           # gates.evaluate_gates_from_disk 产物
EVAL_GLOB = "test_eval*.json"              # evaluate.py 产物（E3 跨站评估另起名，同 glob 覆盖）
RUN_META_FILENAME = "run_meta.json"        # 调度器落盘的 run 级溯源（冻结 schema，见下）
SUMMARY_FILENAME = "summary.json"          # train.py 产物
CONFIG_FILENAME = "config.json"            # run 目录内的 config 副本（train.py 不写，缺失只登记）
#: `40` [S1] C2 run 目录契约中只登记存在性（不做内容校验）的文件：训练日志与 ckpt
RUN_DIR_PRESENCE_ONLY = ("train_log.jsonl", "best_val.ckpt", "last.ckpt")

FIGURE_NAME = "figure_avail_domain.png"
TABLE_NAME = "n_eff_table.csv"
REPORT_NAME = "phase1_report.md"
FIGURE_DPI = 300                          # line1 60 [S15] 15.3 沿用的出图分辨率

#: N_eff 表列序（`40` [S4]"runs × 站 × N_low 明细表"；冻结，不得增删换序）
TABLE_COLUMNS = (
    "run_id", "variant", "seed", "n_low", "station", "split",
    "n_eff", "n_eff_flag", "err_median", "n_beams",
)

#: `test_eval.json` 的必需字段（evaluate.py 冻结 schema；``n_eff_flag`` 键必须在，值可为 null）
EVAL_REQUIRED_FIELDS = (
    "run_id", "variant", "seed", "split", "station", "n_low",
    "err_median", "n_eff", "n_eff_flag", "n_beams",
)

#: `run_meta.json` 的冻结字段（调度器落盘；`00` GC7 三元组 + `40` [S1] C5 时间戳）
RUN_META_REQUIRED = (
    "started_at", "completed_at", "code_version", "data_version", "spec_version",
    "variant", "seed", "n_low", "max_steps",
)

#: `e1_gates.json` 的必需顶层键（缺一即报错——门禁结论是报告第 5 章的唯一来源）
GATES_REQUIRED_KEYS = ("winner_variant", "H1", "H3", "H8")
#: 跨站门禁段：现产出门禁写 ``H7``；``E3`` 为同义键名（矩阵编号口径）
GATES_CROSS_KEYS = ("H7", "E3")
#: 束级 N_eff 快照段：现产出门禁写 ``beam_n_eff_snapshot``；``snapshots`` 为同义键名
GATES_SNAPSHOT_KEYS = ("beam_n_eff_snapshot", "snapshots")
#: 报告第 2/5 章按需渲染的门禁溯源段（缺失登记到第 4 章，不中断）
GATES_OPTIONAL_SECTIONS = (
    "routing_basis", "gate_dims", "val_segment", "test_segment", "curve", "runs", "code_version",
)

#: `20` [S3] N_eff 报告域（上/下溢取端点）
N_EFF_DOMAIN = (1_000.0, 100_000.0)

_GIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
_PALETTE = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860")

#: 各门禁段的必需快照键（`40` [S4] C2：判定表达式的输入数值必须落进报告）
_GATE_SNAPSHOT_KEYS = {
    "H1": ("verdict", "variant", "n_seeds", "median_n_eff", "n_above_n_low", "thresholds", "inputs"),
    "H3": ("verdict", "n_pairs", "joint_wins", "pairing", "inputs"),
    "H7": ("verdict", "rho", "weak", "decay", "variant", "n_seeds", "aggregation",
           "thresholds", "per_seed"),
    "H8": ("verdict", "pass_rate", "n_beams", "beam_ids", "seeds", "n_seeds", "tolerance",
           "per_observable"),
}
#: 条件落盘的快照键（判定不可达时产出不写；缺失以 reason_code 显式解释，不静默）
_GATE_OPTIONAL_KEYS = {
    "H1": ("reason_code",),
    # H3 的披露字段：平局数 / 严格获胜数 / 域顶平局标志 + 其证据（成对 N_eff 的落盘标记）
    "H3": ("n_ties", "n_strict_joint_wins", "tied_at_domain_edge", "n_eff_flags",
           "reason_code"),
    "H7": ("n_eff_station0_median", "n_eff_station1_median", "reason_code"),
    "H8": ("variant", "reason_code"),
}
#: 冻结判定表达式（`40` [S3]；报告据此说明每个快照数字如何被消费）
_GATE_RULES = {
    "H1": ("放大存在性：获胜变体 @1e4 的 5 种子 test N_eff 分布——成立 ⟺ 5/5 种子 N_eff > 1e4 "
           "且中位数 ≥ 2×1e4；不成立 ⟺ 中位数 ≤ 1e4；其余不可判定。"),
    "H3": ("联合优于独立：同种子成对 5 对——成立 ⟺ 5/5 对联合 N_eff ≥ 独立；方向成立 ⟺ 4/5；"
           "不成立 ⟺ ≤ 2/5；3/5 不可判定。"),
    "H7": ("跨站泛化：ρ = N_eff(站1)/N_eff(站0)——成立 ⟺ ρ ≥ 1.0；成立（弱，报衰减系数）⟺ "
           "0.5 ≤ ρ < 1.0；不成立 ⟺ ρ < 0.5；N_eff 触发上/下溢标记使 ρ 无法计算 ⟺ 不可判定"
           "（任一侧携带标记即降级为不可判定，此时 ρ 为截断值之比、仅作披露）。"),
    "H8": ("细结构可恢复性：13 项观测量对容差线的达标率（test 束 × 站 0 × 1e4 × 5 种子中位数输出）"
           "——成立 ⟺ ≥ 80%；不成立 ⟺ ≤ 50%；之间不可判定。"),
}


# --------------------------------------------------------------------- 基础读取
def _display_path(path: str | Path) -> str:
    """产物路径的报告字面量：项目根内写相对路径，根外（如临时目录）写绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: str | Path, what: str) -> dict:
    """读必需 JSON 产物；缺文件报错并说明它是哪份契约的输入（不静默跳过）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"缺产物：{_display_path(path)}（{what}）")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{_display_path(path)}：JSON 解析失败（{exc}）") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{_display_path(path)}：顶层应为对象，实为 {type(payload).__name__}")
    return payload


def _read_json_optional(path: str | Path) -> dict | None:
    """读可选 JSON 产物；缺失返回 None（调用方负责把缺失登记进报告第 4 章）。"""
    path = Path(path)
    return _read_json(path, "可选溯源侧文件") if path.is_file() else None


def _require_keys(payload: Mapping, keys: Sequence[str], source: str, contract: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"{source}：缺键 {missing}（{contract}）")


def _fmt(value: object) -> str:
    """报告用数值字面量：float 取 6 位有效数字，None → ``—``（确定性，无墙钟量）。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        return f"{value:.6g}"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, Sequence):
        parts = [_fmt(item) for item in value]
        nested = all(
            isinstance(item, (Sequence, Mapping)) and not isinstance(item, str) for item in value
        )
        return ("; ".join(parts) if nested else ", ".join(parts)) if parts else "—"
    return str(value)


def _parse_ts(text: object, source: str) -> datetime:
    """ISO 8601 时间戳 → 带时区 datetime；naive 一律按 UTC 解释（跨产物可比）。"""
    try:
        ts = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{source}：时间戳 {text!r} 不可解析（期望 ISO 8601）") from exc
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """markdown 表格行（表头 + 分隔 + 数据行，单元格经 :func:`_fmt`）。"""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows)
    return out


# ------------------------------------------------------------------ N_eff 明细表
def collect_n_eff_table(results_dir: str | Path) -> list[dict]:
    """`40` [S4] N_eff 明细表行：runs × 站 × N_low（含上/下溢标记）。

    消费 ``<results_dir>/*/test_eval*.json``（`evaluate.py` 冻结字段），行序确定为
    (站, N_low, 变体, 种子, run_id)——同输入两次调用逐位一致（[S4] C1）。

    强制的契约：``split`` 必须为 test（`40` [S3] 两段式消费、`00` GC3 出口门禁只吃 test 段）；
    同一 (run_id, 站, N_low, split) 不得出现两份评估（test 只评一次）；N_eff 必须落在
    `20` [S3] 报告域内，越界必须携带与之匹配的显式标记（[S3] C2）。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    paths = sorted(results_dir.glob(f"*/{EVAL_GLOB}"))
    if not paths:
        raise FileNotFoundError(
            f"{_display_path(results_dir)}：未找到任何 */{EVAL_GLOB}——N_eff 表无输入"
        )
    rows: list[dict] = []
    seen: dict[tuple, str] = {}
    for path in paths:
        payload = _read_json(path, "run 级 test 评估")
        source = _display_path(path)
        _require_keys(payload, EVAL_REQUIRED_FIELDS, source, "evaluate.py 冻结 schema")
        if payload["split"] != "test":
            raise ValueError(
                f"{source}：split={payload['split']!r}——产出物只消费 test 段"
                "（`40` [S3] 两段式判据消费、`00` GC3）"
            )
        n_eff = float(payload["n_eff"])
        flag = payload["n_eff_flag"]
        if flag is not None:
            expected = N_EFF_DOMAIN[1] if flag == "overflow" else N_EFF_DOMAIN[0]
            if flag not in ("overflow", "underflow"):
                raise ValueError(f"{source}：n_eff_flag={flag!r} 不是 overflow/underflow/null")
            if n_eff != expected:
                raise ValueError(
                    f"{source}：{flag} 行的 n_eff={n_eff} 应为报告域端点 {expected}"
                    "（`20` [S3] C2：上/下溢取端点 + 显式标记）"
                )
        elif not N_EFF_DOMAIN[0] <= n_eff <= N_EFF_DOMAIN[1]:
            raise ValueError(
                f"{source}：n_eff={n_eff} 越出报告域 {N_EFF_DOMAIN} 且无标记"
                "（`20` [S3] C2：不得无标记越界）"
            )
        key = (str(payload["run_id"]), int(payload["station"]), int(payload["n_low"]),
               str(payload["split"]))
        if key in seen:
            raise ValueError(
                f"{source}：与 {seen[key]} 重复——同 (run_id, 站, N_low, split) 只评一次（`00` GC3）"
            )
        seen[key] = source
        rows.append({col: payload[col] for col in TABLE_COLUMNS})
    rows.sort(key=lambda r: (int(r["station"]), int(r["n_low"]), str(r["variant"]),
                             int(r["seed"]), str(r["run_id"])))
    return rows


def write_n_eff_table(rows: Sequence[Mapping], out_path: str | Path) -> Path:
    """N_eff 明细表 → CSV（列序 = :data:`TABLE_COLUMNS`，LF 行尾，``n_eff_flag`` 空值写空串）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TABLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = {col: row[col] for col in TABLE_COLUMNS}
            if record["n_eff_flag"] is None:
                record["n_eff_flag"] = ""
            writer.writerow(record)
    return out_path


# ------------------------------------------------------------------- run 目录契约
def load_run_meta(run_dir: str | Path) -> dict:
    """读 ``<run_dir>/run_meta.json`` 并校验冻结 schema（`40` [S1] C2、`00` GC7）。

    缺文件 → :class:`FileNotFoundError`；缺字段或 ``code_version`` 非完整 40 位 git hash
    → :class:`ValueError`。
    """
    path = Path(run_dir) / RUN_META_FILENAME
    meta = _read_json(path, "run 级溯源（`40` [S1] C2 run 目录契约）")
    _require_keys(meta, RUN_META_REQUIRED, _display_path(path), f"run_meta 冻结 schema {RUN_META_REQUIRED}")
    code = str(meta["code_version"])
    if not _GIT_HASH_RE.fullmatch(code):
        raise ValueError(
            f"{_display_path(path)}：code_version={code!r} 不是完整 40 位 git hash（`00` GC7）"
        )
    return meta


def run_dir_triple(run_dir: str | Path) -> dict:
    """run 目录契约（`40` [S1] C2）：GC7 三元组 + 目录内产物一致性。

    返回 ``run_id`` / 三元组（``data_version``、``code_version``、``spec_version``）/
    ``variant`` / ``seed`` / ``n_low`` / ``max_steps`` / 时间戳，外加：

    - ``present``：目录内实际落盘的契约文件（run_meta.json、summary.json、config.json、
      训练日志、ckpt）；
    - ``missing``：未落盘的契约文件——`train.py` 只写 ``summary.json``、``train_log.jsonl``
      与 ckpt（config 只以冻结副本形式存在于 ``docs/specs/line2/e1_configs/``，调度器
      ``run_e1`` 从该目录读取，不向 run 目录复制），故缺失只登记到报告第 4 章，
      不报错；
    - ``best_val_n_eff``：val 段选型准绳（`00` GC3），summary.json 缺失时为 None。

    ``run_meta.json`` 与 ``summary.json``/``config.json`` 的 variant/seed/n_low/max_steps
    冲突 → :class:`ValueError`（同一 run 的两份溯源必须指向同一次训练）。
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run 目录不存在：{_display_path(run_dir)}")
    meta = load_run_meta(run_dir)
    record: dict = {
        "run_id": run_dir.name,
        "data_version": meta["data_version"],
        "code_version": meta["code_version"],
        "spec_version": meta["spec_version"],
        "variant": meta["variant"],
        "seed": int(meta["seed"]),
        "n_low": int(meta["n_low"]),
        "max_steps": int(meta["max_steps"]),
        "started_at": meta["started_at"],
        "completed_at": meta["completed_at"],
        "best_val_n_eff": None,
        "present": [RUN_META_FILENAME],
        "missing": [],
    }
    summary_path = run_dir / SUMMARY_FILENAME
    config_path = run_dir / CONFIG_FILENAME
    for name in RUN_DIR_PRESENCE_ONLY:
        (record["present"] if (run_dir / name).is_file() else record["missing"]).append(name)
    for path, shared_keys in (
        (summary_path, ("variant", "seed")),
        (config_path, ("variant", "seed", "n_low", "max_steps")),
    ):
        if not path.is_file():
            record["missing"].append(path.name)
            continue
        record["present"].append(path.name)
        payload = _read_json(path, f"run 目录产物 {path.name}")
        for key in shared_keys:
            if key not in payload:
                continue  # summary/config 各自的字段集由 `30` 定义，此处只校验共有键的取值一致
            if key in ("seed", "n_low", "max_steps"):
                if int(payload[key]) != record[key]:
                    raise ValueError(
                        f"{_display_path(path)}：{key}={payload[key]!r} 与 run_meta 的 "
                        f"{record[key]!r} 冲突（`40` [S1] C2 run 目录契约）"
                    )
            elif str(payload[key]) != str(record[key]):
                raise ValueError(
                    f"{_display_path(path)}：{key}={payload[key]!r} 与 run_meta 的 "
                    f"{record[key]!r} 冲突（`40` [S1] C2 run 目录契约）"
                )
        if "best_val_n_eff" in payload:
            record["best_val_n_eff"] = payload["best_val_n_eff"]
    return record


def collect_run_metas(results_dir: str | Path) -> dict[str, dict]:
    """``<results_dir>/*/run_meta.json`` → {run_id: :func:`run_dir_triple` 记录}。

    只扫存在 ``run_meta.json`` 的 run 目录；一个都没有 → 空字典（调用方登记为不可验证项）。
    存在但 schema 不合法 → 报错（半份溯源比没有更危险）。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    out: dict[str, dict] = {}
    for path in sorted(results_dir.glob(f"*/{RUN_META_FILENAME}")):
        record = run_dir_triple(path.parent)
        out[record["run_id"]] = record
    return out


# ------------------------------------------------------------------ `40` [S1] C5
def check_e0_precedes_runs(
    results_dir: str | Path, run_metas: Mapping[str, Mapping] | None = None
) -> dict:
    """尺子先行审计（`40` [S1] C5、D8）：E0 完成时间戳 SHALL 早于每个 run 的启动时间戳。

    证据 = ``e0_curve.meta.json`` 的 ``completed_at`` 与各 run 目录 ``run_meta.json`` 的
    ``started_at``。返回 ``{"verifiable", "e0_completed_at", "checked", "missing", "violations"}``：

    - 证据齐全且存在 ``started_at ≤ completed_at`` 的 run → :class:`ValueError`
      （顺序约束被破坏 = 该批训练在尺子之前跑，矩阵无效）；
    - 证据缺失（无 E0 侧文件或无任一 run_meta）→ ``verifiable=False`` + ``missing`` 明细，
      由报告第 4 章显式登记为不可验证项（不静默跳过、也不据此判定通过）。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    missing: list[str] = []
    e0_completed: str | None = None
    meta_path = results_dir / E0_META_FILENAME
    meta = _read_json_optional(meta_path)
    if meta is None:
        missing.append(f"{_display_path(meta_path)}（E0 completed_at 无来源）")
    else:
        e0_completed = meta.get("completed_at")
        if e0_completed is None:
            missing.append(f"{_display_path(meta_path)}:completed_at")
    metas = dict(collect_run_metas(results_dir) if run_metas is None else run_metas)
    if not metas:
        missing.append(f"{_display_path(results_dir)}/*/{RUN_META_FILENAME}（无任一 run 时间戳）")

    checked: dict[str, str] = {}
    violations: list[dict] = []
    if e0_completed is not None and metas:
        t_e0 = _parse_ts(e0_completed, _display_path(meta_path))
        for run_id in sorted(metas):
            started = metas[run_id]["started_at"]
            checked[run_id] = str(started)
            if _parse_ts(started, f"{run_id}/{RUN_META_FILENAME}") <= t_e0:
                violations.append(
                    {"run_id": run_id, "started_at": str(started), "e0_completed_at": str(e0_completed)}
                )
    if violations:
        detail = "; ".join(
            f"{v['run_id']} started_at={v['started_at']} ≤ E0 completed_at={v['e0_completed_at']}"
            for v in violations
        )
        raise ValueError(
            f"`40` [S1] C5 尺子先行被破坏：{detail}——E0 曲线必须先于任何 E1 训练完成"
        )
    return {
        "verifiable": bool(checked) and e0_completed is not None,
        "e0_completed_at": e0_completed,
        "checked": checked,
        "missing": missing,
        "violations": violations,
    }


# ------------------------------------------------------------------- 门禁产物读取
def load_gate_results(results_dir: str | Path) -> dict:
    """读 ``e1_gates.json`` 并校验门禁段齐备（`40` [S3] C1 的唯一判定来源）。

    必需：``winner_variant`` + H1/H3/H8 段 + 跨站段（``H7``，同义键 ``E3``）。
    每段再按 :data:`_GATE_SNAPSHOT_KEYS` 校验快照键齐备——缺键即报错，因为 [S4] C2
    要求报告附判定表达式的输入数值，无快照的结论无法被复核。
    """
    path = Path(results_dir) / GATES_FILENAME
    gates = _read_json(path, "出口门禁产物（gates.evaluate_gates_from_disk）")
    _require_keys(gates, GATES_REQUIRED_KEYS, _display_path(path), "`40` [S3] 门禁段")
    if not any(key in gates for key in GATES_CROSS_KEYS):
        raise ValueError(
            f"{_display_path(path)}：缺跨站门禁段（{GATES_CROSS_KEYS} 任一）——`40` [S2] H7 无结论"
        )
    cross = cross_station_gate(gates)
    for name, gate in (("H1", gates["H1"]), ("H3", gates["H3"]), ("H7", cross), ("H8", gates["H8"])):
        if not isinstance(gate, Mapping):
            raise ValueError(
                f"{_display_path(path)}:{name} 应为携带判定快照的对象，实为 "
                f"{type(gate).__name__}——`40` [S4] C2 要求每条结论附判定表达式的输入数值"
            )
        _require_keys(gate, _GATE_SNAPSHOT_KEYS[name], f"{_display_path(path)}:{name}",
                      f"`40` [S4] C2 判定快照键 {_GATE_SNAPSHOT_KEYS[name]}")
    return gates


def cross_station_gate(gates: Mapping) -> dict:
    """跨站门禁段（``H7``，同义键 ``E3``）；:func:`load_gate_results` 已保证其一存在。"""
    for key in GATES_CROSS_KEYS:
        if key in gates:
            return dict(gates[key])
    raise ValueError(f"门禁产物缺跨站段（{GATES_CROSS_KEYS} 任一）")


def beam_snapshot(gates: Mapping) -> dict | None:
    """束级 N_eff 快照段（``beam_n_eff_snapshot``，同义键 ``snapshots``）；缺失返回 None。"""
    for key in GATES_SNAPSHOT_KEYS:
        if key in gates:
            return dict(gates[key])
    return None


# ----------------------------------------------------------------------- 主曲线聚合
def amplify_groups(table: Sequence[Mapping]) -> list[dict]:
    """N_eff 表 → 主曲线聚合：按 (站, 变体, N_low) 分组的放大比 N_eff/N_low。

    组内按种子升序（确定性），并给出中位数/极值与上/下溢计数——图（面板 A 的种子散布与
    中位数线）与报告（第 3 章三点主曲线）共用这一份聚合，避免两处各算一遍。
    """
    if not table:
        raise ValueError("amplify_groups：N_eff 表为空")
    buckets: dict[tuple, dict] = {}
    for row in table:
        key = (int(row["station"]), str(row["variant"]), int(row["n_low"]))
        bucket = buckets.setdefault(
            key, {"station": key[0], "variant": key[1], "n_low": key[2],
                  "seeds": [], "ratios": [], "flags": [], "err_medians": [], "run_ids": []}
        )
        bucket["seeds"].append(int(row["seed"]))
        bucket["ratios"].append(float(row["n_eff"]) / float(row["n_low"]))
        bucket["flags"].append(row["n_eff_flag"])
        bucket["err_medians"].append(float(row["err_median"]))
        bucket["run_ids"].append(str(row["run_id"]))
    groups = []
    for key in sorted(buckets):
        g = buckets[key]
        order = sorted(range(len(g["seeds"])), key=lambda i: (g["seeds"][i], g["run_ids"][i]))
        ratios = [g["ratios"][i] for i in order]
        flags = [g["flags"][i] for i in order]
        groups.append(
            {
                "station": g["station"], "variant": g["variant"], "n_low": g["n_low"],
                "seeds": [g["seeds"][i] for i in order],
                "run_ids": [g["run_ids"][i] for i in order],
                "ratios": ratios, "flags": flags,
                "err_medians": [g["err_medians"][i] for i in order],
                "n_runs": len(order),
                "ratio_median": float(np.median(np.asarray(ratios, dtype=np.float64))),
                "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
                "err_median_median": float(np.median(np.asarray(g["err_medians"], dtype=np.float64))),
                "n_overflow": sum(1 for f in flags if f == "overflow"),
                "n_underflow": sum(1 for f in flags if f == "underflow"),
            }
        )
    return groups


# --------------------------------------------------------------------------- 地图
def render_avail_domain(
    groups: Sequence[Mapping], curve: Mapping, out_path: str | Path
) -> Path:
    """方法可用域地图（`40` [S4] C1）：主曲线 + E0 参考线 + 参照噪声地板线 → PNG。

    ``groups`` = :func:`amplify_groups` 输出；``curve`` = ``e0_curve.json`` 的解析结果
    （需 ``points[{n_low, median, iqr_lo, iqr_hi}]`` 与 ``reference_noise_floor``）。
    坐标语义与两面板取舍见模块 docstring。PNG only、300 dpi、命名沿 ``figure_*.png``。
    """
    points = curve.get("points")
    if not points:
        raise ValueError("e0_curve.json：points 缺失或为空——E0 参考线无输入（`20` [S2]）")
    floor = curve.get("reference_noise_floor")
    if floor is None or not math.isfinite(float(floor)):
        raise ValueError(
            f"e0_curve.json：reference_noise_floor={floor!r} 非有限——"
            "`40` [S4] C1 要求地图含参照噪声地板线（`20` [S2] C3）"
        )
    for i, point in enumerate(points):
        _require_keys(point, ("n_low", "median", "iqr_lo", "iqr_hi"),
                      f"e0_curve.json:points[{i}]", "`20` [S2] 曲线点字段")

    variants = sorted({str(g["variant"]) for g in groups})
    color_of = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(variants)}
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 4.8))

    # ---- 面板 A：主曲线 N_eff/N_low vs N_low
    n_lows = sorted({int(g["n_low"]) for g in groups})
    x_lo, x_hi = n_lows[0] / 2.0, n_lows[-1] * 2.0
    band_x = np.geomspace(x_lo, x_hi, 64)
    ax_a.fill_between(
        band_x, N_EFF_DOMAIN[0] / band_x, N_EFF_DOMAIN[1] / band_x,
        color="#999999", alpha=0.16, linewidth=0,
        label="verifiable domain: N_eff in [1e3, 1e5] (20 [S3])",
        zorder=1,
    )
    ax_a.plot(band_x, np.ones_like(band_x), color="#666666", linestyle=":", linewidth=1.1,
              label="no amplification (N_eff = N_low)", zorder=2)
    ax_a.plot(band_x, N_EFF_DOMAIN[1] / band_x, color="#444444", linestyle="-.", linewidth=1.1,
              label="domain ceiling N_eff = 1e5 (10x at N_low=1e4)", zorder=2)

    for variant in variants:
        color = color_of[variant]
        for station in sorted({int(g["station"]) for g in groups if str(g["variant"]) == variant}):
            sub = sorted((g for g in groups if str(g["variant"]) == variant
                          and int(g["station"]) == station), key=lambda g: int(g["n_low"]))
            xs = [x for g in sub for x in [int(g["n_low"])] * int(g["n_runs"])]
            ys = [r for g in sub for r in g["ratios"]]
            marker = "o" if station == 0 else "^"
            ax_a.scatter(xs, ys, s=24, color=color, marker=marker, alpha=0.8, zorder=4,
                         facecolors=color if station == 0 else "none",
                         label=f"station {station} {variant} ({len(ys)} pts)")
            if station == 0:
                ax_a.plot([int(g["n_low"]) for g in sub], [float(g["ratio_median"]) for g in sub],
                          color=color, linestyle="--", linewidth=1.2, marker="o", markersize=4,
                          zorder=5, label=f"station 0 {variant} seed median")

    flagged_x = [int(g["n_low"]) for g in groups for f in g["flags"] if f]
    flagged_y = [float(r) for g in groups for r, f in zip(g["ratios"], g["flags"]) if f]
    if flagged_x:
        ax_a.scatter(flagged_x, flagged_y, s=58, marker="X", facecolors="none",
                     edgecolors="#222222", linewidths=1.0, zorder=6,
                     label=f"over/underflow: N_eff at domain edge ({len(flagged_x)} pts)")
    ratios_all = [float(r) for g in groups for r in g["ratios"]]
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(x_lo, x_hi)
    ax_a.set_ylim(min(1.0, min(ratios_all)) / 1.6,
                  max(N_EFF_DOMAIN[1] / n_lows[0], max(ratios_all)) * 1.6)
    ax_a.set_xlabel("N_low (input particles)")
    ax_a.set_ylabel("N_eff / N_low (amplification ratio)")
    ax_a.set_title("(a) Main curve: N_eff/N_low vs N_low (E1 1e4 + E2 3e3/1e5)", fontsize=10)
    ax_a.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax_a.legend(fontsize=6.5, loc="upper right")

    # ---- 面板 B：E0 参考线（裸 M1 vs N）+ 参照噪声地板 + 网络 err_median 落点
    ns = [int(p["n_low"]) for p in points]
    medians = [float(p["median"]) for p in points]
    ax_b.fill_between(ns, [float(p["iqr_lo"]) for p in points], [float(p["iqr_hi"]) for p in points],
                      color=color_of[variants[0]] if variants else "#4c72b0", alpha=0.20,
                      linewidth=0, label="E0 IQR (25-75%)", zorder=1)
    ax_b.plot(ns, medians, color="#4c72b0", marker="o", linewidth=1.4, zorder=3,
              label="E0 raw M1 median (20 [S2])")
    ax_b.axhline(float(floor), color="#c44e52", linestyle="--", linewidth=1.3, zorder=2,
                 label=f"reference noise floor = {_fmt(float(floor))} (20 [S2] C3)")
    for variant in variants:
        for station in sorted({int(g["station"]) for g in groups if str(g["variant"]) == variant}):
            sub = [g for g in groups if str(g["variant"]) == variant and int(g["station"]) == station]
            ax_b.scatter([int(g["n_low"]) for g in sub], [float(g["err_median_median"]) for g in sub],
                         s=26, color=color_of[variant], marker="s" if station == 0 else "v",
                         alpha=0.85, zorder=4, label=f"station {station} {variant} err_median")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(ns[0] / 2.0, ns[-1] * 2.0)
    ax_b.set_xlabel("N (subsample particles)")
    ax_b.set_ylabel("M1 relative RMSE")
    ax_b.set_title("(b) E0 reference: raw M1 vs N, with network err_median", fontsize=10)
    ax_b.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax_b.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("line2 Phase 1 applicability-domain map (40 [S4] C1)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- 报告
#: 门禁段标题（报告 5.x 小节名；判定表达式见 :data:`_GATE_RULES`）
_GATE_TITLES = {
    "H1": "放大存在性", "H3": "联合优于独立", "H7": "跨站泛化", "H8": "细结构可恢复性",
}


def _flatten_snapshot_rows(rows: Sequence[tuple[str, object]]) -> list[tuple[str, object]]:
    """H1/H3 的 ``inputs``（变体 → 种子 → N_eff）与 ``pairing`` 摊平成"项/值"两列。

    摊平后每个数值单独占一行——[S4] C2 的快照要能被逐项复核，嵌套字典挤在一格里做不到。
    """
    flat: list[tuple[str, object]] = []
    for key, value in rows:
        if key == "inputs" and isinstance(value, Mapping):
            for arm, per_seed in sorted(value.items(), key=lambda kv: str(kv[0])):
                if isinstance(per_seed, Mapping):
                    flat.extend(
                        (f"inputs.{arm}[{seed}]", n_eff)
                        for seed, n_eff in sorted(per_seed.items(), key=lambda kv: str(kv[0]))
                    )
                else:
                    flat.append((f"inputs.{arm}", per_seed))
        elif key == "pairing" and isinstance(value, Mapping):
            flat.append(("pairing.basis", value.get("basis")))
            flat.append(("pairing.pairs", value.get("pairs")))
        elif key == "n_eff_flags" and isinstance(value, Mapping):
            # H3 披露 `tied_at_domain_edge` 的证据：成对 N_eff 各自的落盘标记，逐臂逐种子摊平
            for arm, per_seed in sorted(value.items(), key=lambda kv: str(kv[0])):
                flat.extend(
                    (f"n_eff_flags.{arm}[{seed}]", flag)
                    for seed, flag in sorted(per_seed.items(), key=lambda kv: str(kv[0]))
                )
        else:
            flat.append((key, value))
    return flat


def _gate_lines(name: str, gate: Mapping) -> list[str]:
    """单条门禁的报告块：结论 + 冻结判定表达式 + 输入数值快照（`40` [S4] C2）。"""
    source = f"e1_gates.json:{name}"
    _require_keys(gate, _GATE_SNAPSHOT_KEYS[name], source,
                  f"`40` [S4] C2 判定快照键 {_GATE_SNAPSHOT_KEYS[name]}")
    lines = [
        f"### 5.{list(_GATE_SNAPSHOT_KEYS).index(name) + 1} {name}（{_GATE_TITLES[name]}）", "",
        f"- **结论**：{gate['verdict']}",
        f"- **判定表达式**（`40` [S3]，阈值冻结）：{_GATE_RULES[name]}",
        "", "**判定表达式的输入数值快照**（取自门禁产物，本脚本不重算判定）：", "",
    ]
    # per_seed / per_observable 另起明细子表，不挤进"项/值"两列
    skip = {"H7": ("per_seed", "flagged_seeds"), "H8": ("per_observable",)}.get(name, ())
    # H7 聚合条目的中位数比值以 `rho_median` 落盘（与冻结快照键 `rho` 同值）：同一数字只渲染
    # 一行，取聚合语义名；标记清单另在下方逐种子小节披露。聚合的降级原因（如
    # `n_eff_flag_present`）经 `_GATE_OPTIONAL_KEYS` 照旧进快照表。
    alias = {"rho": "rho_median"} if name == "H7" and "rho_median" in gate else {}
    keys = [alias.get(key, key) for key in (*_GATE_SNAPSHOT_KEYS[name], *_GATE_OPTIONAL_KEYS[name])]
    rows = [
        (key, gate[key])
        for key in keys
        if key != "verdict" and key not in skip and key in gate
    ]
    lines += _md_table(["项", "值"], _flatten_snapshot_rows(rows) if name in ("H1", "H3") else rows)

    if name == "H3" and gate.get("tied_at_domain_edge"):
        # 平局计入胜是 `40` [S3] 的冻结判据，故结论必须照旧输出；但域端点上的平局是被截断的
        # 饱和值（`20` [S3] 上溢 = 达到可验证域顶、真值不可辨识），不具备区分力——这句是
        # 事实披露，不改判定。
        lines += [
            "",
            f"- **域顶平局提示**：{_fmt(gate.get('n_ties'))}/{_fmt(gate.get('n_pairs'))} 对为"
            "域端点平局（两侧 N_eff 均携带上/下溢标记，见上表 `n_eff_flags` 与 3.1 的 N_eff "
            "表），该档**不具备区分力**；`40` [S3] H3 把平局计入胜，故上面的结论是**字面结论**，"
            f"不表示联合严格更优（严格更优仅 {_fmt(gate.get('n_strict_joint_wins'))} 对）。",
        ]
    if name == "H7":
        per_seed = gate["per_seed"]
        lines += ["", "逐种子跨站比值：", ""]
        lines += (
            _md_table(["seed", "rho", "verdict", "reason_code", "flag(站0)", "flag(站1)",
                       "N_eff(站0)", "N_eff(站1)"],
                      [[seed, item.get("rho"), item.get("verdict"), item.get("reason_code"),
                        item.get("flag_station0"), item.get("flag_station1"),
                        item.get("n_eff_station0"), item.get("n_eff_station1")]
                       for seed, item in sorted(per_seed.items(), key=lambda kv: str(kv[0]))])
            if per_seed else
            ["- 无：`per_seed` 为空（无站 1 的 test 评估，降级原因见上表 `reason_code`）。"]
        )
        flagged = gate.get("flagged_seeds")
        if flagged is None:  # 兼容未落 `flagged_seeds` 的旧门禁产物：按逐种子条目还原
            flagged = {
                seed: {"station0": item.get("flag_station0"),
                       "station1": item.get("flag_station1")}
                for seed, item in sorted(per_seed.items(), key=lambda kv: str(kv[0]))
                if item.get("flag_station0") or item.get("flag_station1")
            }
        lines += [
            "",
            f"- 上/下溢导致的不可判定：{len(flagged)}/{len(per_seed)} 个种子任一侧携带标记"
            f"（{_fmt(flagged) if flagged else '无'}）——该侧的 N_eff 是 `20` [S3] 报告域端点"
            "（1e3 / 1e5）的截断值，其 ρ 不构成跨站泛化的结论依据（`40` [S2]）；"
            "聚合条目因此同样判不可判定（`rho_median` 仅作披露）。",
        ]
    elif name == "H8":
        per_observable = gate["per_observable"]
        lines += ["", "逐观测量达标明细（13 项，`20` [S4]）：", ""]
        lines += (
            _md_table(["index", "name", "n_pass", "n_beams", "pass_rate"],
                      [[o["index"], o["name"], o["n_pass"], o["n_beams"], o["pass_rate"]]
                       for o in per_observable])
            if per_observable else
            ["- 无：`per_observable` 为空（达标率未计算，降级原因见上表 `reason_code`）。"]
        )
    lines.append("")
    return lines


def _curve_lines(curve: Mapping, curve_meta: Mapping | None) -> list[str]:
    """第 1 章的数据段：E0 曲线溯源 + 五档中位数/IQR + 两条地板。"""
    if not curve.get("points"):
        raise ValueError("e0_curve.json：points 缺失或为空（`20` [S2] 曲线点无来源）")
    floor = curve.get("reference_noise_floor")
    if floor is None or not math.isfinite(float(floor)):
        raise ValueError(
            f"e0_curve.json：reference_noise_floor={floor!r} 非有限——"
            "`40` [S4] C1 要求产出物含参照噪声地板（`20` [S2] C3）"
        )
    lines = ["### 1.1 E0 裸误差–N 曲线（`20` [S2]，N_eff 的唯一插值基底）", ""]
    provenance = [
        ("曲线路径", curve.get("source_path") or _display_path(Path(CURVE_FILENAME))),
        ("档位 tiers", curve.get("tiers")),
        ("每档子样本数 k", curve.get("k")),
        ("束数", len(curve.get("beam_ids", []))),
        ("束范围 beams_scope", (curve_meta or {}).get("beams_scope")),
        ("E0 started_at", (curve_meta or {}).get("started_at")),
        ("E0 completed_at", (curve_meta or {}).get("completed_at")),
        ("E0 data_version", (curve_meta or {}).get("data_version")),
        ("E0 code_version", (curve_meta or {}).get("code_version")),
        ("E0 spec_version", (curve_meta or {}).get("spec_version")),
        ("参照噪声地板（`20` [S2] C3）", floor),
    ]
    lines += _md_table(["项", "值"], provenance)
    lines += ["", "五档曲线（跨束 M1 中位数与 IQR；互差地板 = 同档两两子样本 M1 差中位数）：", ""]
    mutual = curve.get("mutual_floor", {})
    lines += _md_table(
        ["N", "M1 中位数", "IQR 下", "IQR 上", "n_samples", "互差地板"],
        [[p["n_low"], p["median"], p["iqr_lo"], p["iqr_hi"], p.get("n_samples"),
          mutual.get(str(p["n_low"]))] for p in curve["points"]],
    )
    lines.append("")
    return lines


def render_report(
    *,
    table: Sequence[Mapping],
    groups: Sequence[Mapping],
    curve: Mapping,
    curve_meta: Mapping | None,
    gates: Mapping,
    run_metas: Mapping[str, Mapping],
    e0_order: Mapping,
    figure_path: str | Path,
    table_path: str | Path,
) -> str:
    """实验报告全文（`40` [S4]：数据/设置/结果/门禁结论/第 7 章设计理由）。

    纯函数：不含墙钟时间与运行时随机量，同输入两次调用逐字符一致（[S4] C1）——
    报告头因此写"输入产物"而不写"生成时间"。
    """
    lines: list[str] = [
        "# line2 Phase 1 实验报告（方法可用域地图 + N_eff 表 + 门禁结论）",
        "",
        "> Spec：`docs/specs/line2/40_experiment_spec.md`（Frozen v1.0）[S4] 产出物；"
        "判据与 N_eff 见 `20`，模型与训练见 `30`，总纲与全局约束见 `00`。",
        "> 生成方式：`python -m src.line2.experiments.map_report --results-dir <dir>`"
        "（只读；同输入逐位一致，[S4] C1）。",
        f"> 输入产物：`e0_curve.json`、`{GATES_FILENAME}`、{len(table)} 份 `*/{EVAL_GLOB}`"
        f"、{len(run_metas)} 份 `*/{RUN_META_FILENAME}`。",
        f"> 代码版本：{_fmt(gates.get('code_version'))}｜数据版本："
        f"{_fmt((curve_meta or {}).get('data_version'))}｜spec 版本："
        f"{_fmt((curve_meta or {}).get('spec_version'))}（`00` GC7）",
        f"> 配套产出物：地图 `{_display_path(figure_path)}`、N_eff 表 `{_display_path(table_path)}`。",
        "",
        "## 1. 数据",
        "",
    ]
    lines += _curve_lines(curve, curve_meta)
    lines += ["### 1.2 GC7 三元组（run 级溯源）", ""]
    if run_metas:
        lines += _md_table(
            ["run_id", "data_version", "code_version", "spec_version", "variant", "seed",
             "n_low", "max_steps"],
            [[run_id, m["data_version"], m["code_version"], m["spec_version"], m["variant"],
              m["seed"], m["n_low"], m["max_steps"]] for run_id, m in sorted(run_metas.items())],
        )
    else:
        lines += [f"无任一 `{RUN_META_FILENAME}`——run 级三元组不可核验（见第 4 章）。"]
    lines.append("")

    lines += ["## 2. 设置", "", "### 2.1 门禁聚合维度（`40` [S3] 预注册）", ""]
    gate_dims = gates.get("gate_dims")
    if gate_dims:
        lines += _md_table(["项", "值"], sorted(gate_dims.items(), key=lambda kv: str(kv[0])))
    else:
        lines += [f"`{GATES_FILENAME}` 无 `gate_dims` 段——聚合维度未随门禁落盘（见第 4 章）。"]
    lines += [
        "",
        "### 2.2 变体路由（`40` [S1] C4、[S3]；只消费 val 段）",
        "",
        f"- 获胜变体：**{gates['winner_variant']}**",
        f"- 路由依据：{_fmt(gates.get('routing_basis'))}",
        "",
    ]
    val_segment = gates.get("val_segment")
    if val_segment:
        lines += _md_table(
            ["项", "值"],
            [["source", val_segment.get("source")], ["consumed_by", val_segment.get("consumed_by")],
             ["n_pairs", val_segment.get("n_pairs")]]
            + [[f"val N_eff[{variant}][{seed}]", value]
               for variant, per_seed in sorted((val_segment.get("per_seed_n_eff") or {}).items())
               for seed, value in sorted(per_seed.items(), key=lambda kv: str(kv[0]))],
        )
    else:
        lines += [f"`{GATES_FILENAME}` 无 `val_segment` 段——val 段消费面未落盘（见第 4 章）。"]
    lines += ["", "### 2.3 尺子先行审计（`40` [S1] C5）", ""]
    if e0_order["verifiable"]:
        lines += [
            f"- E0 completed_at：{_fmt(e0_order['e0_completed_at'])}",
            f"- 已核验 run 数：{len(e0_order['checked'])}（全部 started_at 晚于 E0 completed_at）",
            "",
        ]
        lines += _md_table(["run_id", "started_at"], sorted(e0_order["checked"].items()))
    else:
        lines += [
            "- **不可验证**：顺序证据不足，本节不据此判定通过。缺失项：",
            *[f"  - {item}" for item in e0_order["missing"]],
        ]
    lines += ["", "### 2.4 run 清单（test 段评估，`00` GC3：只评一次）", ""]
    lines += _md_table(
        ["run_id", "variant", "seed", "station", "n_low", "split", "n_beams"],
        [[r["run_id"], r["variant"], r["seed"], r["station"], r["n_low"], r["split"], r["n_beams"]]
         for r in table],
    )
    lines.append("")

    lines += ["## 3. 结果", "", "### 3.1 N_eff 明细表", "",
              f"落盘：`{_display_path(table_path)}`（列序 = {', '.join(TABLE_COLUMNS)}）。", ""]
    lines += _md_table(list(TABLE_COLUMNS), [[r[col] for col in TABLE_COLUMNS] for r in table])
    lines += ["", "### 3.2 主曲线三点（方法可用域地图面板 A 的数值）", "",
              "放大比 = N_eff / N_low；N_eff 取 `20` [S3] 报告域 [1e3, 1e5]，"
              "上/下溢行以域端点计入并在表中计数（图内以空心叉号标注）。", ""]
    lines += _md_table(
        ["station", "variant", "n_low", "n_runs", "ratio 中位数", "ratio [min, max]",
         "上溢数", "下溢数", "err_median 中位数"],
        [[g["station"], g["variant"], g["n_low"], g["n_runs"], g["ratio_median"],
          f"[{_fmt(g['ratio_min'])}, {_fmt(g['ratio_max'])}]", g["n_overflow"], g["n_underflow"],
          g["err_median_median"]] for g in groups],
    )
    lines += ["", "### 3.3 跨站泛化（`40` [S2]，站 1 的 test 束）", ""]
    station1 = [r for r in table if int(r["station"]) == 1]
    if station1:
        lines += _md_table(
            ["run_id", "variant", "seed", "n_low", "n_eff", "n_eff_flag", "err_median"],
            [[r["run_id"], r["variant"], r["seed"], r["n_low"], r["n_eff"], r["n_eff_flag"],
              r["err_median"]] for r in station1],
        )
    else:
        lines += ["N_eff 表内无站 1 行——E3 跨站评估尚未落盘（结论见第 5 章 H7）。"]
    lines.append("")

    lines += ["## 4. 失败模式与不可验证项", ""]
    gaps: list[str] = []
    if curve_meta is None:
        gaps.append(
            f"`{E0_META_FILENAME}` 缺失——第 1.1 节的 E0 束范围、时间戳与三元组无来源。"
        )
    if not run_metas:
        gaps.append(f"无任一 `{RUN_META_FILENAME}`——run 级 GC7 三元组与启动时间戳不可核验。")
    missing_by_file: dict[str, list[str]] = {}
    for run_id, meta in sorted(run_metas.items()):
        for name in meta["missing"]:
            missing_by_file.setdefault(name, []).append(run_id)
    for name, run_ids in sorted(missing_by_file.items()):
        gaps.append(
            f"{len(run_ids)}/{len(run_metas)} 个 run 目录缺 `{name}`（`40` [S1] C2 契约文件）："
            + ", ".join(run_ids)
        )
    if not e0_order["verifiable"]:
        gaps.append(
            "顺序审计（`40` [S1] C5）判**不可验证**（不据此判通过），缺失证据："
            + "；".join(e0_order["missing"])
        )
    for key in GATES_OPTIONAL_SECTIONS:
        if key not in gates:
            gaps.append(f"`{GATES_FILENAME}` 无 `{key}` 段——对应溯源未随门禁落盘。")
    if beam_snapshot(gates) is None:
        gaps.append(f"`{GATES_FILENAME}` 无束级 N_eff 快照段（{GATES_SNAPSHOT_KEYS} 任一）。")
    for name, gate in (("H1", gates["H1"]), ("H3", gates["H3"]),
                       ("H7", cross_station_gate(gates)), ("H8", gates["H8"])):
        if gate.get("reason_code"):
            gaps.append(f"{name}：{gate['reason_code']}（判定降级为 {gate.get('verdict')}）。")
    flagged = [r for r in table if r["n_eff_flag"]]
    if flagged:
        gaps.append(
            f"{len(flagged)} 行 N_eff 触及报告域端点（上溢 "
            f"{sum(1 for r in flagged if r['n_eff_flag'] == 'overflow')} / 下溢 "
            f"{sum(1 for r in flagged if r['n_eff_flag'] == 'underflow')}）——"
            "放大比在端点处为截断值，只声明达到域顶/域底，不可外推（`20` [S3] C2）。"
        )
    lines += [f"- {gap}" for gap in gaps] if gaps else [
        "- 无：必需输入齐备，门禁段无条件降级项，run 目录契约文件无缺失。"
    ]
    lines.append("")

    lines += ["## 5. 门禁结论（每条附判定表达式的输入数值快照，`40` [S4] C2）", ""]
    lines += [f"- 获胜变体：**{gates['winner_variant']}**（路由依据 "
              f"{_fmt(gates.get('routing_basis'))}；`40` [S1] C4）", ""]
    for name in ("H1", "H3", "H7", "H8"):
        gate = cross_station_gate(gates) if name == "H7" else gates[name]
        lines += _gate_lines(name, gate)
    snapshot = beam_snapshot(gates)
    if snapshot:
        lines += ["### 5.5 束级 N_eff 快照（门禁产物的中间量）", ""]
        for variant, rows in sorted(snapshot.items()):
            lines += [f"- **{variant}**", ""]
            lines += _md_table(
                ["beam_id", "n_eff_median", "n_eff_per_seed", "flag_per_seed"],
                [[r.get("beam_id"), r.get("n_eff_median"), r.get("n_eff_per_seed"),
                  r.get("flag_per_seed")] for r in rows],
            )
            lines.append("")

    lines += [
        "## 6. 结论范围与后续",
        "",
        "- 结论范围限于 `00` [S1] 的推广域声明：束群 = 锚点经预注册抖动线性传输，"
        "Phase 1 的方法可用域只在该邻域内成立；真实前端晶格的跨域泛化属 Phase 2。",
        "- 放大比只在可验证域 [1e3, 1e5] 内有定义：N_eff 触顶（上溢）意味着"
        "\"达到 1e4 工况的 10× 可验证上限\"，不意味着更大的真实放大；"
        "触顶区间受参照噪声地板限制不可分辨（`20` [S2]）。",
        "- 本报告只陈述门禁的机械判定结果，不预设研究方向（`00` GC6）；"
        "H1 不成立时按 `00` [S3] 升级用户重议 Phase 2 方向。",
        "- 后续：Phase 2 入口为 L9 束流接口，进入条件 = Phase 1 交付完成 + 用户放行（`00` [S3]）。",
        "",
        "## 7. 设计理由（Why）",
        "",
        "### 7.1 为什么三类产出物由一个只读脚本生成，而不挂在门禁脚本里",
        "",
        "- 选择：`map_report` 单独成模块，只消费 `gates.py`/`evaluate.py`/`e0.py` 的落盘产物。",
        "- 备选：把图表与报告塞进 `evaluate_gates_from_disk`——被否，门禁的职责是判定"
        "（`40` [S3] C1 要求同输入逐位一致的可复算判定），产出物是判定的呈现层；"
        "混在一起会让\"改报告格式\"变成\"改门禁\"，也让 [S3] C2 的\"只消费落盘 summary\"边界模糊。",
        "- 取舍依据：[S4] Global Constraints 要求产出物脚本只读；分层后门禁可独立复算，"
        "报告可反复重生成而不触碰判定。",
        "",
        "### 7.2 为什么报告不写生成时间戳与当次 git HEAD",
        "",
        "- 选择：报告头只写输入产物的溯源（E0 时间戳、门禁产物里的 code_version）。",
        "- 备选：沿 line1 stage_report 头部写\"生成时间 + 当前 commit\"——被否，"
        "[S4] C1 要求同输入逐位一致，墙钟时间与生成时的 HEAD 会让两次运行不可比对，"
        "也无法用文件哈希证明\"报告未被手工改过\"。",
        "- 取舍依据：可复现性优先于\"报告看起来很新\"；需要生成时刻时看文件 mtime 即可。",
        "",
        "### 7.3 为什么地图分两个面板，而不是共用一条 x 轴的双 y 轴",
        "",
        "- 选择：面板 A 画 N_eff/N_low vs N_low（无量纲放大比），面板 B 画裸 M1 vs N"
        "（相对 RMSE）+ 参照噪声地板 + 网络 err_median 落点。",
        "- 备选 1：单轴同画两条线——被否，y 量纲不同（比值 vs 相对 RMSE），"
        "同轴等于宣布二者可比。备选 2：共享 x 的 twinx 双 y 轴——被否，"
        "两线的视觉交点由轴缩放任意决定，读者会把交点读成\"网络达到裸曲线水平\"这类无依据结论。",
        "- 取舍依据：[S4] 只要求地图含主曲线、E0 参考线与噪声地板线，未要求同轴；"
        "分面板后 [S4] C1 的三条要素都在，且噪声地板画在它被定义的 M1 面板内"
        "（`20` [S2] C3 的量纲）。面板 B 叠加网络 err_median 是为了让 E0 参考线"
        "真正起\"参照\"作用——否则它与本批 run 无视觉关联。",
        "",
        "### 7.4 为什么主曲线画比值 N_eff/N_low，且上/下溢用域端点显式标注",
        "",
        "- 选择：y = N_eff/N_low（对数），三档 N_low ∈ {3e3, 1e4, 1e5} 直接可比；"
        "带标记的行画在域端点（1e3 或 1e5 对应的比值）上并用空心叉号区分。",
        "- 备选：y = N_eff 绝对值——被否，3e3 档的 N_eff = 1e5 与 1e4 档的 N_eff = 1e5"
        "语义强度不同（前者 33×、后者 10×），绝对值会把\"可用域\"读平。",
        "- 取舍依据：`20` [S3] C2 规定 N_eff 要么落在 [1e3, 1e5] 内、要么携带显式标记，"
        "端点值是截断而非实测；不标注就会把域顶读成实测放大（`00` [S1]"
        "\"可验证放大上限 = 10×\"的语义边界）。",
        "",
        "### 7.5 为什么 N_eff 表按 run 级聚合，束级明细留在门禁产物里",
        "",
        "- 选择：CSV 一行 = 一个 (run, 站, N_low) 的 test 评估，含上/下溢标记。",
        "- 备选：一行 = 一个 (run, 束)——被否，150 束 × 20 run 会让 [S4] 要求的"
        "\"runs × 站 × N_low 明细表\"淹没在束级行里；束级中位数与逐种子值已由门禁产物"
        "`beam_n_eff_snapshot` 落盘，报告第 5.5 节原样引用，不做第二份实现。",
        "- 取舍依据：单一实现点（`20` Global Constraints 的\"禁止旁路实现\"同理适用于产物层）。",
        "",
        "### 7.6 为什么顺序审计（`40` [S1] C5）在证据缺失时判\"不可验证\"而不判通过",
        "",
        "- 选择：证据齐全且存在 run 的 started_at ≤ E0 completed_at → 抛错；"
        "证据缺失 → verifiable=False，第 4 章显式登记。",
        "- 备选：证据缺失时默认通过——被否，那等于把\"没记录\"读成\"守规矩\"，"
        "与 D8\"顺序约束不编码就会在执行中被跳过\"的立意相反。",
        "- 取舍依据：与门禁的\"不可判定\"一等地位一致（D4）：缺证据是一种结论，"
        "不是通过。违例则必须中断——尺子后行的批次其 N_eff 无定义基底，"
        "继续生成报告等于把无效矩阵包装成交付物。",
        "",
        "### 7.7 为什么图内标签用英文而报告正文用中文",
        "",
        "- 选择：`figure_avail_domain.png` 的标题、轴标签与图例用 ASCII 英文；"
        "报告正文、CSV 列名与代码 docstring 用中文。",
        "- 备选：图内也用中文——被否，matplotlib 的中文字形依赖运行机上装了什么 CJK 字体，"
        "缺字体时字形降级为方框、且不同字体下渲染出的像素不同，"
        "直接破坏 [S4] C1 的\"同输入逐位一致\"（同一份产物在两台机器上哈希不同）。",
        "- 取舍依据：与 line1 出图惯例一致（`src/evaluation/plots.py` 的图内标签同为英文），"
        "并让 PNG 只依赖 matplotlib 自带的 DejaVu Sans。",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- 装配
def generate(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> dict:
    """一键生成三类产出物（`40` [S4] C1）。返回三份产物路径与行数统计。

    ``out_dir`` 默认 = ``results_dir``（只托管 N_eff 表）；``reports_dir`` 默认 =
    ``<results_dir>/../reports``（`00` [S2] 研究线布局）；``figure_dir`` 默认 = ``reports_dir``
    （图与报告同为交付物、同目录入库）。除这三个路径外不写任何文件。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    out_dir = Path(out_dir) if out_dir is not None else results_dir
    reports_dir = Path(reports_dir) if reports_dir is not None else results_dir.parent / "reports"
    figure_dir = Path(figure_dir) if figure_dir is not None else reports_dir

    curve_path = results_dir / CURVE_FILENAME
    curve = _read_json(curve_path, "E0 裸误差–N 曲线 dump（`20` [S2]、`40` [S1] C5 尺子先行）")
    curve["source_path"] = _display_path(curve_path)
    curve_meta = _read_json_optional(results_dir / E0_META_FILENAME)
    gates = load_gate_results(results_dir)
    table = collect_n_eff_table(results_dir)
    groups = amplify_groups(table)
    run_metas = collect_run_metas(results_dir)
    e0_order = check_e0_precedes_runs(results_dir, run_metas=run_metas)

    figure_path = render_avail_domain(groups, curve, figure_dir / FIGURE_NAME)
    table_path = write_n_eff_table(table, out_dir / TABLE_NAME)
    report_text = render_report(
        table=table, groups=groups, curve=curve, curve_meta=curve_meta, gates=gates,
        run_metas=run_metas, e0_order=e0_order, figure_path=figure_path, table_path=table_path,
    )
    report_path = reports_dir / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return {
        "figure": figure_path,
        "table": table_path,
        "report": report_path,
        "n_rows": len(table),
        "n_runs": len({str(r["run_id"]) for r in table}),
        "e0_order_verifiable": bool(e0_order["verifiable"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.line2.experiments.map_report",
        description="line2 Phase 1 三类产出物（spec 40 [S4]；只读落盘产物）",
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="results 目录（含 e0_curve.json / e1_gates.json / <run_id>/）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"N_eff 表输出目录（默认 = --results-dir；只落 {TABLE_NAME}）")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help=f"报告输出目录（默认 = <results-dir>/../reports；落 {REPORT_NAME}）")
    parser.add_argument("--figure-dir", type=Path, default=None,
                        help=f"地图输出目录（默认 = --reports-dir；落 {FIGURE_NAME}）")
    args = parser.parse_args(argv)
    out = generate(args.results_dir, out_dir=args.out_dir, reports_dir=args.reports_dir,
                   figure_dir=args.figure_dir)
    print(
        json.dumps(
            {
                "figure": _display_path(out["figure"]),
                "table": _display_path(out["table"]),
                "report": _display_path(out["report"]),
                "n_rows": out["n_rows"],
                "n_runs": out["n_runs"],
                "e0_order_verifiable": out["e0_order_verifiable"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
