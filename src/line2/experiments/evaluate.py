"""L4/L5 test 端评估：ckpt + 数据集 → N_eff 与 13 维观测向量（spec line2/40 [S3]）。

契约（`40` [S3] C2、`20` [S3]/[S4]）：
- 只消费落盘 ckpt 与数据集，产 `results/<run_id>/test_eval.json`——字段冻结，
  门禁（`gates.evaluate_gates_from_disk`）与地图脚本按此消费；
- 前向路径与 `train.py::_forward_variant` 逐分支一致（joint 用 κ、independent 用
  edges + log₁₀N），κ 的 log₁₀N 分量由 counts 派生（`30` [S1] C5 防泄露）；
- 确定性（`40` [S3] C1）：eval 模式 + no_grad + 无随机算子，同输入逐位一致；
- N_eff = `20` [S3] 的 log-N 插值，报告域 [1e3, 1e5]，上/下溢携带显式标记；
- 聚合次序 = 先对子样本取中位数（err / 13 维观测量各自取中位数），再映射 N_eff。

用法：
    python -m src.line2.experiments.evaluate --ckpt <run_dir>/best_val.ckpt \
        --data-dir <数据集目录> --split test --station 0 --n-low 10000 \
        --subs 0-19 --curve <results/e0_curve.json> [--out <json>] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.line2.data_pipeline.dataset import git_head
from src.line2.metrics.metrics import group_rel_rmse
from src.line2.metrics.n_eff import CurvePoint, CurveResult, n_eff_from_err
from src.line2.metrics.tolerance import observable_vector
from src.line2.training.train import build_model
from src.line2.training.unet import condition_vector, normalize_input_group

__all__ = [
    "LoadedCurve",
    "curve_from_json",
    "load_model",
    "predict_group",
    "evaluate_run",
    "parse_subs",
    "EVAL_FILENAME",
    "main",
]

EVAL_FILENAME = "test_eval.json"


@dataclass
class LoadedCurve(CurveResult):
    """:class:`CurveResult` + 落盘溯源（曲线路径、束范围、档位）。

    `n_eff_from_err` 只消费 ``points``，故本类可直接替代 CurveResult 使用；
    额外字段供 `test_eval.json` 的 ``curve`` 段做溯源（`40` [S3] C2）。
    """

    source_path: str | None = None
    beams_scope: str | None = None
    beam_ids: tuple[int, ...] = field(default_factory=tuple)
    tiers: tuple[int, ...] = field(default_factory=tuple)
    k: int | None = None


def curve_from_json(path: str | Path) -> CurveResult:
    """读回 `e0.error_n_curve` 的 dump（含同目录 `.meta.json` 的束范围）。

    dump 字段 = points / mutual_floor / reference_noise_floor（+ raw_samples 等）；
    ``beams_scope`` 只存在于 meta 侧文件，缺失时留 None 由调用方按 manifest 推断。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"曲线 dump 不存在：{path}")
    dump = json.loads(path.read_text(encoding="utf-8"))
    if "points" not in dump:
        raise ValueError(f"{path}：缺 points 字段，不是 e0 曲线 dump")
    points = [
        CurvePoint(
            n_low=int(p["n_low"]),
            median=float(p["median"]),
            iqr_lo=float(p["iqr_lo"]),
            iqr_hi=float(p["iqr_hi"]),
            n_samples=int(p["n_samples"]),
        )
        for p in dump["points"]
    ]
    meta_path = path.with_suffix(".meta.json")
    scope = None
    if meta_path.is_file():
        scope = json.loads(meta_path.read_text(encoding="utf-8")).get("beams_scope")
    return LoadedCurve(
        points=points,
        mutual_floor={int(t): float(v) for t, v in dump.get("mutual_floor", {}).items()},
        reference_noise_floor=float(dump.get("reference_noise_floor", float("nan"))),
        raw_samples={int(t): [float(v) for v in vals] for t, vals in dump.get("raw_samples", {}).items()},
        source_path=str(path),
        beams_scope=scope,
        beam_ids=tuple(int(b) for b in dump.get("beam_ids", ())),
        tiers=tuple(int(t) for t in dump.get("tiers", ())),
        k=dump.get("k"),
    )


def _curve_scope(curve: CurveResult, manifest: dict) -> str | None:
    """曲线束范围：优先 meta 侧文件，否则由 dump 的 beam_ids 与 manifest 切分比对。"""
    scope = getattr(curve, "beams_scope", None)
    if scope:
        return str(scope)
    beam_ids = set(getattr(curve, "beam_ids", ()) or ())
    if not beam_ids:
        return None
    splits = {name: set(int(b) for b in ids) for name, ids in manifest["splits"].items()}
    for name, combo in (
        ("val", splits.get("val", set())),
        ("test", splits.get("test", set())),
        ("train", splits.get("train", set())),
        ("valtest", splits.get("val", set()) | splits.get("test", set())),
    ):
        if beam_ids == combo:
            return name
    return "custom"


def load_model(ckpt: Path) -> tuple[nn.Module, dict]:
    """由 ckpt 复原 eval 模式模型与其 config（结构分支复用 `train.build_model`）。"""
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt}")
    # config 段可含任意 Python 对象（路径/嵌套字典），weights_only 白名单会拒绝
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"{ckpt}：缺 state_dict，不是本线训练产出的 checkpoint")
    cfg = dict(payload.get("config") or {})
    model = build_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def predict_group(
    model: nn.Module,
    cfg: dict,
    counts: np.ndarray,
    edges: np.ndarray,
    n_low: int | float,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """单图组推理：counts (6,128,128) + edges (4,129) → 概率图 (6,128,128) float64。

    变体分支与 `train.py::_forward_variant` 一致；``n_low`` 为调用方声明的档位
    （记账用），κ 的 log₁₀N 分量与训练同源——由 counts 派生而非取声明值，
    以保证 test 端前向与训练前向逐位对齐（`30` [S1] C5）。
    """
    variant = str(cfg.get("variant", "joint"))
    dev = torch.device(device)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.float32).unsqueeze(0).to(dev)
    edges_t = torch.as_tensor(np.asarray(edges), dtype=torch.float32).unsqueeze(0).to(dev)
    was_training = model.training
    model.to(dev).eval()
    try:
        with torch.no_grad():
            shapes, _totals = normalize_input_group(counts_t)
            if variant == "joint":
                prob = model(shapes, condition_vector(counts_t, edges_t))
            elif variant == "independent":
                n_low_vec = counts_t.sum(dim=(1, 2, 3)).clamp_min(1.0)
                prob = model(shapes, edges_t, torch.log10(n_low_vec))
            else:
                raise ValueError(f"未知 variant：{variant!r}（应 ∈ joint/independent）")
            out = prob[0].detach().to(torch.float64).cpu().numpy()
    finally:
        model.train(was_training)
    if out.shape != (6, 128, 128):
        raise ValueError(f"概率图形状异常：{out.shape}（应 (6, 128, 128)）")
    return np.ascontiguousarray(out, dtype=np.float64)


def parse_subs(spec: str | int | Sequence[int] | None) -> list[int]:
    """子样本选择：``"a-b"`` 闭区间 / ``"a,b,c"`` 枚举 / 整数或整数序列 / None = 全 20 个。"""
    if spec is None:
        return list(range(20))
    if isinstance(spec, int):
        return [spec]
    if not isinstance(spec, str):
        return [int(s) for s in spec]
    text = spec.strip()
    if not text:
        return list(range(20))
    if "-" in text:
        lo, hi = (int(part) for part in text.split("-", 1))
        if hi < lo:
            raise ValueError(f"--subs 区间反向：{spec!r}")
        return list(range(lo, hi + 1))
    return sorted({int(part) for part in text.split(",")})


def evaluate_run(
    ckpt: str | Path,
    data_dir: str | Path,
    split: str,
    station: int,
    n_low: int,
    subs: Sequence[int] | int | str | None,
    curve: CurveResult | str | Path,
    device: str | torch.device = "cpu",
    out: str | Path | None = None,
) -> dict:
    """该 split 全部束 × 指定站 × 指定子样本的 test 端一次性评估（`40` [S3] test 段）。

    每束：逐子样本推理 → M1（`group_rel_rmse`）与 13 维观测量（`observable_vector`）
    → 各自对子样本取中位数 → 由中位数 err 映射 N_eff（`20` [S3]，含上/下溢标记）。
    结果落盘并原样返回（run_id = ckpt 父目录名）；默认落盘路径为
    ``<ckpt 父目录>/test_eval.json``。

    ``per_beam`` 的两个 N_eff 口径（`20` [S2] 曲线口径 = 每（束, 站, 档）对子样本取
    中位数后再映射）：
    - ``n_eff``：**门禁口径**——先对子样本 err 取中位数、再映射 N_eff；与裸曲线
      "档位中位数"同一聚合次序，出口门禁（`40` [S3] H1/H3/H8，含 H8 容差线最近档
      映射）只消费本字段；
    - ``n_eff_median``：先逐子样本映射、再取中位数，仅作快照/复核（N_eff 对 err 单调，
      二者只在偶数子样本或触及报告域端点时可分辨），不进任何判定表达式。

    ``curve`` 接受 E0 dump 路径或已载入的 :class:`CurveResult`。E3 跨站评估复用同一
    ckpt（`40` [S2]），故站 1 的评估 SHALL 传 ``out`` 另起文件名——否则与站 0 产物
    同名相覆，出口门禁会失去站 0 输入。
    """
    ckpt = Path(ckpt)
    data_dir = Path(data_dir)
    curve_obj = curve_from_json(curve) if isinstance(curve, (str, Path)) else curve
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if split not in manifest["splits"]:
        raise ValueError(f"{split!r} 不在 manifest 切分中：{sorted(manifest['splits'])}")
    beam_ids = [int(b) for b in manifest["splits"][split]]
    if not beam_ids:
        raise ValueError(f"切分 {split!r} 为空（n_beams 过小时 15% 取整为 0）")
    sub_list = parse_subs(subs)
    station = int(station)
    n_low = int(n_low)

    model, cfg = load_model(ckpt)
    if "seed" not in cfg:
        raise ValueError(f"{ckpt}：config 缺 seed，test_eval.json 的种子字段无来源")
    variant = str(cfg.get("variant", "joint"))
    per_beam: list[dict] = []
    with h5py.File(data_dir / f"{split}.h5", "r") as h5:
        for beam_id in beam_ids:
            stg = h5[f"beam_{beam_id:04d}"][f"station{station}"]
            ref = stg["target_full"][...].astype(np.float64)
            errs: list[float] = []
            n_effs: list[float] = []
            obs_rows: list[np.ndarray] = []
            for sub_idx in sub_list:
                name = f"tier_{n_low:06d}_sub_{sub_idx:02d}"
                if name not in stg:
                    raise KeyError(f"{split}.h5 beam_{beam_id:04d}/station{station} 缺 {name}")
                dset = stg[name]
                counts = dset[...]
                edges = np.asarray(dset.attrs["bin_edges"], dtype=np.float64)
                prob = predict_group(model, cfg, counts, edges, n_low, device)
                err = group_rel_rmse(prob, ref)
                errs.append(err)
                n_effs.append(n_eff_from_err(err, curve_obj)[0])
                obs_rows.append(observable_vector(prob, ref, edges))
            err_median = float(np.median(np.asarray(errs, dtype=np.float64)))
            beam_n_eff, beam_flag = n_eff_from_err(err_median, curve_obj)
            obs_median = np.median(np.stack(obs_rows), axis=0)
            per_beam.append(
                {
                    "beam_id": beam_id,
                    "err": err_median,
                    "n_eff": float(beam_n_eff),
                    "n_eff_flag": beam_flag,
                    "obs": [float(v) for v in obs_median],
                    # 快照口径：先逐子样本映射 N_eff、再取中位数（与上方"先中位数后
                    # 映射"的 n_eff 互为对照；门禁只消费 n_eff）
                    "n_eff_median": float(np.median(np.asarray(n_effs, dtype=np.float64))),
                }
            )

    err_median = float(np.median(np.asarray([b["err"] for b in per_beam], dtype=np.float64)))
    run_n_eff, run_flag = n_eff_from_err(err_median, curve_obj)
    payload = {
        "run_id": ckpt.parent.name,
        "variant": variant,
        "seed": int(cfg["seed"]),
        "split": split,
        "station": station,
        "n_low": n_low,
        "subs": sub_list,
        "curve": {
            "path": getattr(curve_obj, "source_path", None) or "",
            "beams_scope": _curve_scope(curve_obj, manifest),
            "reference_noise_floor": float(curve_obj.reference_noise_floor),
        },
        "n_beams": len(per_beam),
        "n_groups": len(per_beam) * len(sub_list),
        "err_median": err_median,
        "n_eff": float(run_n_eff),
        "n_eff_flag": run_flag,
        "per_beam": per_beam,
        "code_version": git_head(),
    }
    out_path = Path(out) if out is not None else ckpt.parent / EVAL_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 test 端评估（spec 40 [S3]）")
    parser.add_argument("--ckpt", type=Path, required=True, help="run 目录内的 ckpt（best_val.ckpt）")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", help="评估切分（出口门禁只消费 test）")
    parser.add_argument("--station", type=int, default=0)
    parser.add_argument("--n-low", type=int, default=10_000)
    parser.add_argument("--subs", default="0-19", help="子样本：a-b 闭区间 / a,b,c / 单个整数")
    parser.add_argument("--curve", type=Path, required=True, help="results/e0_curve.json（E0 dump）")
    parser.add_argument("--out", type=Path, default=None,
                        help="默认 <ckpt 父目录>/test_eval.json；跨站（E3）评估须另起名")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    out = args.out or args.ckpt.parent / EVAL_FILENAME
    payload = evaluate_run(
        args.ckpt,
        args.data_dir,
        args.split,
        args.station,
        args.n_low,
        args.subs,
        args.curve,
        device=args.device,
        out=out,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "variant": payload["variant"],
                "seed": payload["seed"],
                "split": payload["split"],
                "station": payload["station"],
                "n_low": payload["n_low"],
                "n_beams": payload["n_beams"],
                "n_groups": payload["n_groups"],
                "err_median": payload["err_median"],
                "n_eff": payload["n_eff"],
                "n_eff_flag": payload["n_eff_flag"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
