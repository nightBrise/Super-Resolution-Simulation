"""可视化模块（70 [S8]、60 [S15] 15.3）。

基础可视化三件套：
- 2D 相空间并排对比（对数色标 LogNorm，70 [S8] C1）；
- 1D 电流/能谱剖面曲线对比（70 [S8] C2）；
- 残差图（发散色标，红蓝区分多余/丢失粒子，70 [S8] C3）。

5 张核心图（90 [S3]，M6 最终报告用）：
- figure_01_2d_phasespace / figure_02_1d_profile / figure_05_residual_map（样本三件套，输入 predictions npz）；
- figure_03_physics_error_bar（物理量误差柱状图，输入 metrics.csv，误差棒为 bootstrap 95% CI）；
- figure_04_error_vs_gamma_scatter（误差 vs γ 散点图，输入 metrics.csv）。

输出格式 PNG，300 dpi（M4+ 用户指定，2026-08-28），文件命名 ``figure_0<N>_<type>.png``（90 [S3]）。

CLI：
``python -m src.evaluation.plots --mode intermediate --config <path> --predictions <npz> --out <dir>``
（中间可视化三件套，predictions npz 由 ``infer`` 输出，默认模式）；
``python -m src.evaluation.plots --mode core --config <path> --predictions <npz> --metrics <metrics.csv> --out <dir>``
（5 张核心图，80 [S8] C8：3 张来自 predictions，2 张来自 metrics.csv）。
色标类型（log / diverging）与配置文件中的可视化设置一致（70 [S8] C4）。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from src.evaluation.metrics import bootstrap_ci  # noqa: E402
from src.generators.f_beam import pixel_center_coordinates  # noqa: E402
from src.utils.config_utils import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 5 张核心图文件名（90 [S3]、60 [S15] 15.3；M6 对账用）。
CORE_FIGURES = {
    "figure_01_2d_phasespace": "2D 相空间并排对比（对数色标）",
    "figure_02_1d_profile": "1D 电流/能谱剖面对比",
    "figure_03_physics_error_bar": "物理量误差条形图",
    "figure_04_error_vs_gamma_scatter": "误差 vs γ 散点图",
    "figure_05_residual_map": "残差图（发散色标）",
}


def _coords() -> tuple[np.ndarray, np.ndarray]:
    coords, _ = pixel_center_coordinates(256)
    return np.meshgrid(coords, coords, indexing="ij")


def plot_2d_phasespace(
    panels: dict[str, np.ndarray],
    out_path: str | Path,
    log_scale: bool = True,
) -> Path:
    """2D 相空间并排对比图（70 [S8] C1：并排 L_up/P2、H、Ĥ_*，对数色标）。

    ``panels``：标题 → 图像（如 ``{"L_up": .., "H": .., "Ĥ_A": ..}``）。
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels.items()):
        img = np.asarray(img, dtype=np.float64)
        norm = LogNorm(vmin=max(img[img > 0].min(), 1e-8), vmax=img.max()) if log_scale else None
        im = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), norm=norm, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("δ")
        ax.set_ylabel("z")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save_figure(fig, out_path)


def plot_1d_profiles(
    profiles: dict[str, dict[str, np.ndarray]],
    out_path: str | Path,
) -> Path:
    """1D 剖面对比（70 [S8] C2：电流 I(z) 与能谱 S(δ) 预测 vs 真值）。"""
    coords, _ = pixel_center_coordinates(256)
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.0 * len(profiles), 4.2))
    if len(profiles) == 1:
        axes = [axes]
    for ax, (title, curves) in zip(axes, profiles.items()):
        for label, values in curves.items():
            ax.plot(coords, values, label=label, linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("z / δ")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, out_path)


def plot_residual_map(H: np.ndarray, H_hat: np.ndarray, out_path: str | Path) -> Path:
    """残差图（70 [S8] C3：发散色标，正负可区分——多余/丢失粒子）。"""
    Z, D = _coords()
    residual = np.asarray(H_hat, dtype=np.float64) / H_hat.sum() - np.asarray(H, dtype=np.float64) / H.sum()
    vmax = max(np.abs(residual).max(), 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(residual, origin="lower", extent=(-1, 1, -1, 1), norm=norm, cmap="RdBu_r")
    ax.set_title("Ĥ − H (residual)")
    ax.set_xlabel("δ")
    ax.set_ylabel("z")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save_figure(fig, out_path)


def _save_figure(fig, out_path: str | Path) -> Path:
    """PNG + PDF 成对保存（60 [S15] 15.3）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    return out_path


def _load_metrics_rows(metrics_csv: str | Path) -> list[dict[str, str]]:
    """读取 metrics.csv（长表，每行 = 方案 × 样本，80 [S8] 列名规范）。"""
    with open(metrics_csv, newline="") as fh:
        return list(csv.DictReader(fh))


def plot_physics_error_bar(
    metrics_csv: str | Path,
    out_path: str | Path,
    seed: int = 20260825,
) -> Path:
    """figure_03 物理量误差柱状图（90 [S3] Fig.3）。

    X 轴为物理指标（ε_z、I_peak），每方案一根柱；柱高为逐样本相对误差均值，
    误差棒为 bootstrap 95% CI（复用 70 [S7.2] bootstrap_ci，n_boot=1000，
    固定 seed 可复现，05 [S6] 随机纪律）。输入 metrics.csv 的
    ``e_eps_z`` / ``e_I_peak`` / ``scheme`` 列（80 [S8]）。
    """
    rows = _load_metrics_rows(metrics_csv)
    schemes = ["A", "B", "C"]
    metric_keys = ["e_eps_z", "e_I_peak"]
    metric_labels = ["ε_z", "I_peak"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    width = 0.25
    x = np.arange(len(metric_keys))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for j, scheme in enumerate(schemes):
        means: list[float] = []
        errs_lo: list[float] = []
        errs_hi: list[float] = []
        for key in metric_keys:
            d = np.array(
                [float(r[key]) for r in rows if r["scheme"] == scheme], dtype=np.float64
            )
            mean = float(d.mean()) if d.size else float("nan")
            lo, hi = bootstrap_ci(d, n_boot=1000, seed=seed) if d.size else (float("nan"), float("nan"))
            means.append(mean)
            errs_lo.append(mean - lo)
            errs_hi.append(hi - mean)
        ax.bar(
            x + (j - 1) * width,
            means,
            width,
            yerr=[errs_lo, errs_hi],
            capsize=3,
            label=f"Scheme {scheme}",
            color=colors[j],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Relative error")
    ax.legend()
    ax.set_title("Physics metric relative error (bootstrap 95% CI)")
    fig.tight_layout()
    return _save_figure(fig, out_path)


def plot_error_vs_gamma_scatter(metrics_csv: str | Path, out_path: str | Path) -> Path:
    """figure_04 误差 vs γ 散点图（90 [S3] Fig.4）。

    X 轴为 γ（c_high 精细结构参数），Y 轴为 ε_z 相对误差，颜色映射三方案；
    展示先验在何种参数区间开始失效。输入 metrics.csv 的
    ``gamma`` / ``e_eps_z`` / ``scheme`` 列（80 [S8] C4）。
    """
    rows = _load_metrics_rows(metrics_csv)
    schemes = ["A", "B", "C"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for j, scheme in enumerate(schemes):
        xs = [float(r["gamma"]) for r in rows if r["scheme"] == scheme]
        ys = [float(r["e_eps_z"]) for r in rows if r["scheme"] == scheme]
        ax.scatter(xs, ys, s=10, alpha=0.45, color=colors[j], label=f"Scheme {scheme}")
    ax.set_xlabel("γ (fine-structure parameter)")
    ax.set_ylabel("ε_z relative error")
    ax.legend(markerscale=2)
    ax.set_title("Error vs γ")
    fig.tight_layout()
    return _save_figure(fig, out_path)


def generate_core_figures(
    predictions_npz: str | Path,
    metrics_csv: str | Path,
    out_dir: str | Path,
    sample_indices: list[int] | None = None,
    log_scale: bool = True,
) -> list[Path]:
    """5 张核心图（90 [S3]、80 [S8] C8），PNG+PDF 成对输出。

    figure_01/02/05 取 predictions 的抽样样本（默认第 1 个），
    figure_03/04 由 metrics.csv 聚合（bootstrap CI 固定 seed）。
    """
    data = np.load(predictions_npz, allow_pickle=True)
    raw_ids = list(data["sample_id"])
    sample_ids = [sid.decode("utf-8") if isinstance(sid, bytes) else str(sid) for sid in raw_ids]
    indices = sample_indices if sample_indices is not None else [0]
    i = indices[0]
    if i >= len(sample_ids):
        raise IndexError(f"sample index {i} 超出 predictions 样本数 {len(sample_ids)}")
    H = data["H"][i]
    L_up = data["L_up"][i]
    H_hat = data["H_hat"][i]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_2d_phasespace(
            {"L_up": L_up, "H": H, "Ĥ": H_hat},
            out_dir / "figure_01_2d_phasespace.png",
            log_scale=log_scale,
        ),
        plot_1d_profiles(
            {
                "I(z)": {"H": H.sum(axis=1), "Ĥ": H_hat.sum(axis=1)},
                "S(δ)": {"H": H.sum(axis=0), "Ĥ": H_hat.sum(axis=0)},
            },
            out_dir / "figure_02_1d_profile.png",
        ),
        plot_physics_error_bar(metrics_csv, out_dir / "figure_03_physics_error_bar.png"),
        plot_error_vs_gamma_scatter(metrics_csv, out_dir / "figure_04_error_vs_gamma_scatter.png"),
        plot_residual_map(H, H_hat, out_dir / "figure_05_residual_map.png"),
    ]


def generate_visuals(
    predictions_npz: str | Path,
    out_dir: str | Path,
    sample_indices: list[int] | None = None,
    log_scale: bool = True,
) -> list[Path]:
    """对 predictions.npz 中的抽样样本生成 2D/1D/残差三件套（PNG+PDF）。

    中间可视化命名 ``vis_<EXP>_<sample_id>_<type>.png``（60 [S15] 15.3）。
    """
    data = np.load(predictions_npz, allow_pickle=True)
    raw_ids = list(data["sample_id"])
    sample_ids = [sid.decode("utf-8") if isinstance(sid, bytes) else str(sid) for sid in raw_ids]
    n = len(sample_ids)
    indices = sample_indices if sample_indices is not None else list(range(min(n, 3)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for i in indices:
        sid = str(sample_ids[i])
        H = data["H"][i]
        L_up = data["L_up"][i]
        H_hat = data["H_hat"][i]
        created.append(
            plot_2d_phasespace(
                {"L_up": L_up, "H": H, "Ĥ": H_hat},
                out_dir / f"vis_{sid}_2d.png",
                log_scale=log_scale,
            )
        )
        created.append(
            plot_1d_profiles(
                {
                    "I(z)": {"H": H.sum(axis=1), "Ĥ": H_hat.sum(axis=1)},
                    "S(δ)": {"H": H.sum(axis=0), "Ĥ": H_hat.sum(axis=0)},
                },
                out_dir / f"vis_{sid}_1d.png",
            )
        )
        created.append(
            plot_residual_map(H, H_hat, out_dir / f"vis_{sid}_residual.png")
        )
    return created


def _run_check_env() -> None:
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
    except Exception as exc:  # pragma: no cover
        print(f"[check_env] 无法运行：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.plots",
        description="生成可视化（70 [S8]、60 [S15] 15.3）",
    )
    parser.add_argument("--mode", choices=["intermediate", "core"], default="intermediate",
                        help="intermediate=样本三件套（默认）；core=5 张核心图（90 [S3]）")
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--predictions", required=True, help="infer 输出的 predictions_<split>.npz")
    parser.add_argument("--metrics", help="metrics.csv 路径（core 模式必填）")
    parser.add_argument("--out", required=True, help="输出目录（visuals/ 或 assets/）")
    parser.add_argument("--samples", type=int, default=None, help="抽样样本数（默认 3）")
    args = parser.parse_args(argv)

    _run_check_env()
    config = load_config(args.config)
    log_scale = bool(config.get("visualization", {}).get("log_scale", True))
    if args.mode == "core":
        if not args.metrics:
            parser.error("--mode core 需要 --metrics <metrics.csv>")
        created = generate_core_figures(
            args.predictions, args.metrics, args.out,
            sample_indices=[0], log_scale=log_scale,
        )
        kind = "5 张核心图"
    else:
        n_total = int(np.load(args.predictions, allow_pickle=True)["H_hat"].shape[0])
        sample_indices = list(range(min(args.samples or 3, n_total)))
        created = generate_visuals(args.predictions, args.out, sample_indices, log_scale=log_scale)
        kind = "中间可视化"
    print(f"[plots] 生成 {len(created)} 张图（{kind}）-> {args.out}")
    for path in created[:3]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
