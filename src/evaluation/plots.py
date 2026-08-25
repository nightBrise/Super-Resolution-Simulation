"""可视化模块（70 [S8]、60 [S15] 15.3）。

基础可视化三件套：
- 2D 相空间并排对比（对数色标 LogNorm，70 [S8] C1）；
- 1D 电流/能谱剖面曲线对比（70 [S8] C2）；
- 残差图（发散色标，红蓝区分多余/丢失粒子，70 [S8] C3）。

输出格式 PNG + PDF（60 [S15] 15.3），核心图命名预留（90 [S3]）：
``figure_0<N>_<type>.<png|pdf>``（figure_01_2d_phasespace /
figure_02_1d_profile / figure_03_physics_error_bar /
figure_04_error_vs_gamma_scatter / figure_05_residual_map）。

CLI：``python -m src.evaluation.plots --config <path> --predictions <npz> --out <dir>``
（predictions npz 由 ``infer`` 输出）。色标类型（log / diverging）与
配置文件中的可视化设置一致（70 [S8] C4）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

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
    fig.savefig(str(out_path), dpi=150)
    fig.savefig(str(out_path.with_suffix(".pdf")))
    plt.close(fig)
    return out_path


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
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--predictions", required=True, help="infer 输出的 predictions_<split>.npz")
    parser.add_argument("--out", required=True, help="输出目录（visuals/）")
    parser.add_argument("--samples", type=int, default=None, help="抽样样本数（默认 3）")
    args = parser.parse_args(argv)

    _run_check_env()
    config = load_config(args.config)
    log_scale = bool(config.get("visualization", {}).get("log_scale", True))
    n_total = int(np.load(args.predictions, allow_pickle=True)["H_hat"].shape[0])
    sample_indices = list(range(min(args.samples or 3, n_total)))
    created = generate_visuals(args.predictions, args.out, sample_indices, log_scale=log_scale)
    print(f"[plots] 生成 {len(created)} 张图 -> {args.out}")
    for path in created[:3]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
