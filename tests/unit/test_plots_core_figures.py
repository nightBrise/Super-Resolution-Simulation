"""核心图 figure_03/04 生成单元测试（90 [S3] 图 3/4，80 [S8] C8）。

覆盖：
- plot_physics_error_bar / plot_error_vs_gamma_scatter 输出 PNG+PDF 成对；
- 固定 seed 下两次生成逐字节一致（bootstrap CI 可复现，05 [S6] 随机纪律）。

测试只断言文件契约与可复现性（05 [S6]：禁止断言研究结果）。
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.evaluation.plots import plot_error_vs_gamma_scatter, plot_physics_error_bar

pytestmark = pytest.mark.unit

METRIC_KEYS = ("sample_id", "scheme", "gamma", "e_eps_z", "e_I_peak")


@pytest.fixture
def metrics_csv(tmp_path):
    rows = []
    for scheme in "ABC":
        for i in range(20):
            rows.append(
                {
                    "sample_id": f"test-{scheme}-{i:02d}",
                    "scheme": scheme,
                    "gamma": f"{0.1 + 0.01 * i:.6f}",
                    "e_eps_z": f"{1e-3 * (1 + 0.3 * (scheme != 'A') + 0.05 * np.sin(i)):.9f}",
                    "e_I_peak": f"{2e-3 * (1 + 0.2 * (i % 3)):.9f}",
                }
            )
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(METRIC_KEYS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_physics_error_bar_outputs_png_only(metrics_csv, tmp_path):
    # 用户 2026-08-28 要求 PNG only（300dpi），不再生成 PDF；断言 PDF 不产出
    out = tmp_path / "figure_03_physics_error_bar.png"
    created = plot_physics_error_bar(metrics_csv, out)
    assert created == out
    assert out.exists() and out.stat().st_size > 0
    assert not out.with_suffix(".pdf").exists()


def test_error_vs_gamma_scatter_outputs_png_only(metrics_csv, tmp_path):
    out = tmp_path / "figure_04_error_vs_gamma_scatter.png"
    created = plot_error_vs_gamma_scatter(metrics_csv, out)
    assert created == out
    assert out.exists() and out.stat().st_size > 0
    assert not out.with_suffix(".pdf").exists()


def test_physics_error_bar_reproducible_with_fixed_seed(metrics_csv, tmp_path):
    out_a = tmp_path / "fig_a.png"
    out_b = tmp_path / "fig_b.png"
    plot_physics_error_bar(metrics_csv, out_a, seed=20260825)
    plot_physics_error_bar(metrics_csv, out_b, seed=20260825)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_error_vs_gamma_scatter_reproducible(metrics_csv, tmp_path):
    out_a = tmp_path / "scatter_a.png"
    out_b = tmp_path / "scatter_b.png"
    plot_error_vs_gamma_scatter(metrics_csv, out_a)
    plot_error_vs_gamma_scatter(metrics_csv, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()
