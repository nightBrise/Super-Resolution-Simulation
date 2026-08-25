"""评估模块：指标、评估 CLI、推理 CLI 与可视化（70 规格）。

- ``metrics``：图像级 / 物理级 / 精细结构级指标 + 统计判定协议
  （70 [S3]–[S7]）；
- ``evaluate``：评估 CLI（输出 metrics.csv + summary.json，80 [S8]）；
- ``infer``：推理 CLI（复用 checkpoint 输出预测，60 [S15] 15.7）；
- ``plots``：可视化（70 [S8]，PNG + PDF，核心图命名预留 90 [S3]）。
"""

from src.evaluation.metrics import (
    PRIMARY_METRIC_COL,
    SECONDARY_METRIC_COL,
    TAU,
    TRIGGER_RATE,
    bootstrap_ci,
    dog_sigma_outer,
    e_high_doG,
    evaluate_sample,
    hallucination_flag,
    high_freq_energy_ratio,
    holm_correction,
    overshoot_smooth_class,
    paired_wilcoxon,
    prior_gain_stats,
    three_class,
    veto_verdict,
)

__all__ = [
    "PRIMARY_METRIC_COL",
    "SECONDARY_METRIC_COL",
    "TAU",
    "TRIGGER_RATE",
    "bootstrap_ci",
    "dog_sigma_outer",
    "e_high_doG",
    "evaluate_sample",
    "hallucination_flag",
    "high_freq_energy_ratio",
    "holm_correction",
    "overshoot_smooth_class",
    "paired_wilcoxon",
    "prior_gain_stats",
    "three_class",
    "veto_verdict",
]
