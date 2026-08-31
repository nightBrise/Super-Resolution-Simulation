"""run_output_dir 研究线根（study_root）单测（方案 B）。

验证：config 顶层 `study_root` 非空时，实验输出目录落于
`<root>/studies/<study_root>/results/<name>`；为空/缺省时兜底
`<root>/results/<name>`（向后兼容，行为与迁移前一致）。
"""
from pathlib import Path

import pytest

from src.utils.config_utils import run_output_dir

pytestmark = pytest.mark.unit


def _cfg(**overrides) -> dict:
    base = {
        "experiment_id": "EXP-02",
        "scheme": "A",
        "seed_index": 0,
        "run_tag": "run1",
    }
    base.update(overrides)
    return base


def test_run_output_dir_default_falls_back_to_results():
    """缺省 study_root → 兜底 <root>/results/<name>（迁移前行为）。"""
    out = run_output_dir(_cfg(), project_root=Path("/proj"))
    assert out == Path("/proj/results/EXP-02_A_seed0_run1")


def test_run_output_dir_with_study_root():
    """study_root 非空 → 走 <root>/studies/<study_root>/results/<name>。"""
    out = run_output_dir(_cfg(study_root="line1_substitute_sr"), project_root=Path("/proj"))
    assert out == Path("/proj/studies/line1_substitute_sr/results/EXP-02_A_seed0_run1")


def test_run_output_dir_with_config_tag():
    """config_tag 非空 → 名称追加 <tag>，且研究线根仍生效。"""
    out = run_output_dir(
        _cfg(study_root="line1_substitute_sr", config_tag="D2"), project_root=Path("/proj")
    )
    assert out == Path("/proj/studies/line1_substitute_sr/results/EXP-02_A_seed0_run1_D2")


def test_run_output_dir_blank_study_root_acts_like_default():
    """study_root 为空白串 → 视同缺省，兜底 results。"""
    out = run_output_dir(_cfg(study_root="   "), project_root=Path("/proj"))
    assert out == Path("/proj/results/EXP-02_A_seed0_run1")
