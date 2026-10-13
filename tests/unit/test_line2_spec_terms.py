"""[S1] C5 的验收测试：术语唯一来源声明由一致性检查器兑现。

`50_phase15_respec_spec.md` [S1] C5：`scripts/check_spec_consistency.py` 重跑
SHALL exit 0 且 `spec_claim_index.md` 含 50 的 Claim 条目。同时守卫该检查器的
SPEC_PATTERN 能匹配含数字的 spec 文件名（`50_…`——一审 [S1] C5 缺口的回归钉）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "scripts" / "check_spec_consistency.py"
INDEX = REPO / "spec_claim_index.md"


def test_spec_terms_registered():
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"check_spec_consistency.py exit={result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    index = INDEX.read_text(encoding="utf-8")
    assert "line2/50" in index, "spec_claim_index.md 缺 line2/50 的 Claim 条目（[S1] C5）"


def test_spec_pattern_matches_numbered_names():
    """回归钉：SPEC_PATTERN 必须匹配 `50_phase15_respec_spec.md`（含数字段）。"""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import check_spec_consistency as checker
    finally:
        sys.path.pop(0)
    assert checker.SPEC_PATTERN.match("50_phase15_respec_spec.md"), (
        "SPEC_PATTERN 匹配不到含数字段的 spec 文件名"
        "（50_phase15_respec_spec.md 将再次从索引中消失）"
    )
    assert checker.SPEC_PATTERN.match("10_data_pipeline_spec.md")
