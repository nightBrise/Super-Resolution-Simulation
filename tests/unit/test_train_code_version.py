"""train.py code_version 守卫单元测试（N4 纪律，60 [S15] C11）。

覆盖 _ensure_code_version：占位符/缺失 → 覆盖为当前 git HEAD 40 位 hash；
合法 40 位 hash → 保持不变（不重写历史运行记录）。
"""

from __future__ import annotations

import re

import pytest

from src.training.train import _ensure_code_version

pytestmark = pytest.mark.unit

HEAD_RE = re.compile(r"^[0-9a-f]{40}$")


def test_placeholder_overwritten():
    cfg = {"code_version": "<git rev-parse HEAD 完整 40 位 hash>"}
    _ensure_code_version(cfg)
    assert HEAD_RE.match(str(cfg["code_version"]))


def test_missing_filled():
    cfg: dict = {}
    _ensure_code_version(cfg)
    assert HEAD_RE.match(str(cfg["code_version"]))


def test_valid_hash_untouched():
    cfg = {"code_version": "f4b2d07cce1bbd803bc60a97cd774f7065510de2"}
    _ensure_code_version(cfg)
    assert cfg["code_version"] == "f4b2d07cce1bbd803bc60a97cd774f7065510de2"


def test_short_hash_overwritten():
    cfg = {"code_version": "abc123"}
    _ensure_code_version(cfg)
    assert HEAD_RE.match(str(cfg["code_version"]))
