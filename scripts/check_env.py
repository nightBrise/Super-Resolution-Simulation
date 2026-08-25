#!/usr/bin/env python3
"""环境校验脚本（漏洞 2.3）。

每个训练开始前自动跑。不通过仅警告，不视为失败；登记 99。
"""
import sys
import torch
import numpy
import h5py
import scipy
import skimage
import pytest

EXPECTED = {
    'torch': '2.4.0',
    'numpy': '1.26.4',
    'h5py': '3.11.0',
    'scipy': '1.13.0',
    'skimage': '0.24.0',
    'pytest': '>=8',  # 与 environment.yml 的 pytest>=8 一致
}

ACTUAL = {
    'torch': torch.__version__,
    'numpy': numpy.__version__,
    'h5py': h5py.__version__,
    'scipy': scipy.__version__,
    'skimage': skimage.__version__,
    'pytest': pytest.__version__,
}


def _version_ok(expected: str, actual: str) -> bool:
    """按期望串比较版本：支持 `>=X` 下限形式与精确相等两种。"""
    if expected.startswith('>='):
        floor = expected[2:]
        return actual == floor or _parse(actual) >= _parse(floor)
    return actual == expected


def _parse(version: str) -> tuple:
    """取版本号的数字前缀元组用于比较（忽略 `+`/字母后缀）。"""
    parts = []
    for piece in version.split('.'):
        digits = ''.join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def main() -> int:
    mismatches = []
    for pkg, expected in EXPECTED.items():
        actual = ACTUAL[pkg]
        if not _version_ok(expected, actual):
            mismatches.append(f'  { pkg}: expected {expected}, got {actual}')

    if not mismatches:
        print('OK: 环境依赖版本一致')
        return 0

    print('WARN: 环境依赖版本不一致')
    for line in mismatches:
        print(line)
    print('\n建议：运行 conda env update -f environment.yml 或重新创建环境')
    return 1


if __name__ == '__main__':
    sys.exit(main())
