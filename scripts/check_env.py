#!/usr/bin/env python3
"""环境校验脚本（漏洞 2.3）。

每个训练开始前自动跑。不通过仅警告，不视为失败；登记 99。
"""
import sys
import torch
import numpy
import h5py
import scipy

EXPECTED = {
    'torch': '2.4.0',
    'numpy': '1.26.4',
    'h5py': '3.11.0',
    'scipy': '1.13.0',
}

ACTUAL = {
    'torch': torch.__version__,
    'numpy': numpy.__version__,
    'h5py': h5py.__version__,
    'scipy': scipy.__version__,
}


def main() -> int:
    mismatches = []
    for pkg, expected in EXPECTED.items():
        actual = ACTUAL[pkg]
        if actual != expected:
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
