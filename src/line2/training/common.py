"""L3b 共享工具：config 哈希（GC7 溯源；[S1] C3 载入校验用）。"""

from __future__ import annotations

import hashlib
import json

__all__ = ["config_hash"]


def config_hash(cfg: dict) -> str:
    """训练配置的确定性哈希（前 16 位）——ckpt 载入时校验防错配。"""
    return hashlib.sha256(
        json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
