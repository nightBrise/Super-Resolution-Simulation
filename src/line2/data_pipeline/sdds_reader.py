"""最小 SDDS1 二进制读取器（line2 锚点格式）。

pylhc ``sdds==0.4.3`` 的二进制列读取未实现（``_read_bin_column`` 直接抛
``NotImplementedError``），故按锚点文件的确切格式自实现：

- ``SDDS1`` + ``!# little-endian``；
- ``&parameter``/``&column`` 定义（本管线仅需 ``type=double``）；
- ``&data mode=binary``；二进制页布局（对锚点文件实测验证）：
  ``int32 行数`` → ``double 参数值（按定义序）`` → ``行主序列数据``。

字段定义从 header 解析而非硬编码列名；遇到本格式之外的情形（大端、
非 double 数值列、多页、文件截断）抛 :class:`SDDSFormatError`——
明确报错优于静默错误。验收契约：spec line2/10 [S2] C1（与手工页布局
解析逐位一致）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SDDSFormatError", "SDDSPage", "read_sdds_binary"]

_LE_MARKER = b"!# little-endian"
_DATA_LINE_PREFIX = b"&data"
_DOUBLE = "double"


class SDDSFormatError(Exception):
    """锚点 SDDS 文件格式不在自实现读取器的覆盖范围内。"""


@dataclass
class SDDSPage:
    """单个 SDDS 二进制数据页的解析结果。"""

    row_count: int
    parameter_values: dict[str, float]
    column_names: tuple[str, ...]
    columns: dict[str, "np.ndarray"] = field(default_factory=dict)


def _parse_header(raw: bytes) -> tuple[list[dict[str, str]], bool, int]:
    """解析 SDDS1 头部，返回 (字段定义列表, 是否小端, 数据区起始偏移)。"""
    definitions: list[dict[str, str]] = []
    little_endian = False
    pos = 0
    for line in raw.split(b"\n"):
        stripped = line.strip()
        if stripped == _LE_MARKER:
            little_endian = True
        elif stripped.startswith(b"&parameter") or stripped.startswith(b"&column"):
            text = stripped.decode("ascii", "replace")
            entry: dict[str, str] = {"kind": text[1:].split()[0]}
            for key in ("name", "type"):
                m = re.search(rf"\b{key}=(\w+)", text)
                if m:
                    entry[key] = m.group(1)
            if "name" not in entry or "type" not in entry:
                raise SDDSFormatError(f"字段定义缺少 name/type：{stripped!r}")
            definitions.append(entry)
        elif stripped.startswith(_DATA_LINE_PREFIX):
            if b"binary" not in stripped:
                raise SDDSFormatError("仅支持 &data mode=binary（锚点格式）")
            return definitions, little_endian, pos + len(line) + 1
        pos += len(line) + 1
    raise SDDSFormatError("未找到 &data 段")


def read_sdds_binary(path: str | Path) -> SDDSPage:
    """读取单页 SDDS1 二进制文件（锚点格式：int32 行数 → double 参数 → 行主序 double 列）。"""
    import numpy as np

    raw = Path(path).read_bytes()
    definitions, little_endian, offset = _parse_header(raw)
    if not little_endian:
        raise SDDSFormatError("仅支持 '!# little-endian'（锚点格式）")
    data = raw[offset:]

    params = [d for d in definitions if d["kind"] == "parameter"]
    columns = [d for d in definitions if d["kind"] == "column"]
    for d in params + columns:
        if d["type"] != _DOUBLE:
            raise SDDSFormatError(f"仅支持 type=double，发现 {d['type']!r}（{d['name']}）")

    (row_count,) = np.frombuffer(data[:4], dtype="<i4")
    if row_count <= 0:
        raise SDDSFormatError(f"页行数非法：{row_count}")
    cursor = 4
    parameter_values: dict[str, float] = {}
    for d in params:
        (value,) = np.frombuffer(data[cursor : cursor + 8], dtype="<f8")
        parameter_values[d["name"]] = float(value)
        cursor += 8

    row_size = 8 * len(columns)
    body = len(data) - cursor
    if body != row_count * row_size:
        raise SDDSFormatError(
            f"数据区长度 {body} 与 行数 {row_count} × 行宽 {row_size} 不符（多页/截断不支持）"
        )
    names = tuple(d["name"] for d in columns)
    records = np.frombuffer(
        data[cursor : cursor + body], dtype=np.dtype([(n, "<f8") for n in names])
    )
    page = SDDSPage(row_count=int(row_count), parameter_values=parameter_values, column_names=names)
    for n in names:
        page.columns[n] = np.ascontiguousarray(records[n], dtype=np.float64)
    return page
