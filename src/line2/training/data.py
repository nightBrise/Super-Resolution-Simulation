"""L3b 数据加载：Phase 1 训练对（spec line2/30 + `10` [S5] 数据集）。

样本 = (低 N 输入计数图组, 2e5 目标图组, bin_edges, N_low)。κ 条件向量在
模型入口由输入派生（防泄露，`30` [S1] C5）——本模块不接触目标以构造 κ。
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

__all__ = ["Line2Pairs", "dataset_manifest", "val_curve_for"]


def dataset_manifest(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def val_curve_for(data_dir: str | Path, stations: tuple[int, ...] = (0, 1)):
    """val 束的裸误差–N 曲线（早停/选型准绳 val N_eff 的插值基底）。"""
    from src.line2.metrics.n_eff import error_n_curve

    manifest = dataset_manifest(data_dir)
    val_beams = [int(b) for b in manifest["splits"]["val"]]
    if not val_beams:
        raise ValueError(
            "val 切分为空：无法构建 val N_eff 曲线（n_beams 过小导致 15% 切分取整为 0）"
        )
    return error_n_curve(data_dir, beam_ids=val_beams, stations=stations)


class Line2Pairs(torch.utils.data.Dataset):
    """训练/验证对：指定切分 HDF5 中的 (tier N_low 子样本, target_full) 对。

    - 打开固定 HDF5 句柄（num_workers=0 使用；进程不 fork 句柄）；
    - 每束每站每子样本一个样本（station 由构造参数固定或全部）；
    - 顺序由调用方的确定性 DataLoader 派生（GC2）。
    """

    def __init__(
        self,
        h5_path: str | Path,
        beam_ids: list[int],
        n_low: int = 10_000,
        stations: tuple[int, ...] = (0, 1),
        subs: tuple[int, ...] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.beam_ids = list(beam_ids)
        self.n_low = int(n_low)
        self.stations = tuple(stations)
        self.subs = tuple(subs) if subs is not None else tuple(range(20))
        self._h5 = h5py.File(self.h5_path, "r")
        self._index: list[tuple[int, int, int]] = []
        for beam_id in self.beam_ids:
            grp = self._h5[f"beam_{beam_id:04d}"]
            for station_idx in self.stations:
                stg = grp[f"station{station_idx}"]
                for sub_idx in self.subs:
                    name = f"tier_{self.n_low:06d}_sub_{sub_idx:02d}"
                    if name in stg:
                        self._index.append((beam_id, station_idx, sub_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        beam_id, station_idx, sub_idx = self._index[i]
        grp = self._h5[f"beam_{beam_id:04d}"]
        stg = grp[f"station{station_idx}"]
        img = stg[f"tier_{self.n_low:06d}_sub_{sub_idx:02d}"][...]
        target = stg["target_full"][...]
        edges = stg["tier_001000_sub_00"].attrs["bin_edges"]
        return {
            "counts": torch.from_numpy(img.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "edges": torch.from_numpy(edges.astype(np.float32)),
            "beam_id": beam_id,
            "station": station_idx,
            "sub_idx": sub_idx,
        }
