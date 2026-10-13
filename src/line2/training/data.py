"""L3b 数据加载 + Phase 1.5 混合档批源（spec line2/30 + `50` [S4]）。

样本 = (低 N 输入计数图组, 2e5 目标图组, bin_edges, N_low)。κ 条件向量在
模型入口由输入派生（防泄露，`30` [S1] C5）——本模块不接触目标以构造 κ。

:class:`MixedTierBatcher` 承载 `50` [S4] 的混合档训练：每 step 按档均匀抽批，
档位流与批序流由 :func:`training_streams` 从 SeedSequence(20260913) 独立 spawn
（与数据子样本链不共享熵池，禁裸随机）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

import h5py
import numpy as np
import torch

from src.line2.data_pipeline.seeds import MASTER_SEED

__all__ = [
    "Line2Pairs",
    "MixedTierBatcher",
    "collate_pairs",
    "dataset_manifest",
    "training_streams",
    "val_curve_for",
]

#: 训练侧种子树的根分支（`50` D7）：数据子样本链占 `spawn(beam_idx+1)` 的 0..999，
#: 训练流另起 1000 号分支，两者不共享熵池。
TRAIN_BRANCH = 1000


def training_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """(档位流, 批序流)——逐 run 从 SeedSequence(20260913) 派生、独立 spawn。

    派生根 = ``SeedSequence(20260913).spawn(TRAIN_BRANCH + 1)[TRAIN_BRANCH]``；
    其下以 (master_seed, run_seed) 为熵、沿用根分支的 spawn_key 生成 run 节点，
    再由该节点 ``spawn(2)`` 分出两条互不共享状态的流（`50` D7：档位流与批序流
    独立，且都与数据子样本流独立）。同 (seed) ⇒ 同序列；run 种子 {100..104}
    不同 ⇒ 序列不同。
    """
    root = np.random.SeedSequence(MASTER_SEED).spawn(TRAIN_BRANCH + 1)[TRAIN_BRANCH]
    run = np.random.SeedSequence([MASTER_SEED, int(seed)], spawn_key=root.spawn_key)
    tier_ss, batch_ss = run.spawn(2)
    return np.random.default_rng(tier_ss), np.random.default_rng(batch_ss)


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
        dset = stg[f"tier_{self.n_low:06d}_sub_{sub_idx:02d}"]
        img = dset[...]
        target = stg["target_full"][...]
        edges = dset.attrs["bin_edges"]
        return {
            "counts": torch.from_numpy(img.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "edges": torch.from_numpy(edges.astype(np.float32)),
            "beam_id": beam_id,
            "station": station_idx,
            "sub_idx": sub_idx,
        }


def collate_pairs(items: list[dict]) -> dict:
    """(束, 子样本) 样本列表 → 批张量字典（训练/评估侧共用，``station`` 等留列表）。"""
    batch = {
        key: torch.stack([item[key] for item in items])
        for key in ("counts", "target", "edges")
    }
    batch["beam_id"] = [item["beam_id"] for item in items]
    batch["station"] = [item["station"] for item in items]
    batch["sub_idx"] = [item["sub_idx"] for item in items]
    return batch


class MixedTierBatcher:
    """混合档批源（`50` [S4]）：每 step 按档均匀抽一档，再在该档池内无放回抽 batch。

    ``datasets`` = {tier: Line2Pairs}；档位次序取 ``sorted(datasets)``（确定性）。
    ``__iter__`` 为无限序列，训练循环按 step 顺序消费 ⇒ 同 seed 同 config 的
    批序列逐位确定；档位流与批序流独立 spawn（:func:`training_streams`），故
    改 ``batch_size`` 只改批序流、不改档位序列。
    """

    def __init__(
        self,
        datasets: Mapping[int, "Line2Pairs"],
        batch_size: int,
        seed: int,
        tiers: tuple[int, ...] | None = None,
    ) -> None:
        self.datasets = {int(tier): ds for tier, ds in datasets.items()}
        self.tiers = tuple(sorted(self.datasets)) if tiers is None else tuple(tiers)
        self.batch_size = int(batch_size)
        for tier in self.tiers:
            if len(self.datasets[tier]) < self.batch_size:
                raise ValueError(
                    f"档 {tier} 的样本数 {len(self.datasets[tier])} < batch_size "
                    f"{self.batch_size}（无放回抽批不可能）"
                )
        self._tier_rng, self._batch_rng = training_streams(seed)

    def draw(self) -> dict:
        """抽一批：返回 :func:`collate_pairs` 的批字典 + ``tier`` 键。"""
        tier = self.tiers[int(self._tier_rng.integers(len(self.tiers)))]
        dset = self.datasets[tier]
        indices = self._batch_rng.choice(len(dset), size=self.batch_size, replace=False)
        batch = collate_pairs([dset[int(i)] for i in indices])
        batch["tier"] = tier
        return batch

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.draw()
