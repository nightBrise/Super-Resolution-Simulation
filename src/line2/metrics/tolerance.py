"""L2 [S4] 观测量向量与容差线（spec line2/20 [S4]）。

13 维观测量向量 = M4（4 对角相对误差 + 6 相关变化 + ε_4D 相对误差）+ M5 + M6。
容差线：每档裸子样本 |q_裸 − q_参照| 分布的 95% 分位；N_eff → 最近档映射
（对数距离，预注册）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import halo_fraction_gap, hf_energy_ratio_gap, reconstruct_sigma, sigma_errors

__all__ = ["observable_vector", "deviation_table", "nearest_node", "tolerance_for"]

N_OBSERVABLES = 13


def observable_vector(group: np.ndarray, ref: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """13 维观测量偏差向量 |q(被评) − q(参照)|（预注册，无现场新增）。"""
    s_e = reconstruct_sigma(group, edges)
    s_r = reconstruct_sigma(ref, edges)
    errs = sigma_errors(s_e, s_r)
    vec = list(errs["diag_rel"]) + list(errs["abs_delta_rho"]) + [errs["eps4d_rel"]]
    vec.append(abs(halo_fraction_gap(group, ref, s_r, edges)))
    vec.append(abs(hf_energy_ratio_gap(group, ref)))
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape != (N_OBSERVABLES,) or not np.isfinite(vec).all():
        raise ValueError(f"观测量向量异常：shape={vec.shape}, finite={np.isfinite(vec).all()}")
    return vec


def deviation_table(
    data_dir: str | Path,
    beam_ids: list[int],
    tiers: tuple[int, ...],
    k: int,
    quantile: float = 0.95,
) -> dict[int, np.ndarray]:
    """各档裸子样本的观测量偏差 95% 分位（容差线数据）。

    从 `10` [S5] 数据集消费：每（束, 站, 档, 子样本）计算 13 维 |Δ|，
    按档聚合取分位。返回 {tier: 13 维分位向量}。
    """
    import h5py

    from .n_eff import _dataset_layout, _beam_split  # 复用数据集读取

    manifest, data_dir = _dataset_layout(data_dir)
    if beam_ids is None:
        beam_ids = [b for s in ("val", "test") for b in manifest["splits"][s]]
    table: dict[int, list[np.ndarray]] = {t: [] for t in tiers}
    files: dict[str, h5py.File] = {}
    try:
        for beam_id in beam_ids:
            split = _beam_split(manifest, beam_id)
            h5 = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            grp = h5[f"beam_{beam_id:04d}"]
            for station_idx in (0, 1):
                stg = grp[f"station{station_idx}"]
                ref = stg["target_full"][...]
                edges = stg["target_full"].attrs["bin_edges"]
                for tier in tiers:
                    for sub_idx in range(k):
                        img = stg[f"tier_{tier:06d}_sub_{sub_idx:02d}"][...]
                        table[tier].append(observable_vector(img, ref, edges))
    finally:
        for f in files.values():
            f.close()
    return {
        t: np.percentile(np.asarray(v), quantile * 100, axis=0) for t, v in table.items()
    }


def nearest_node(curve_nodes: tuple[int, ...] | list[int], n_eff: float) -> int:
    """N_eff → 曲线上对数距离最近的档位（上/下溢时取端点档）。"""
    nodes = np.asarray(curve_nodes, dtype=np.float64)
    return int(nodes[np.argmin(np.abs(np.log10(nodes) - np.log10(max(n_eff, 1.0))))])


def tolerance_for(
    dev_table: dict[int, np.ndarray], curve_nodes: tuple[int, ...] | list[int], n_eff: float
) -> np.ndarray:
    """给定 N_eff 的容差线（13 维）：最近档位的 95% 分位偏差向量。"""
    node = nearest_node(curve_nodes, n_eff)
    return np.asarray(dev_table[node], dtype=np.float64)
