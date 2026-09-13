"""L2 [S5] 矩匹配基线（B-moment，参考级基线尺）。

从输入图组估计 Σ（reconstruct_sigma）+ 投影均值，渲染 2D 高斯期望计数图
（总计数 = 参照总计数，与 M1 比较语义一致）。确定性（同输入逐位一致）。
"""

from __future__ import annotations

import numpy as np

from .metrics import reconstruct_sigma

__all__ = ["moment_matching_baseline"]


def moment_matching_baseline(
    input_group: np.ndarray, ref_group: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """输入六图 → Σ̂ → 渲染六张 2D 高斯期望计数图（和 = 参照逐投影总计数）。"""
    sigma = reconstruct_sigma(input_group, edges)
    out = np.zeros_like(input_group, dtype=np.float64)
    pairs = (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))
    axis_of = {a: i for i, a in enumerate(("x", "xp", "y", "yp"))}
    for gi, (ai, aj) in enumerate(pairs):
        i, j = axis_of[ai], axis_of[aj]
        sub = np.array([[sigma[i, i], sigma[i, j]], [sigma[j, i], sigma[j, j]]])
        det = sub[0, 0] * sub[1, 1] - sub[0, 1] * sub[1, 0]
        if det <= 0:
            # Σ̂ 非正定（输入退化）：回退为对角化
            sub = np.diag(np.clip(np.diag(sub), 1e-12, None))
            det = sub[0, 0] * sub[1, 1]
        inv = np.array([[sub[1, 1], -sub[0, 1]], [-sub[1, 0], sub[0, 0]]]) / det
        ui = 0.5 * (edges[i][:-1] + edges[i][1:])
        uj = 0.5 * (edges[j][:-1] + edges[j][1:])
        # 投影均值：由输入图组的边缘分布估计（比 0 假设更贴近真实束心）
        img_in = input_group[gi].astype(np.float64)
        tot_in = max(img_in.sum(), 1.0)
        mean_i = float((img_in.sum(axis=1) * ui).sum() / tot_in)
        mean_j = float((img_in.sum(axis=0) * uj).sum() / tot_in)
        du0 = ui - mean_i
        du1 = uj - mean_j
        quad = (
            inv[0, 0] * du0[:, None] ** 2
            + 2.0 * inv[0, 1] * np.outer(du0, du1)
            + inv[1, 1] * du1[None, :] ** 2
        )
        w = np.exp(-0.5 * quad)
        n_high = ref_group[gi].sum()  # 渲染总计数 = 参照（N_high 语义）
        rate = w / w.sum() * n_high
        out[gi] = rate
    return out
