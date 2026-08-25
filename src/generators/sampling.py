"""内容参数采样：压缩因子三态联合采样加 W1–W8 拒绝筛选。

本模块实现 20 规格 [S9] 的采样规则：先按预注册比例（默认各 1/3）抽取
压缩三态（欠压缩 ``C ∈ (0.1, 0.9]``、最佳压缩 ``C ∈ [-0.1, 0.1]``、
过压缩 ``C ∈ [-0.5, -0.1)``），再抽取 ``a_1`` 并由 ``α = (C−1)/a_1``
精确导出 ``α``；候选组合必须全部通过掩膜 W1–W8 方可入选（20 [S9] C7）。
随机数全部由显式 ``master_seed`` 经 ``numpy.random.SeedSequence`` 派生，
生成器不自选随机源（60 [S14] C4）。
"""

from __future__ import annotations

import numpy as np

from src.generators.masks import (
    MASK_VERSION,
    apply_masks,
    fine_structure_width,
    DELTA_PX,
    MASK_NAMES,
)

#: 压缩三态的取值区间，与 20 [S9] 联合约束 1 的区间定义一致。
_COMPRESSION_BANDS: dict[str, tuple[float, float]] = {
    "under": (0.1, 0.9),
    "optimal": (-0.1, 0.1),
    "over": (-0.5, -0.1),
}

#: 各参数名称在样本字典中的登记顺序。
PARAMETER_KEYS: tuple[str, ...] = (
    "A",
    "sigma_z",
    "n",
    "eta",
    "b0",
    "a1",
    "alpha",
    "a2",
    "a3",
    "beta",
    "gamma",
    "b1",
    "C",
)

#: 掩膜预筛阶段（σ_K 未给定时）至少收集的 W1–W7 通过候选数，
#: 用于以中位数稳定估计 σ_K 初始值（30 [S12] 口径：σ_K ≈ 2×w_fine 中位数）。
_PRESCREEN_MIN_ACCEPTED = 512


def _draw_candidates(
    n: int, rng: np.random.Generator, ood_scale: float = 1.0
) -> dict[str, np.ndarray]:
    """按 20 [S9]「参数数值范围」小节抽取 ``n`` 组未筛选的候选参数。

    抽取顺序遵循联合约束 1：先抽压缩三态，再抽 ``a_1``，随后由
    ``α = (C−1)/a_1`` 导出 ``α``；``γ`` 的符号恒与 ``β`` 相反
    （chicane 二阶与三阶项反号）。

    ``ood_scale`` 用于 EXP-06 极端参数采样（80 [S7]）：``β``/``γ`` 的幅度
    区间整体乘以该倍数（双叶按 |·| 上界对称放大，``ood_scale=1.0`` 时输出
    与原始区间逐位一致）。
    """
    sigma_z = rng.uniform(0.30, 0.70, n)
    n_exp = rng.uniform(1.0, 3.0, n)
    eta = rng.uniform(-0.3, 0.3, n)
    b0 = rng.uniform(0.04, 0.09, n)

    a1_mag = rng.uniform(0.35, 0.75, n)
    a1_sign = rng.choice([-1.0, 1.0], n)
    a1 = a1_sign * a1_mag

    state_index = rng.integers(0, 3, n)
    bands = [_COMPRESSION_BANDS[name] for name in ("under", "optimal", "over")]
    c_value = np.empty(n)
    for i, (low, high) in enumerate(bands):
        sel = state_index == i
        c_value[sel] = rng.uniform(low, high, int(sel.sum()))
    alpha = (c_value - 1.0) / a1

    a2 = rng.uniform(-0.10, 0.10, n)

    beta_mag = rng.uniform(0.9, 2.0, n) * ood_scale
    beta_sign = rng.choice([-1.0, 1.0], n)
    beta = beta_sign * beta_mag

    a3 = rng.uniform(-0.05, 0.05, n)
    gamma = -beta_sign * (rng.uniform(0.1, 0.6, n) * ood_scale)
    b1 = rng.uniform(-0.10, 0.20, n)

    return {
        "A": np.ones(n),
        "sigma_z": sigma_z,
        "n": n_exp,
        "eta": eta,
        "b0": b0,
        "a1": a1,
        "alpha": alpha,
        "a2": a2,
        "a3": a3,
        "beta": beta,
        "gamma": gamma,
        "b1": b1,
        "C": c_value,
    }


def _state_of(c_value: np.ndarray) -> np.ndarray:
    """把压缩因子取值划分到三态：0 欠压缩、1 最佳压缩、2 过压缩。"""
    state = np.full(c_value.shape, 1, dtype=np.int64)
    state[c_value > 0.1] = 0
    state[c_value < -0.1] = 2
    return state


def sample_parameters(
    n: int,
    master_seed: int,
    sigma_K: float | None = None,
    max_candidates: int = 50_000,
    gamma_block: tuple[str, float, float] | None = None,
) -> tuple[list[dict[str, float]], dict]:
    """拒绝采样 ``n`` 组全部通过 W1–W8 的内容参数组合。

    参数
    ----
    n: 需要的入选样本数。
    master_seed: 主随机种子，经 ``SeedSequence`` 派生全部随机数。
    sigma_K: 以 H 像素为单位的模糊核宽度；``None`` 时先抽取通过 W1–W7 的
        候选并以 ``σ_K = 2 × median(w_fine)`` 估计初始值（20 [S9] 掩膜备注
        与 30 [S12] 同口径），再筛选 W8。
    max_candidates: 候选抽取总数上限，超限仍未集满 ``n`` 个样本时抛出
        ``RuntimeError``。
    gamma_block: γ 块条件采样（60 [S8] C4），``("inside", lo, hi)`` 仅接受
        ``|γ| ∈ [lo, hi]`` 的候选（参数分块留出 test_pb），
        ``("outside", lo, hi)`` 拒绝块内候选（train / val / test_id 块外
        采样）；``None`` 不加约束。块边界由固定总体分位数确定，不经此函数
        重新估计。

    返回
    ----
    ``(samples, stats)``：``samples`` 为 ``n`` 个参数映射（键见
    ``PARAMETER_KEYS`` 加 ``compression_state``）；``stats`` 记录掩膜版本、
    逐掩膜拒绝计数、W8 通过比例、γ 块拒绝计数与三态计数（20 [S9] C8）。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if gamma_block is not None:
        mode, lo, hi = gamma_block
        if mode not in ("inside", "outside"):
            raise ValueError(f"gamma_block 模式必须为 inside/outside，收到 {mode!r}")
        if lo >= hi:
            raise ValueError(f"gamma_block 区间非法：[{lo}, {hi}]")
    rng = np.random.default_rng(np.random.SeedSequence(master_seed))

    stats: dict = {
        "mask_version": MASK_VERSION,
        "master_seed": int(master_seed),
        "n_requested": int(n),
        "n_candidates": 0,
        "per_mask_rejected": {name: 0 for name in MASK_NAMES},
        "w1_w7_passers": 0,
        "gamma_block": (
            {"mode": gamma_block[0], "interval": [gamma_block[1], gamma_block[2]]}
            if gamma_block is not None
            else None
        ),
        "gamma_block_rejected": 0,
        "w8_fraction_among_w1_w7_passers": float("nan"),
        "sigma_K_px": float(sigma_K) if sigma_K is not None else float("nan"),
    }

    if sigma_K is None:
        sigma_K = _prescreen_sigma_k(rng, stats)
        stats["sigma_K_px"] = float(sigma_K)

    # 三态配额按预注册比例均分，收集满额即停，保证三态覆盖严格均衡。
    quotas = np.full(3, n // 3, dtype=np.int64)
    quotas[: n - quotas.sum()] += 1
    collected: list[np.ndarray] = []
    state_counts = np.zeros(3, dtype=np.int64)
    w8_passed = 0
    w1_w7_screened = 0

    batch_size = max(4096, 4 * n)
    while stats["n_candidates"] < max_candidates and (state_counts < quotas).any():
        draw = min(batch_size, max_candidates - stats["n_candidates"])
        stats["n_candidates"] += draw
        cand = _draw_candidates(draw, rng)
        masks = apply_masks(cand, sigma_K)

        passed_w1_w7 = np.ones(draw, dtype=bool)
        for name in MASK_NAMES[:-1]:
            ok = np.asarray(masks[name], dtype=bool)
            stats["per_mask_rejected"][name] += int((~ok).sum())
            passed_w1_w7 &= ok
        stats["w1_w7_passers"] += int(passed_w1_w7.sum())
        w1_w7_screened += int(passed_w1_w7.sum())

        ok_w8 = np.asarray(masks["W8"], dtype=bool)
        stats["per_mask_rejected"]["W8"] += int((passed_w1_w7 & ~ok_w8).sum())

        if gamma_block is not None:
            block_mode, lo, hi = gamma_block
            gamma_mag = np.abs(cand["gamma"])
            inside = (gamma_mag >= lo) & (gamma_mag <= hi)
            block_ok = inside if block_mode == "inside" else ~inside
            stats["gamma_block_rejected"] += int((~block_ok).sum())
        else:
            block_ok = np.ones(draw, dtype=bool)

        passed = passed_w1_w7 & ok_w8 & block_ok
        w8_passed += int(passed.sum())

        states = _state_of(cand["C"])
        for row in np.nonzero(passed)[0]:
            s = states[row]
            if state_counts[s] < quotas[s]:
                state_counts[s] += 1
                collected.append(
                    np.array([cand[key][row] for key in PARAMETER_KEYS])
                )

    if (state_counts < quotas).any():
        raise RuntimeError(
            f"候选上限 {max_candidates} 内仅收集 {int(state_counts.sum())}/{n} "
            f"个通过 W1–W8 的参数组合；请复核参数范围与掩膜定义"
        )

    # W8 通过比例只统计经过 W8 判定的候选（不含 σ_K 预筛阶段的候选，
    # 该阶段以临时放宽的 σ_K 抽取，不应计入分母；20 [S9] C9 统计口径）。
    stats["w8_fraction_among_w1_w7_passers"] = (
        w8_passed / w1_w7_screened if w1_w7_screened > 0 else float("nan")
    )
    stats["w1_w7_passers_screened"] = int(w1_w7_screened)
    stats["n_accepted"] = int(state_counts.sum())
    stats["acceptance_rate"] = stats["n_accepted"] / stats["n_candidates"]
    stats["state_counts"] = {
        name: int(count)
        for name, count in zip(("under", "optimal", "over"), state_counts)
    }

    matrix = np.vstack(collected)
    samples = []
    for i in range(n):
        sample = {key: float(matrix[i, j]) for j, key in enumerate(PARAMETER_KEYS)}
        sample["compression_state"] = ("under", "optimal", "over")[
            _state_of(np.array([sample["C"]]))[0]
        ]
        samples.append(sample)
    return samples, stats


def _prescreen_sigma_k(
    rng: np.random.Generator,
    stats: dict,
    min_accepted: int = _PRESCREEN_MIN_ACCEPTED,
) -> float:
    """以通过 W1–W7 的候选估计 ``σ_K = 2 × median(w_fine)``（像素单位）。"""
    widths_px: list[float] = []
    while len(widths_px) < min_accepted:
        if stats["n_candidates"] >= 50_000:
            raise RuntimeError("σ_K 预筛阶段候选耗尽，未收集到足够 W1–W7 通过样本")
        draw = 4096
        stats["n_candidates"] += draw
        cand = _draw_candidates(draw, rng)
        masks = apply_masks(cand, sigma_K=1.0e9)  # W8 上界临时放开，仅筛 W1–W7
        passed_w1_w7 = np.ones(draw, dtype=bool)
        for name in MASK_NAMES[:-1]:
            ok = np.asarray(masks[name], dtype=bool)
            stats["per_mask_rejected"][name] += int((~ok).sum())
            passed_w1_w7 &= ok
        stats["w1_w7_passers"] += int(passed_w1_w7.sum())
        widths = fine_structure_width(cand)[passed_w1_w7] / DELTA_PX
        widths_px.extend(widths.tolist())

    return float(2.0 * np.median(widths_px))


def estimate_sigma_k_initial(
    master_seed: int, min_accepted: int = _PRESCREEN_MIN_ACCEPTED
) -> float:
    """独立估计 D2 初始模糊核宽度 ``σ_K = 2 × median(w_fine)``（30 [S12] C3）。

    与 ``sample_parameters(sigma_K=None)`` 内部的预筛同口径：以由
    ``master_seed`` 经 ``SeedSequence`` 派生的独立随机流抽取 W1–W7 通过
    候选并取逐样本 ``w_fine`` 中位数的两倍（像素单位）。数据集构建方用它
    一次性确定各划分共享的 ``σ_K``，保证划分间一致。
    """
    rng = np.random.default_rng(np.random.SeedSequence(int(master_seed)))
    stats: dict = {
        "mask_version": MASK_VERSION,
        "master_seed": int(master_seed),
        "n_candidates": 0,
        "per_mask_rejected": {name: 0 for name in MASK_NAMES},
        "w1_w7_passers": 0,
    }
    return _prescreen_sigma_k(rng, stats, min_accepted=min_accepted)


def sample_ood_parameters(
    n: int,
    master_seed: int,
    sigma_K: float,
    scale: float = 1.5,
    max_candidates: int = 50_000,
) -> tuple[list[dict[str, float]], dict]:
    """EXP-06 极端参数采样（80 [S7]）：β/γ 幅度放大 ``scale`` 倍、豁免掩膜。

    采样范围：``|β| ∈ [0.9, 2.0]×scale``、``|γ| ∈ [0.1, 0.6]×scale``、
    ``γ = −sign(β)·|γ|``，其余参数沿用 20 [S9] 范围；不施加 W1–W8 拒绝
    （放大后的 β/γ 必被 W7/W8 拒绝，批次二十 Q10 豁免），仅逐样本记录
    掩膜通过率（80 [S7]：豁免后单独记录）。

    返回 ``(samples, stats)``；``samples`` 全部入选（``acceptance_rate=1``），
    ``stats`` 含逐掩膜通过率与 W8 比例（诊断用）。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if scale <= 0:
        raise ValueError("scale 必须为正数")
    rng = np.random.default_rng(np.random.SeedSequence(int(master_seed)))

    stats: dict = {
        "mask_version": MASK_VERSION,
        "master_seed": int(master_seed),
        "sigma_K_px": float(sigma_K),
        "ood_scale": float(scale),
        "n_requested": int(n),
        "n_candidates": 0,
        "masks_exempt": True,
        "per_mask_rejected": {name: 0 for name in MASK_NAMES},
        "w1_w7_passers": 0,
    }

    drawn = 0
    collected: list[np.ndarray] = []
    w8_passed_among_w1w7 = 0
    while drawn < n:
        draw = min(max_candidates, n - drawn + 4096)
        draw = min(draw, max_candidates - stats["n_candidates"])
        if draw <= 0:
            raise RuntimeError(
                f"候选上限 {max_candidates} 内无法抽取 {n} 组 OOD 参数"
            )
        stats["n_candidates"] += draw
        cand = _draw_candidates(draw, rng, ood_scale=scale)
        masks = apply_masks(cand, sigma_K)
        passed_w1_w7 = np.ones(draw, dtype=bool)
        for name in MASK_NAMES[:-1]:
            ok = np.asarray(masks[name], dtype=bool)
            stats["per_mask_rejected"][name] += int((~ok).sum())
            passed_w1_w7 &= ok
        stats["w1_w7_passers"] += int(passed_w1_w7.sum())
        ok_w8 = np.asarray(masks["W8"], dtype=bool)
        stats["per_mask_rejected"]["W8"] += int((passed_w1_w7 & ~ok_w8).sum())
        w8_passed_among_w1w7 += int((passed_w1_w7 & ok_w8).sum())
        # 豁免掩膜：全部候选入选
        for row in range(draw):
            collected.append(np.array([cand[key][row] for key in PARAMETER_KEYS]))
        drawn += draw

    stats["w8_fraction_among_w1_w7_passers"] = (
        w8_passed_among_w1w7 / stats["w1_w7_passers"]
        if stats["w1_w7_passers"] > 0
        else float("nan")
    )
    stats["n_accepted"] = len(collected)
    stats["acceptance_rate"] = 1.0

    matrix = np.vstack(collected[:n])
    samples = []
    for i in range(n):
        sample = {key: float(matrix[i, j]) for j, key in enumerate(PARAMETER_KEYS)}
        sample["compression_state"] = ("under", "optimal", "over")[
            _state_of(np.array([sample["C"]]))[0]
        ]
        samples.append(sample)
    return samples, stats
