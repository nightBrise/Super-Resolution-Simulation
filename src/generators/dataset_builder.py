"""数据集构建：60 [S8] 划分协议 + 60 [S14] 工件契约。

划分协议（60 [S8] C2/C4）
-------------------------
- 先按内容参数 ``c`` 采样——同一 ``c`` 的噪声实现不跨划分（每个划分用独立
  的 ``SeedSequence`` 分支抽取自己的 ``c`` 与噪声种子，样本级种子只由样本
  索引派生，划分间不存在共享）。
- ``train`` / ``val`` / ``test_id`` 经 γ 块外条件采样：拒绝
  ``|γ| ∈ [0.3, 0.4]`` 的候选。
- ``test_pb`` 经 γ 块内条件采样恰生成与 ``test_id`` 1:1 的样本
  （标准 1,000 / 调试 250）。
- 块边界由固定总体分位数确定（``|γ| ~ U[0.1, 0.6]`` 分位秩 ``[0.4, 0.6]``
  → ``|γ| ∈ [0.3, 0.4]``，即 ``γ ∈ [-0.4,-0.3] ∪ [0.3,0.4]``），
  SHALL NOT 采用经验样本分位数（``GAMMA_BLOCK`` 常量直接由分位公式导出）。
- ``test_ood`` 为 EXP-06 极端参数子集（80 [S7]）：β/γ 幅度放大 1.5 倍、
  豁免 W1–W8 掩膜，规模固定 500。

工件契约（60 [S14]）
--------------------
每划分一个 HDF5 文件：图像 ``H``(256²)/``H_neg_ch``(256², c_high 清零版,
70 [S7.1] C2)/``L``(64²)/``L_clean``(64²)/
``L_up``(256², bilinear)/``P2``(256²) 均 float32、gzip level 4、按样本切分
（chunks=(1, H, W)）；元数据含 ``sample_id``（``<划分>-<序号>``）、全部
内容参数 ``c``（c_low/c_mid/c_high 全字段）、物理标签 ``m``（全集）、导出
物理量（``eps_z``、``I_peak``、能谱剖面 ``S_delta``）、``seed_i``、掩膜
标记 W1–W8、退化配置标记（D2）与退化元数据 ``m_L``。另有 ``manifest.json``
登记主种子、data_version、各划分样本数与 ``sample_id`` 清单、γ 块信息、
mask_revalidation（G3）、标定采用初始值、生成时间戳、code_version（完整
40 位 git commit hash，00 [S6] 约束 8 N4）与 spec_version（``v1.0+<99 最
近批准批次>``）。

写入分工（60 [S14]）：20 写 ``H``/``c``/``m``/物理标签；30 追加
``L``/``L_clean``/``L_up``/``m_L``；40 追加 ``P2``——实现在单条管线内完成，
代码按分工注释组织（``_generate_sample`` 内部按 20 → 30 → 40 顺序）。

种子派生（60 [S14] C4）：样本 ``seed_i`` 由
``SeedSequence(SeedSequence(master_seed).spawn(8)[分支]).spawn(n)[i]`` 的
第 ``i`` 个分支给出；各生成器不自选随机源。并行与串行逐位一致：worker 只
依赖 ``(c, seed_i)``，与处理顺序无关。
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from src.generators.f_beam import C_HIGH_KEYS, C_LOW_KEYS, C_MID_KEYS, f_beam
from src.generators.f_deg import DEFAULT_DOWNSAMPLE, f_deg
from src.generators.f_prior import f_prior
from src.generators.image_ops import normalize_intensity, upsample_4x_bilinear
from src.generators.masks import DELTA_PX, MASK_NAMES, apply_masks, fine_structure_width
from src.generators.sampling import (
    estimate_sigma_k_initial,
    sample_ood_parameters,
    sample_parameters,
)

#: 规格冻结版本（与 00 [S6] 全局约束 8 配套的 spec_version 基础版本）。
SPEC_VERSION = "v1.0"

#: γ 幅度采样范围（20 [S9]：γ = −sign(β)·[0.1, 0.6]）。
GAMMA_MAG_RANGE = (0.1, 0.6)

#: |γ| 的固定总体分位秩（60 [S8] C4：幅度中央 20% 分位带）。
GAMMA_QUANTILE_RANKS = (0.4, 0.6)


def latest_approved_batch_date(change_log: str | Path | None = None) -> str:
    """99 最近已批准批次的日期（00 [S6] 约束 8 N4 的 spec_version 批次标识）。

    从 ``99_change_log.md`` 变更日志表提取 Status 为 ``Approved*``
    （含 ``Approved-PendingTests``）或 ``Implemented`` 的行，取日期最大的
    Date 字段（ISO 格式可直接排序）。例如 2026-08-26 P0 报批包 →
    ``"2026-08-26"``。解析失败时返回空串。
    """
    if change_log is None:
        change_log = (
            Path(__file__).resolve().parents[2] / "docs" / "specs" / "99_change_log.md"
        )
    text = Path(change_log).read_text(encoding="utf-8")
    dates: list[str] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        status = cells[7]  # 表列为 Date/Version/Module/Type/Description/Reason/Impact/Status
        if status.startswith("Approved") or status == "Implemented":
            date = cells[0]
            if len(date) == 10 and date.startswith("20"):
                dates.append(date)
    return max(dates) if dates else ""


def resolve_spec_version(change_log: str | Path | None = None) -> str:
    """spec_version = ``<冻结版本>+<最新 99 批准批次>``（00 [S6] 约束 8 N4）。

    语义为「冻结版本文本 + 截至该批次的全部已批准变更」；持久化载体
    （manifest、config.yaml、summary.json、训练日志）写入的 spec_version
    SHALL 与生成时刻 99 最新已批准批次一致。解析失败时回退基础版本。
    """
    batch = latest_approved_batch_date(change_log)
    return f"{SPEC_VERSION}+{batch}" if batch else SPEC_VERSION


def gamma_block_interval() -> tuple[float, float]:
    """由固定总体分位数导出 γ 块区间（60 [S8] C4，SHALL NOT 用经验分位数）。

    ``|γ| ~ U[0.1, 0.6]`` 的分位秩 ``[0.4, 0.6]`` →
    ``|γ| ∈ [0.1+0.4×0.5, 0.1+0.6×0.5] = [0.3, 0.4]``。
    """
    lo, hi = GAMMA_MAG_RANGE
    q_lo, q_hi = GAMMA_QUANTILE_RANKS
    width = hi - lo
    return (round(lo + q_lo * width, 6), round(lo + q_hi * width, 6))


#: γ 块区间常量（|γ| ∈ [0.3, 0.4]；测试直接引用本常量）。
GAMMA_BLOCK: tuple[float, float] = gamma_block_interval()

#: 划分定义：名称 → (样本数配置键, γ 块模式, 是否施加掩膜)。
#: test_ood 规模固定 500（80 [S7]：EXP-05/06 测试集规模各 500 样本）。
SPLIT_BLOCK_MODE: dict[str, str | None] = {
    "train": "outside",
    "val": "outside",
    "test_id": "outside",
    "test_pb": "inside",
    "test_ood": None,
}

#: 划分 → SeedSequence 分支索引（manifest 登记，样本种子派生用）。
SPLIT_BRANCH: dict[str, int] = {
    "train": 0,
    "val": 1,
    "test_id": 2,
    "test_pb": 3,
    "test_ood": 4,
}

#: 标定初始值估计的分支索引（与划分分支同源，共 8 个分支）。
BRANCH_SIGMA_K = 5
BRANCH_SIGMA_N = 6
BRANCH_PROBE = 7

#: M1 已确立的初始值口径：σ_n 取「尾部信噪比 2 档」（mean(L_clean)/2，
#: 30 [S9] C3 的 SNR 定义；M1 验收 test_m1 的 INITIAL_TAIL_SNR=2.0）。
INITIAL_TAIL_SNR = 2.0

#: M1 采用的先验平滑倍数：σ_smooth,P = 2.6×σ_smooth,H（40 [S5] C5 候选上档）。
#: 注：2026-08-26 P0 修订（σ_smooth,H = 0.125×w_fine）后，该 2.6× 仅为
#: 管线默认值——40 [S5] C5/AC14 出口流程已按扩展候选集取 15×（见
#: tests/acceptance/test_m1_generators.py）；数据集重建（EXP-01d 标定登记）
#: 时按登记值更新本常量并递增数据版本。
SIGMA_SMOOTH_P_MULTIPLE = 15.0  # EXP-01d 采用值（2026-08-26）：AC14 L1 比值 0.5288 最接近 0.55 锚点，预授权扩展后选定

#: EXP-06 极端参数放大倍数（80 [S7]：相对训练集采样范围上界放大 50%）。
OOD_SCALE = 1.5

#: test_ood 固定规模（80 [S7]）。
DEFAULT_OOD_SIZE = 500

#: HDF5 压缩：gzip level 4、按样本切分（60 [S15] C2）。
GZIP_LEVEL = 4

#: 生成批大小（逐批生成写 HDF5，控制内存；20k 样本不一次性加载）。
BATCH_SIZE = 256

#: 并行 worker 数默认值（本机 CPU 112 线程，80 [S3] 建议 16–32）。
DEFAULT_WORKERS = 32

#: 图像字段（float32、chunks=(1, H, W)）。
#: ``H_neg_ch`` 为 c_high（a₃/γ/b₁）清零版真值（70 [S7.1] C2 掩膜能量成分
#: 分解与先验泄漏指数 Π_leak 的差分基准，2026-08-26 P0 报批包第 10 项）。
IMAGE_KEYS: tuple[str, tuple[int, int], ...] = (
    ("H", (256, 256)),
    ("H_neg_ch", (256, 256)),
    ("L", (64, 64)),
    ("L_clean", (64, 64)),
    ("L_up", (256, 256)),
    ("P2", (256, 256)),
)

#: m 标量物理标签（导出物理量 eps_z / I_peak 包含其中，60 [S14]）。
#: Level 为整数标签，单独以 int64 落盘，不在此列。
M_SCALAR_KEYS: tuple[str, ...] = (
    "Q",
    "mu_z",
    "mu_delta",
    "sigma_z",
    "sigma_delta",
    "C_zdelta",
    "h_eff",
    "eps_z",
    "I_peak",
    "compression_factor",
)

#: m 字符串标签。
M_STRING_KEYS: tuple[str, ...] = ("compression_state",)

#: m 剖面（导出物理量：电流剖面 I(z)、能谱剖面 S(δ)）。
M_PROFILE_KEYS: tuple[str, ...] = ("I_z", "S_delta")

#: m_L 退化元数据字段（30 [S9] C1）。
M_L_SCALAR_KEYS: tuple[str, ...] = ("r", "sigma_K", "sigma_n", "SNR", "seed")
M_L_STRING_KEYS: tuple[str, ...] = ("noise_model", "degradation_order")


def derive_sample_seed(master_seed: int, split: str, index: int, split_size: int) -> int:
    """按 60 [S14] C4 派生样本 ``seed_i``（``<划分>-<序号>`` 的第 i 个分支）。

    ``split_size`` 为该划分的样本总数，用于 ``spawn`` 的分支数；同一
    ``(master_seed, split, index, split_size)`` 恒返回同一种子，与生成顺序
    无关（并行/串行逐位一致的基础）。
    """
    branch = np.random.SeedSequence(int(master_seed)).spawn(8)[SPLIT_BRANCH[split]]
    child = branch.spawn(int(split_size))[index]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def estimate_sigma_n_initial(
    master_seed: int, sigma_K: float, n: int = 256
) -> float:
    """估计 D2 初始全局噪声常数（30 [S12] C4 初始口径）。

    取使「逐样本全局 SNR = mean(L_clean)/σ_n」批量中位数为 2.0 的全局常数
    （30 [S9] C3 的 SNR 定义；M1 的 INITIAL_TAIL_SNR=2.0 口径）。σ_n 越大
    高频信噪比越低，故取登记带（2–5）下档 2.0 与 M1 一致。
    """
    params, _ = sample_parameters(
        n,
        master_seed=int(np.random.SeedSequence(master_seed).spawn(8)[BRANCH_SIGMA_N].generate_state(1, dtype=np.uint32)[0]),
        sigma_K=sigma_K,
        gamma_block=("outside", *GAMMA_BLOCK),
    )
    means = []
    for c in params:
        sigma_smooth_h = 0.125 * float(fine_structure_width(c) / DELTA_PX)
        H, _, _ = f_beam(c, sigma_smooth=sigma_smooth_h)
        L_clean = f_deg(H, sigma_K=sigma_K, sigma_n=0.0, seed=0)[1]
        means.append(float(L_clean.mean()))
    return float(np.median(means) / INITIAL_TAIL_SNR)


def _generate_sample(args: tuple) -> dict[str, Any]:
    """生成单个样本的全部字段（并行 worker 的可 pickle 顶层函数）。

    写入分工（60 [S14]）：20 写 ``H``/``c``/``m``/物理标签；30 追加
    ``L``/``L_clean``/``L_up``/``m_L``；40 追加 ``P2``。
    随机性只来自 ``seed_i``（f_deg 内部经 SeedSequence 派生），其余生成函数
    为确定性函数；worker 不接触任何全局随机源（60 [S14] C4）。
    """
    split, index, c, seed_i, sigma_K, sigma_n, deg_level = args

    sigma_smooth_h = 0.125 * float(fine_structure_width(c) / DELTA_PX)

    # ---- 20：H、物理标签 m、内容参数 c（含导出物理量） ----
    H, m, c_rec = f_beam(c, sigma_smooth=sigma_smooth_h)

    # 20：H_neg_ch = c_high 清零版真值（a₃=γ=b₁=0，70 [S7.1] C2 差分基准；
    # 与 H 同 σ_smooth,H、同网格，仅供评估端成分分解/Π_leak 使用）。
    c_neg_ch = dict(c)
    for key in C_HIGH_KEYS:
        c_neg_ch[key] = 0.0
    H_neg_ch, _, _ = f_beam(c_neg_ch, sigma_smooth=sigma_smooth_h)

    # ---- 30：退化 L / L_clean / L_up 与退化元数据 m_L ----
    L, L_clean, d, m_L = f_deg(
        H, sigma_K=sigma_K, sigma_n=sigma_n, r=DEFAULT_DOWNSAMPLE, seed=int(seed_i)
    )
    L_up = normalize_intensity(upsample_4x_bilinear(L))

    # ---- 40：图像先验 P2 ----
    P2, meta = f_prior(
        c, level="P2", sigma_smooth=SIGMA_SMOOTH_P_MULTIPLE * sigma_smooth_h
    )

    masks = apply_masks(c, sigma_K)

    return {
        "split": split,
        "index": index,
        "c": c_rec,
        "H": H,
        "H_neg_ch": H_neg_ch,
        "m": m,
        "L": L,
        "L_clean": L_clean,
        "L_up": L_up,
        "P2": P2,
        "m_L": m_L,
        "d": d,
        "seed_i": int(seed_i),
        "masks": masks,
        "meta": meta,
        "deg_level": deg_level,
    }


def _generate_batch(task_args: list, workers: int) -> list[dict]:
    """以 ``workers`` 个进程（或串行）生成一批样本，串/并行逐位一致。"""
    if workers <= 1:
        return [_generate_sample(a) for a in task_args]
    with multiprocessing.Pool(workers) as pool:
        return pool.map(_generate_sample, task_args, chunksize=1)


def _create_split_file(
    path: Path,
    split: str,
    n: int,
    version: str,
    master_seed: int,
    code_version: str,
) -> h5py.File:
    """创建划分 HDF5 文件的骨架（预分配数据集，60 [S14] 字段契约）。"""
    f = h5py.File(str(path), "w")
    f.attrs["split"] = split
    f.attrs["data_version"] = version
    f.attrs["spec_version"] = resolve_spec_version()
    f.attrs["code_version"] = code_version
    f.attrs["master_seed"] = int(master_seed)
    f.attrs["has_h_neg_ch"] = True
    f.attrs["gzip_level"] = GZIP_LEVEL
    f.attrs["compression"] = "gzip"
    f.attrs["chunking"] = "per-sample"

    str_dtype = h5py.string_dtype("utf-8")
    chunk_n = min(4096, max(1, n))

    for name, shape in IMAGE_KEYS:
        f.create_dataset(
            name,
            shape=(n, *shape),
            dtype="float32",
            chunks=(1, *shape),
            compression="gzip",
            compression_opts=GZIP_LEVEL,
        )

    f.create_dataset("sample_id", (n,), dtype=str_dtype, chunks=(chunk_n,))

    for group_name, keys in (
        ("c_low", C_LOW_KEYS),
        ("c_mid", C_MID_KEYS),
        ("c_high", C_HIGH_KEYS),
    ):
        g = f.create_group(group_name)
        for key in keys:
            g.create_dataset(
                key, (n,), dtype="float64", chunks=(chunk_n,), compression="gzip",
                compression_opts=GZIP_LEVEL,
            )
    f.create_dataset(
        "c/C", (n,), dtype="float64", chunks=(chunk_n,), compression="gzip",
        compression_opts=GZIP_LEVEL,
    )

    m = f.create_group("m")
    for key in M_SCALAR_KEYS:
        m.create_dataset(
            key, (n,), dtype="float64", chunks=(chunk_n,), compression="gzip",
            compression_opts=GZIP_LEVEL,
        )
    m.create_dataset(
        "Level", (n,), dtype="int64", chunks=(chunk_n,), compression="gzip",
        compression_opts=GZIP_LEVEL,
    )
    for key in M_STRING_KEYS:
        m.create_dataset(key, (n,), dtype=str_dtype, chunks=(chunk_n,))
    for key in M_PROFILE_KEYS:
        m.create_dataset(
            key, (n, 256), dtype="float64", chunks=(1, 256), compression="gzip",
            compression_opts=GZIP_LEVEL,
        )

    f.create_dataset(
        "seed_i", (n,), dtype="int64", chunks=(chunk_n,), compression="gzip",
        compression_opts=GZIP_LEVEL,
    )

    masks = f.create_group("masks")
    for name in MASK_NAMES:
        masks.create_dataset(
            name, (n,), dtype="uint8", chunks=(chunk_n,), compression="gzip",
            compression_opts=GZIP_LEVEL,
        )

    f.create_dataset("deg_level", (n,), dtype=str_dtype, chunks=(chunk_n,))
    m_L = f.create_group("m_L")
    for key in M_L_SCALAR_KEYS:
        m_L.create_dataset(
            key, (n,), dtype="float64", chunks=(chunk_n,), compression="gzip",
            compression_opts=GZIP_LEVEL,
        )
    for key in M_L_STRING_KEYS:
        m_L.create_dataset(key, (n,), dtype=str_dtype, chunks=(chunk_n,))
    return f


def _write_batch(h5: h5py.File, records: list[dict], start: int) -> None:
    """把一批样本写入 HDF5 预分配数据集（按样本切分的切片写入）。"""
    n = len(records)
    for name, _shape in IMAGE_KEYS:
        arr = np.stack([r[name] for r in records]).astype("float32")
        h5[name][start : start + n] = arr

    h5["sample_id"][start : start + n] = np.array(
        [r["_sample_id"] for r in records], dtype=h5["sample_id"].dtype
    )

    for group_name in ("c_low", "c_mid", "c_high"):
        g = h5[group_name]
        for key in g.keys():
            arr = np.array([r["c"][key] for r in records], dtype="float64")
            g[key][start : start + n] = arr
    h5["c/C"][start : start + n] = np.array(
        [r["c"]["C"] for r in records], dtype="float64"
    )

    m = h5["m"]
    for key in M_SCALAR_KEYS:
        arr = np.array([r["m"][key] for r in records], dtype="float64")
        m[key][start : start + n] = arr
    m["Level"][start : start + n] = np.array(
        [int(r["m"]["Level"]) for r in records], dtype="int64"
    )
    for key in M_STRING_KEYS:
        m[key][start : start + n] = np.array(
            [r["m"][key] for r in records], dtype=m[key].dtype
        )
    for key in M_PROFILE_KEYS:
        arr = np.stack([np.asarray(r["m"][key], dtype="float64") for r in records])
        m[key][start : start + n] = arr

    h5["seed_i"][start : start + n] = np.array(
        [r["seed_i"] for r in records], dtype="int64"
    )

    for name in MASK_NAMES:
        h5["masks"][name][start : start + n] = np.array(
            [1 if r["masks"][name] else 0 for r in records], dtype="uint8"
        )

    h5["deg_level"][start : start + n] = np.array(
        [r["deg_level"] for r in records], dtype=h5["deg_level"].dtype
    )
    m_L = h5["m_L"]
    for key in M_L_SCALAR_KEYS:
        arr = np.array([r["m_L"][key] for r in records], dtype="float64")
        m_L[key][start : start + n] = arr
    for key in M_L_STRING_KEYS:
        m_L[key][start : start + n] = np.array(
            [r["m_L"][key] for r in records], dtype=m_L[key].dtype
        )


def _mask_revalidation(
    split: str, params: list[dict], stats: dict
) -> dict[str, Any]:
    """G3 掩膜复核（00 [S6] 约束 8 / 60 [S8] C4，2026-08-26 P0 报批包第 3 项）。

    生成期复核 γ 块总体分位数、W8 覆盖率与 γ 块内外计数，给出
    ``revalidation_verdict``（pass / drift）：

    - γ 块总体分位数复核：块边界 ``|γ| ∈ [0.3, 0.4]``（总体分位秩
      ``[0.4, 0.6]``）在样本分布上的经验分位秩，与块模式期望值比对
      （outside 模式条件采样后 ``|γ| ~ U[0.1,0.3) ∪ (0.4,0.6]`` →
      期望 ``(0.5, 0.5)``；inside 模式全部落块内 → 期望 ``(0.0, 1.0)``；
      test_ood 豁免不做分位比对）；
    - W8 覆盖率：``w8_fraction_among_w1_w7_passers ≥ 0.6``（20 [S9] C9，
      OOD 豁免不计）；
    - γ 块内外计数：样本的块内/块外计数。

    复核不改变任何样本（纯记录性诊断，G3 提示用）；漂移仅触发 99 登记，
    不影响数据本身。
    """
    gmag = np.abs(np.array([float(c["gamma"]) for c in params]))
    lo, hi = GAMMA_BLOCK
    n_inside = int(((gmag >= lo) & (gmag <= hi)).sum())
    n_outside = int(len(gmag) - n_inside)
    q_lo = float((gmag < lo).mean())
    q_hi = float((gmag <= hi).mean())

    w8_frac = stats.get("w8_fraction_among_w1_w7_passers")
    w8_frac = (
        float(w8_frac)
        if isinstance(w8_frac, (int, float)) and w8_frac == w8_frac
        else None
    )

    mode = SPLIT_BLOCK_MODE.get(split)
    if mode == "inside":
        expected = (0.0, 1.0)
        quantile_checks = [abs(q_lo - 0.0) <= 0.05, abs(q_hi - 1.0) <= 0.05]
    elif mode == "outside":
        expected = (0.5, 0.5)
        quantile_checks = [abs(q_lo - 0.5) <= 0.05, abs(q_hi - 0.5) <= 0.05]
    else:  # test_ood：掩膜豁免，分位复核不计
        expected = None
        quantile_checks = []

    w8_checks = [] if mode is None else [w8_frac is not None and w8_frac >= 0.6]
    verdict = "pass" if all(quantile_checks) and all(w8_checks) else "drift"
    return {
        "split": split,
        "gamma_block_quantile_ranks": {
            "observed": [q_lo, q_hi],
            "expected_by_mode": expected,
        },
        "w8_fraction_among_w1_w7_passers": w8_frac,
        "gamma_block_counts": {"inside": n_inside, "outside": n_outside},
        "revalidation_verdict": verdict,
    }


def build_split(
    split: str,
    n: int,
    out_dir: Path,
    version: str,
    master_seed: int,
    sigma_K: float,
    sigma_n: float,
    workers: int,
    code_version: str,
    deg_level: str = "D2",
    ood_scale: float = OOD_SCALE,
    max_candidates: int = 200_000,
) -> dict:
    """生成单个划分：采样 c → 派生种子 → 并行生成 → HDF5 落盘。

    返回 manifest 用的划分节（样本数、sample_id 清单与掩膜统计）。
    """
    split_seed = int(
        np.random.SeedSequence(master_seed).spawn(8)[SPLIT_BRANCH[split]]
        .generate_state(1, dtype=np.uint32)[0]
    )

    mode = SPLIT_BLOCK_MODE[split]
    if split == "test_ood":
        params, stats = sample_ood_parameters(
            n, master_seed=split_seed, sigma_K=sigma_K, scale=ood_scale,
            max_candidates=max_candidates,
        )
    elif mode is None:
        params, stats = sample_parameters(
            n, master_seed=split_seed, sigma_K=sigma_K, max_candidates=max_candidates
        )
    else:
        params, stats = sample_parameters(
            n,
            master_seed=split_seed,
            sigma_K=sigma_K,
            max_candidates=max_candidates,
            gamma_block=(mode, *GAMMA_BLOCK),
        )

    width = max(3, len(str(n - 1)))
    sample_ids = [f"{split}-{i:0{width}d}" for i in range(n)]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.h5"
    h5 = _create_split_file(path, split, n, version, master_seed, code_version)

    seed_seq = np.random.SeedSequence(master_seed).spawn(8)[SPLIT_BRANCH[split]]
    sample_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_seq.spawn(n)
    ]

    try:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            task_args = [
                (
                    split,
                    i,
                    params[i],
                    sample_seeds[i],
                    sigma_K,
                    sigma_n,
                    deg_level,
                )
                for i in range(start, end)
            ]
            records = _generate_batch(task_args, workers)
            for rec, sid in zip(records, sample_ids[start:end]):
                rec["_sample_id"] = sid
            _write_batch(h5, records, start)
    finally:
        h5.close()

    return {
        "count": int(n),
        "sample_ids": sample_ids,
        "mask_stats": stats,
        "mask_revalidation": _mask_revalidation(split, params, stats),
    }


def git_head() -> str:
    """当前 git commit hash（code_version；git 不可用时回退标记）。

    SHALL 为完整 40 位 hash（00 [S6] 约束 8 N4：7 位短 hash 仅可用于展示，
    持久化载体一律使用完整 hash）；``git rev-parse HEAD`` 输出非 40 位时
    回退 ``"working-tree"`` 标记。
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=10,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            if len(sha) == 40:
                return sha
    except Exception:
        pass
    return "working-tree"


def load_config(path: str | Path) -> dict:
    """读取 ``config.yaml`` 并回填缺失的版本三元组（60 [S15] C11）。"""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config.setdefault("code_version", git_head())
    config.setdefault("spec_version", resolve_spec_version())
    return config


def build_dataset(
    config: dict,
    project_root: Path,
    splits: list[str] | None = None,
    workers: int | None = None,
    code_version: str | None = None,
) -> dict:
    """按配置构建数据集（60 [S8] + [S14]）。

    参数
    ----
    config: 配置字典（master_seed、dataset.version/各划分样本数、
        calibration.sigma_K/sigma_n、degradation.level 等；模板结构见
        ``config.yaml.template``）。
    project_root: 项目根目录；数据落盘于 ``<project_root>/data/<version>/``。
    splits: 只构建的划分列表；``None`` 表示全部（train/val/test_id/test_pb/
        test_ood）。
    workers: 并行进程数；``None`` 用配置 ``dataset.workers`` 或默认值。
    code_version: 覆盖配置中的 code_version（git commit hash）。

    返回
    ----
    manifest 字典（同时落盘 ``manifest.json``；部分构建不重写 manifest）。
    """
    version = str(config["dataset"]["version"])
    master_seed = int(config["master_seed"])
    # N4 版本对账：code_version 以生成时 git HEAD（完整 40 位）为准，
    # config 中的旧/占位值不得覆盖（00 [S6] 约束 8）
    code_version = code_version or git_head()

    calibration = config.get("calibration", {})
    sigma_K = float(calibration.get("sigma_K") or 0.0)
    sigma_n = float(calibration.get("sigma_n") or 0.0)
    if sigma_K <= 0.0:
        sigma_K = estimate_sigma_k_initial(
            int(np.random.SeedSequence(master_seed).spawn(8)[BRANCH_SIGMA_K]
                .generate_state(1, dtype=np.uint32)[0])
        )
    if sigma_n <= 0.0:
        sigma_n = estimate_sigma_n_initial(master_seed, sigma_K)

    if workers is None:
        workers = int(config.get("dataset", {}).get("workers") or DEFAULT_WORKERS)

    ds = config["dataset"]
    train_n = int(ds["train_size"])
    val_n = int(ds["val_size"])
    test_id_n = int(ds["test_id_size"])
    test_pb_n = int(ds.get("test_pb_size") or test_id_n)
    if test_pb_n != test_id_n:
        raise ValueError(
            f"test_pb 与 test_id 必须 1:1（60 [S8] C4）：{test_pb_n} vs {test_id_n}"
        )
    ood_n = int(ds.get("test_ood_size") or DEFAULT_OOD_SIZE)

    sizes = {
        "train": train_n,
        "val": val_n,
        "test_id": test_id_n,
        "test_pb": test_pb_n,
        "test_ood": ood_n,
    }
    block_cfg = ds.get("test_pb_block", {})
    block_range = [float(block_cfg["range"][0]), float(block_cfg["range"][1])]
    if not np.allclose(block_range, GAMMA_BLOCK, atol=1e-6):
        raise ValueError(
            f"配置 γ 块区间 {block_range} 与固定总体分位数导出值 "
            f"{list(GAMMA_BLOCK)} 不一致（60 [S8] C4）"
        )

    if splits is None:
        splits = list(sizes)
    for split in splits:
        if split not in sizes:
            raise ValueError(f"未知划分：{split}")

    deg_level = str(config.get("degradation", {}).get("level") or "D2")
    # 研究线根（方案 B）：config `study_root` 非空时落盘于
    # `<project_root>/studies/<study_root>/data/<version>/`，为空/缺省时兜底
    # `<project_root>/data/<version>/`（向后兼容）。
    study_root = str(config.get("study_root", "")).strip()
    data_root = (project_root / "studies" / study_root / "data") if study_root else (project_root / "data")
    out_dir = data_root / version

    split_sections: dict[str, dict] = {}
    for split in splits:
        split_sections[split] = build_split(
            split,
            sizes[split],
            out_dir,
            version,
            master_seed,
            sigma_K,
            sigma_n,
            workers,
            code_version,
            deg_level=deg_level,
        )

    if set(splits) == set(sizes):
        manifest: dict[str, Any] = {
            "data_version": version,
            # N4（00 [S6] 约束 8）：载体写入的 spec_version SHALL 与生成时刻
            # 99 change log 的最新已批准批次一致（不随 config 覆盖）。
            "spec_version": resolve_spec_version(),
            "code_version": code_version,
            "master_seed": master_seed,
            "has_h_neg_ch": True,
            "mask_revalidation": {
                split: split_sections[split]["mask_revalidation"]
                for split in sizes
            },
            "revalidation_verdicts": {
                split: split_sections[split]["mask_revalidation"]["revalidation_verdict"]
                for split in sizes
            },
            "seed_derivation": (
                "sample_seed_i = SeedSequence(master_seed).spawn(8)[split_branch]"
                ".spawn(n)[i].generate_state(1, uint32)"
            ),
            "split_branches": SPLIT_BRANCH,
            "gamma_block": {
                "dimension": "abs(gamma)",
                "interval": list(GAMMA_BLOCK),
                "signed_interval": [
                    [-GAMMA_BLOCK[1], -GAMMA_BLOCK[0]],
                    [GAMMA_BLOCK[0], GAMMA_BLOCK[1]],
                ],
                "population_quantile": {
                    "distribution": "|gamma| ~ U[0.1, 0.6]",
                    "ranks": list(GAMMA_QUANTILE_RANKS),
                    "derivation": (
                        f"lo = 0.1 + {GAMMA_QUANTILE_RANKS[0]}*(0.6-0.1), "
                        f"hi = 0.1 + {GAMMA_QUANTILE_RANKS[1]}*(0.6-0.1)"
                    ),
                },
                "mode_per_split": {
                    k: (v if v is not None else "exempt")
                    for k, v in SPLIT_BLOCK_MODE.items()
                },
                "sample_counts": {
                    k: sizes[k] for k in ("train", "val", "test_id", "test_pb")
                },
            },
            "splits": split_sections,
            "calibration": {
                "sigma_K_px": float(sigma_K),
                "sigma_K_rule": "2 x median(w_fine)（M1 验收 fixture 口径 10.37，"
                "为 30 [S6] C8 判据裕度取 11.0，见 B 类登记）",
                "sigma_n": float(sigma_n),
                "sigma_n_rule": (
                    "全局常数：median(mean(L_clean))/2（M1 '尾部信噪比 2 档' "
                    "口径，30 [S9] C3 SNR 定义）"
                ),
                "sigma_smooth_H": {
                    "rule": "0.125 x w_fine，逐样本（20 [S3] C4，2026-08-26 "
                    "P0 修订：原 0.5x 见 99 OQ-20-03）",
                },
                "sigma_smooth_P": {
                    "rule": f"{SIGMA_SMOOTH_P_MULTIPLE} x sigma_smooth_H，逐样本",
                },
            },
            "degradation": {
                "level": deg_level,
                "r": int(config.get("degradation", {}).get("r") or DEFAULT_DOWNSAMPLE),
                "noise_model": "L = max(0, L_clean + n)",
                "snr_hf_threshold": float(
                    config.get("degradation", {}).get("snr_hf_threshold") or 0.1
                ),
            },
            "mask_stats": {
                split: split_sections[split]["mask_stats"] for split in sizes
            },
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "src.generators.dataset_builder",
        }
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return manifest

    # 部分构建：返回部分 manifest 节（不落盘，完整 manifest 由全量构建写）。
    return {"splits": split_sections, "data_version": version}
