"""L1 集成：[S5] C4 的**跨进程**端到端确定性（spec line2/10 [S5] C4；`00` C3/GC2）。

与 `tests/unit/test_line2_dataset.py::test_build_dataset_deterministic_and_traceable`
的分工：该测试在**同进程**内构建两次后比对数据级快照；本测试在全新解释器（同一 conda
环境）里以同一 master_seed 重跑全链（真实锚点 → 抖动 → FODO → 分箱 → 子采样 → HDF5），
比对全部数据集逐位内容 + 全部 attrs + manifest 正文——进程级全局态、哈希种子、OCELOT
初始化差异等因素造成的偏移只在此暴露。摘要函数自身另有鉴别力自检（改一个计数单元即
改变摘要），避免"摘要忽略内容"导致的假通过。
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.line2.data_pipeline.anchor import read_anchor_particles
from src.line2.data_pipeline.dataset import build_dataset

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANCHOR_COPY = REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
SPLITS = ("train", "val", "test")
#: 小批量（1 束 ≈ 5 s）：Claim 是逐位确定性，规模不影响有效性
N_BEAMS = 2
DATA_VERSION = "xproc_contract"

_CHILD_SCRIPT = (
    "import sys; sys.path.insert(0, {root!r})\n"
    "from tests.integration.test_line2_pipeline_contract import build_and_digest\n"
    "print(build_and_digest(sys.argv[1]))\n"
)


def _digest(root: pathlib.Path) -> str:
    """全量摘要：三个切分文件的每个节点名 + 全部 attrs + 数据集逐位内容 + manifest 正文。"""
    digest = hashlib.sha256()
    for split in SPLITS:
        with h5py.File(root / f"{split}.h5", "r") as h5:

            def visit(name: str, obj: h5py.HLObject) -> None:
                digest.update(name.encode())
                for key in sorted(obj.attrs):
                    digest.update(f"\0{key}\0{obj.attrs[key]!r}".encode())
                if isinstance(obj, h5py.Dataset):
                    digest.update(np.ascontiguousarray(obj[...]).tobytes())

            h5.visititems(visit)
    digest.update((root / "manifest.json").read_bytes())
    return digest.hexdigest()


def build_and_digest(out_dir: str) -> str:
    """真实锚点 → 全链构建 → 摘要（子进程导入本函数用同一代码重跑）。"""
    anchor = read_anchor_particles(ANCHOR_COPY)
    build_dataset(out_dir, anchor, n_beams=N_BEAMS, data_version=DATA_VERSION)
    return _digest(pathlib.Path(out_dir))


@pytest.fixture(scope="module")
def parent_dataset(tmp_path_factory) -> pathlib.Path:
    """模块级：本进程构建一次数据集，供跨进程比对与摘要鉴别力自检共享。"""
    out = tmp_path_factory.mktemp("parent") / "ds"
    build_dataset(out, read_anchor_particles(ANCHOR_COPY), n_beams=N_BEAMS, data_version=DATA_VERSION)
    return out


def test_pipeline_cross_process_bitwise(parent_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """[S5] C4：子进程重跑全链，数据集逐位 + attrs + manifest 与父进程完全一致。"""
    child_dir = tmp_path / "child"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT.format(root=str(REPO_ROOT)), str(child_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"子进程失败：{proc.stderr[-2000:]}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"子进程无输出：{proc.stderr[-2000:]}"
    assert lines[-1] == _digest(parent_dataset), "跨进程数据集内容不一致"
    assert (parent_dataset / "manifest.json").read_bytes() == (
        child_dir / "manifest.json"
    ).read_bytes()
    assert sorted(p.name for p in child_dir.iterdir()) == sorted(
        p.name for p in parent_dataset.iterdir()
    )


def test_pipeline_digest_discriminates_content(
    parent_dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """摘要鉴别力自检：任一计数单元 +1 即改变摘要 ⇒ 上一条的"逐位一致"非恒真。

    另核对该单元确实位于落盘件内（改动生效），否则本自检会在无声中退化为恒真。
    """
    modified = tmp_path / "modified"
    shutil.copytree(parent_dataset, modified)
    with h5py.File(modified / "train.h5", "r+") as h5:
        beam_name = next(iter(h5))
        path = f"{beam_name}/station0/tier_001000_sub_00"
        image = h5[path][...]
        original = int(image[0, 0, 0])
        image[0, 0, 0] = original + 1
        h5[path][...] = image
    # 重新打开确认改动落盘（否则本自检会在无声中退化为恒真）
    with h5py.File(modified / "train.h5", "r") as h5:
        assert int(h5[path][0, 0, 0]) == original + 1
    assert _digest(modified) != _digest(parent_dataset)
