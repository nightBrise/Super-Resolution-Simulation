"""Smoke 训练测试（05 [S4] test_smoke_train，L2 单卡 GPU）。

微训练：256 样本（v1 train 子集）、代理变体 C0=24、batch 4、A/B/C 各
100 步、单卡 cuda:0。断言：
1. 三方案 forward/backward 可执行；
2. 损失有限（无 NaN/Inf）；
3. ``mean(loss[后 50 步]) < mean(loss[前 50 步])``（下降趋势，不要求收敛）；
4. Ĥ ≥ 0 且形状 256×256；
5. checkpoint 与 seeds.json 落盘成功；
6. 显存峰值 < 6GB（05 [S2] L2 纪律）。

只断言协议/不变量，不断言研究结果（05 [S1] C1）。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from src.evaluation.evaluate import load_model  # noqa: E402
from src.models.schemes import forward_scheme  # noqa: E402
from src.training.train import train  # noqa: E402
from src.utils.h5data import H5Dataset, collate_samples  # noqa: E402
from tests.smoke.conftest import smoke_config  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.gpu, pytest.mark.m3]

#: 显存峰值上限（05 [S2]：L2 显存峰值 < 6GB）。
GPU_MEM_LIMIT_BYTES = 6 * 1024**3


def test_smoke_train_all_schemes(smoke_device, smoke_indices, tmp_path):
    """A/B/C 各 100 步微训练：可执行、损失下降、非负、落盘、显存受控。"""
    train_idx, _ = smoke_indices
    for scheme in ("A", "B", "C"):
        cfg = smoke_config(scheme)
        out = tmp_path / f"run_{scheme}"
        torch.cuda.reset_peak_memory_stats(0)
        stats = train(cfg, out, train_indices=train_idx, max_steps_override=100)

        # 1. 可执行 + 步数
        assert stats["scheme"] == scheme
        assert stats["steps_run"] == 100

        # 2/3. 损失有限且下降趋势（05 [S4] 断言 2/3）
        curve = np.asarray(stats["train_loss_curve"], dtype=np.float64)
        assert curve.shape == (100,)
        assert np.isfinite(curve).all()
        assert curve[50:].mean() < curve[:50].mean(), f"{scheme} 后 50 步均值未低于前 50 步"

        # 4. 输出非负且 256×256（05 [S4] 断言 4）
        model, _ = load_model(cfg, out / "checkpoints" / "best_val.ckpt", "cpu")
        model.eval()
        ds = H5Dataset("studies/line1_substitute_sr/data/v1/train.h5", "train")
        batch = collate_samples([ds[int(i)] for i in train_idx[:4]])
        with torch.no_grad():
            H_hat = forward_scheme(model, batch, "cpu")
        assert H_hat.shape == (4, 1, 256, 256)
        assert H_hat.min().item() >= 0.0

        # 5. checkpoint 与 seeds.json 落盘（05 [S4] 断言 5）
        assert (out / "checkpoints" / "best_val.ckpt").exists()
        assert (out / "checkpoints" / "last.ckpt").exists()
        assert (out / "seeds.json").exists()
        assert (out / "logs" / "train.log").exists()

        # 6. 显存峰值 < 6GB（05 [S2]）
        peak = torch.cuda.max_memory_allocated(0)
        assert peak < GPU_MEM_LIMIT_BYTES, f"{scheme} 显存峰值 {peak / 1e9:.2f}GB ≥ 6GB"
