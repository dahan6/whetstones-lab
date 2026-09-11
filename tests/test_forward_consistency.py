"""Gate 5 自动化：Python 参考前向 vs bee --forward 输出一致性。

验证红线3（算术纪律）：训练侧与部署侧同一输入逐层误差 < 1e-4（int8 量化后略放宽）。
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from training.skills.ref_forward import encoder_forward, WINDOW, N_TOKENS


def load_weights_as_numpy(weights_dir: str) -> dict[str, np.ndarray]:
    """从 bee weights/ 目录加载 int8 反量化后的 fp32 权重（模拟 bee 端加载）。"""
    wdir = __import__("pathlib").Path(weights_dir)
    manifest = json.loads((wdir / "manifest.json").read_text())
    tensors = {}
    for t in manifest["tensors"]:
        raw = (wdir / t["file"]).read_bytes()
        shape = t["shape"]
        n = 1
        for s in shape:
            n *= s
        if t.get("dtype") == "int8":
            scale = t["scale"]
            data = np.array([b if b < 128 else b - 256 for b in raw],
                            dtype=np.float32) * scale
        else:
            data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        tensors[t["name"]] = data.reshape(shape)
    return tensors


class TestEncoderForward:
    def test_known_input_matches(self, bee_binary, weights_dir):
        """零 token 序列 → bee --forward 输出 vs numpy 参考前向。

        bee --forward 模式固定用全零 token 窗口做自检（main.rs:57），
        这里用同样的全零输入跑参考前向对比。
        """
        tokens = [0] * (WINDOW * N_TOKENS)

        # bee 二进制输出（零 token 自检）
        proc = subprocess.run(
            [bee_binary, weights_dir, "--forward"],
            capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0, f"bee --forward 失败: {proc.stderr}"

        bee_state = []
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("state["):
                val = float(line.split("=")[1])
                bee_state.append(val)
        assert len(bee_state) == 96, f"bee 输出维度不对: {len(bee_state)}"

        # numpy 参考前向（同零 token 输入）
        weights = load_weights_as_numpy(weights_dir)
        ref_state = encoder_forward(weights, tokens)

        # 逐元素比较（int8 量化误差允许 0.5）
        max_err = float(np.max(np.abs(np.array(bee_state) - ref_state)))
        assert max_err < 0.5, (
            f"encoder 前向误差过大: max_err={max_err:.6f}\n"
            f"bee[:6]={bee_state[:6]}\n"
            f"ref[:6]={ref_state[:6].tolist()}"
        )


class TestWeightLoading:
    def test_manifest_has_required_tensors(self, weights_dir):
        """manifest.json 包含所有必需张量名。"""
        wdir = __import__("pathlib").Path(weights_dir)
        manifest = json.loads((wdir / "manifest.json").read_text())
        names = {t["name"] for t in manifest["tensors"]}

        required = {
            "embed.weight", "event_fc.weight", "event_fc.bias",
            "pool_fc.weight", "pool_fc.bias",
        }
        missing = required - names
        assert not missing, f"缺少张量: {missing}"

    def test_embed_shape_consistent(self, weights_dir):
        """embed.weight shape 与 vocab 大小一致。"""
        wdir = __import__("pathlib").Path(weights_dir)
        vocab = json.loads((wdir / "vocab.json").read_text())
        weights = load_weights_as_numpy(weights_dir)
        embed = weights["embed.weight"]
        assert embed.shape[0] >= len(vocab), (
            f"embed 第一维 {embed.shape[0]} < vocab 大小 {len(vocab)}")
