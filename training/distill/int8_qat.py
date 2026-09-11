"""int8 量化工具（ES 评估即量化的核心）。

红线（ARCHITECTURE.md §4 红线3 修订版）：决策栈权重 int8，
ES 每个候选先 int8 量化再 int8 前向评估——收敛权重天然适合 int8。
defender 例外（连续 surprise 打分器，保 fp32/per-channel 校准）。
"""
from __future__ import annotations

import numpy as np
import torch


def quantize_tensor_int8(w: torch.Tensor) -> torch.Tensor:
    """单张量 per-tensor int8 量化（scale=max/127，round 到 [-127,127]）。
    返回反量化后的 fp32（量化噪声注入，但保持 fp32 算术——bee 端
    加载 int8 存储，前向时同样反量化到 fp32，两端数值一致）。"""
    s = w.abs().max() / 127.0
    if s == 0:
        return w.clone()
    q = torch.clamp(torch.round(w / s), -127, 127)
    return q * s


def quantize_model_int8(model: torch.nn.Module) -> None:
    """就地 per-tensor int8 量化模型所有参数（评估即量化用）。"""
    with torch.no_grad():
        for p in model.parameters():
            p.data = quantize_tensor_int8(p.data)


def int8_bytes(model: torch.nn.Module) -> int:
    """模型 int8 存储的总字节数（每参数 1 字节 + per-tensor fp32 scale）。"""
    n_params = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in model.parameters())
    return n_params + n_tensors * 4
