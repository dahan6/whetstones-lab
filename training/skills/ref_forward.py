"""纯 numpy 定序参考前向（红线3/门5 的验证基准）。

与 bee/src/inference.rs 严格同一循环顺序：
  - 定序累加，禁 BLAS 快路径（显式 for 循环）
  - 全程 float32 累加，与 Rust f32 一致
  - mean⊕max 双池化（与 Encoder 一致）

用法:
  from training.skills.ref_forward import encoder_forward
  state = encoder_forward(weights_dict, token_ids)  # 与 bee 输出比对
"""
from __future__ import annotations

import numpy as np

WINDOW = 16
N_TOKENS = 7
EVENT_DIM = 48
STATE_DIM = 96


def _linear_t(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PyTorch Linear 布局 [out, in]，定序累加（与 Rust linear_t 同序）。"""
    n_out, n_in = w.shape
    out = np.zeros(n_out, dtype=np.float32)
    for o in range(n_out):
        acc = np.float32(b[o])
        for i in range(n_in):
            acc = np.float32(acc + np.float32(x[i] * w[o, i]))
        out[o] = acc
    return out


def encoder_forward(weights: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    """112 token ids → state(96)，与 Rust Encoder::forward 同序同算术。"""
    assert len(tokens) == WINDOW * N_TOKENS
    embed = weights["embed.weight"]
    events = []
    for ev in range(WINDOW):
        chunk = tokens[ev * N_TOKENS:(ev + 1) * N_TOKENS]
        x = np.concatenate([embed[t] for t in chunk]).astype(np.float32)
        h = _linear_t(x, weights["event_fc.weight"], weights["event_fc.bias"])
        events.append(np.maximum(h, 0))

    # mean pool
    mean = np.zeros(EVENT_DIM, dtype=np.float32)
    for h in events:
        mean = np.float32(mean + h)
    mean = np.float32(mean / np.float32(WINDOW))
    # max pool
    maxp = np.stack(events).max(axis=0).astype(np.float32)

    pooled = np.concatenate([mean, maxp])
    state = _linear_t(pooled, weights["pool_fc.weight"], weights["pool_fc.bias"])
    return np.maximum(state, 0)
