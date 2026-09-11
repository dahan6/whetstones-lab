"""共享状态编码器 + 预训练头。

Encoder（部署用）:
  112 = 7 token × 16 embed → Linear(112,48) ReLU → 事件向量
  窗口 W=16 事件向量 → mean pool ⊕ max pool → cat(96)
  → Linear(96,96) ReLU → state(96)

为什么 mean+max 双池化：扫描爆发/高危动作是"少数显著事件"，
纯 mean pool 在 16 事件窗口里把它们稀释到不可见（实测 scan_active
探针 0.73 过不去）。max pool 保留显著事件，mean pool 保留背景节律。

PretrainHeads（训练用脚手架，预训练后丢弃）:
  7 × Linear(96, vocab) — 预测下一事件的 7 个 token
"""
from __future__ import annotations

import torch
import torch.nn as nn

WINDOW = 16
N_TOKENS = 7
TOK_DIM = 16
EVENT_DIM = 48
POOLED_DIM = EVENT_DIM * 2   # mean ⊕ max
STATE_DIM = 96


class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, TOK_DIM)
        self.event_fc = nn.Linear(N_TOKENS * TOK_DIM, EVENT_DIM)
        self.pool_fc = nn.Linear(POOLED_DIM, STATE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W*7) token ids → state: (B, STATE_DIM)"""
        b = x.size(0)
        e = self.embed(x)                          # (B, W*7, 16)
        e = e.view(b, WINDOW, N_TOKENS * TOK_DIM)  # (B, W, 112)
        ev = torch.relu(self.event_fc(e))          # (B, W, 48)
        pooled = torch.cat([ev.mean(dim=1), ev.max(dim=1).values], dim=-1)  # (B, 96)
        return torch.relu(self.pool_fc(pooled))    # (B, 96)


class PretrainModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.heads = nn.ModuleList(
            [nn.Linear(STATE_DIM, vocab_size) for _ in range(N_TOKENS)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        state = self.encoder(x)
        return [h(state) for h in self.heads]      # 7 × (B, vocab)
