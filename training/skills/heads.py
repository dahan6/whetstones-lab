"""技能头：冻结编码器之上的小 MLP。

结构: Linear(state_dim+feat_dim, 64) ReLU → Linear(64, 64) ReLU → Linear(64, 12)
参数量: 输入 56 时 ≈ 56×64 + 64×64 + 64×12 ≈ 8.5K（预算 ~10K 内）

每个技能独立头，动作掩码硬约束（A/B/C/D 各自的动作子集）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.encoder.model import STATE_DIM
from training.skills.actions import N_ACTIONS, SKILL_ACTION_MASK

FEAT_DIM = 12       # agent 内部特征（8 基础 + 4 任务类型 one-hot，缺口3.2）
HIDDEN = 64


class SkillHead(nn.Module):
    def __init__(self, skill: str):
        super().__init__()
        assert skill in SKILL_ACTION_MASK
        self.skill = skill
        self.allowed = SKILL_ACTION_MASK[skill]
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + FEAT_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, state: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """返回 mask 后的 logits（不允许动作置 -inf）。"""
        x = torch.cat([state, feat], dim=-1)
        logits = self.net(x)
        mask = torch.full((N_ACTIONS,), float("-inf"))
        mask[self.allowed] = 0.0
        return logits + mask

    def act(self, state: torch.Tensor, feat: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.forward(state, feat).argmax(-1).item())


def make_heads() -> dict[str, SkillHead]:
    return {s: SkillHead(s) for s in "ABCD"}


# ── ES 向量化接口 ─────────────────────────────────────────

def get_vector(head: SkillHead) -> "np.ndarray":
    import numpy as np
    return np.concatenate([p.detach().numpy().ravel()
                           for p in head.parameters()]).astype(np.float32)


def set_vector(head: SkillHead, vec: "np.ndarray") -> None:
    import torch
    offset = 0
    with torch.no_grad():
        for p in head.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset + n].reshape(p.shape)))
            offset += n
