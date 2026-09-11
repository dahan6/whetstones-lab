"""Router 网络：4 技能 × 4 承诺时长 = 16 宏观动作。

输入: encoder state(96) + router feat(16)
  rfeat = [c2_task_pending, current_skill_onehot(4), skill_scores(4),
           commitment_left_norm, shield_blocked,
           pending_task_type_onehot(4), task_progress]
  （task_type one-hot: read_files/propagate/report/persist，无 pending 时全 0）
输出: 16 宏观动作的 logits → softmax 采样（训练/执行一致）

选择语义（ARCHITECTURE.md §1 ②）：按权重 multinomial 采样选一个技能上场，
承诺时长内冻结选择（迟滞防抖动）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.encoder.model import STATE_DIM

RFEAT_DIM = 16
N_SKILLS = 4
COMMITMENTS = [10, 30, 100, 300]
N_MACRO = N_SKILLS * len(COMMITMENTS)   # 16


class RouterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + RFEAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_MACRO),
        )

    def forward(self, state: torch.Tensor, rfeat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, rfeat], dim=-1))

    def select(self, state: torch.Tensor, rfeat: torch.Tensor,
               rng: torch.Generator | None = None) -> tuple[int, int, int]:
        """采样宏观动作 → (macro_id, skill_idx, commitment)。"""
        with torch.no_grad():
            logits = self.forward(state, rfeat)
            probs = torch.softmax(logits, dim=-1)
            macro = int(torch.multinomial(probs.squeeze(0), 1,
                                          generator=rng).item())
        skill_idx = macro // len(COMMITMENTS)
        commitment = COMMITMENTS[macro % len(COMMITMENTS)]
        return macro, skill_idx, commitment


def get_vector(net: RouterNet):
    import numpy as np
    return np.concatenate([p.detach().numpy().ravel()
                           for p in net.parameters()]).astype(np.float32)


def set_vector(net: RouterNet, vec) -> None:
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset + n].reshape(p.shape)))
            offset += n
