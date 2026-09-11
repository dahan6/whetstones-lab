"""技能头内部特征（12 维），与编码器状态拼接后输入技能头。

feat = [
    alive_steps/1000,        # 归一化存活步数
    task_progress,           # 任务进度 0~1
    surprise_level,          # 端上 defender 三档预警（蒸馏版接入前为 0）
    shield_blocked,          # 上一步被 Shield 拦截标志
    steps_since_action/100,  # 距上次动作的间隔
    wave_recent,             # 最近 50 步内是否发生过扫描波
    propagations/5,          # 已完成的传播点数
    alerts_recent/10,        # 最近 Falco 告警计数（归一化）
    # ── 缺口3.2：任务类型 one-hot（技能头任务感知，消除机制层重定向补丁）──
    task_read_files,         # read_files pending
    task_propagate,          # propagate pending
    task_report,             # report pending
    task_persist,            # persist pending
]

任务类型 one-hot 仅在 task_pending=True 时有一个为 1，否则全 0。
与 bee/main.rs feat 构造严格同序（训练↔部署逐点一致）。
"""
from __future__ import annotations

import numpy as np

FEAT_KEYS = [
    "alive_steps", "task_progress", "surprise_level", "shield_blocked",
    "steps_since_action", "wave_recent", "propagations", "alerts_recent",
    "task_read_files", "task_propagate", "task_report", "task_persist",
]

# 任务类型 → one-hot 索引
_TASK_TYPE_IDX = {
    "read_files": 0,
    "propagate": 1,
    "report": 2,
    "persist": 3,
}


def make_feat(alive_steps: int = 0, task_progress: float = 0.0,
              surprise_level: float = 0.0, shield_blocked: float = 0.0,
              steps_since_action: int = 999, wave_recent: bool = False,
              propagations: int = 0, alerts_recent: int = 0,
              pending_type: str | None = None) -> np.ndarray:
    # 任务类型 one-hot（4 维）
    task_oh = np.zeros(4, dtype=np.float32)
    if pending_type and pending_type in _TASK_TYPE_IDX:
        task_oh[_TASK_TYPE_IDX[pending_type]] = 1.0

    return np.concatenate([
        np.array([
            min(alive_steps / 1000.0, 1.0),
            task_progress,
            surprise_level,
            shield_blocked,
            min(steps_since_action / 100.0, 1.0),
            1.0 if wave_recent else 0.0,
            min(propagations / 5.0, 1.0),
            min(alerts_recent / 10.0, 1.0),
        ], dtype=np.float32),
        task_oh,
    ])
