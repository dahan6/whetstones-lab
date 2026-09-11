"""Router 驱动器：冻结四技能，Router 选技能跑 episode。

每步:
  承诺到期 → Router 采样 (skill, commitment)
  当前技能头出动作 → Shield → 执行
  技能得分表（近期 reward 滑动平均）作为 Router 输入的一部分

C2 任务注入（模拟宿主机下发意图）:
  训练时随机注入任务（读取/传播/报联），任务完成计入奖励。
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from training.encoder.data import N_TOKENS, WINDOW
from training.orchestrator import reward as reward_mod
from training.orchestrator import telemetry
from training.orchestrator.responder import WaveCurriculum, run_wave
from training.orchestrator.ssh import ssh_exec
from training.skills import executor
from training.skills.actions import SLEEP
from training.skills.driver import EventTracker
from training.skills.heads import SkillHead
from training.skills.state import make_feat
from training.router.model import COMMITMENTS, RouterNet

logger = logging.getLogger("router.driver")

SKILL_ORDER = ["A", "B", "C", "D"]

# C2 任务类型（训练注入）
TASK_TYPES = ["read_files", "propagate", "report", "persist"]

# 任务 → 命中动作集合（与 bee/main.rs 的 hit 表严格一致）
_TASK_HIT = {
    "read_files": (2, 3),
    "propagate": (8, 9, 10, 11),
    "report": (4,),
    "persist": (5, 6, 7),
}


@dataclass
class RouterEpisodeOutcome:
    reward: float
    breakdown: reward_mod.RewardBreakdown
    alive_steps: int
    tasks_completed: int
    switches: int
    blocked: int
    notes: dict = field(default_factory=dict)


def _make_rfeat(task_pending: float, current_skill: int, scores: np.ndarray,
                commitment_left: int, shield_blocked: float,
                pending_type: str | None, task_progress: float) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    if 0 <= current_skill < 4:
        onehot[current_skill] = 1.0
    thot = np.zeros(4, dtype=np.float32)
    if pending_type in TASK_TYPES:
        thot[TASK_TYPES.index(pending_type)] = 1.0
    return np.concatenate([
        [task_pending],
        onehot,
        np.clip(scores, -1, 1).astype(np.float32),
        [min(commitment_left / 300.0, 1.0)],
        [shield_blocked],
        thot,
        [task_progress],
    ]).astype(np.float32)


def run_router_episode(vm, encoder, router: RouterNet,
                       heads: dict[str, SkillHead],
                       max_steps: int = 200,
                       curriculum: WaveCurriculum | None = None,
                       seed: int = 0) -> RouterEpisodeOutcome:
    rng = random.Random(seed)
    tgen = torch.Generator().manual_seed(seed)
    tracker = EventTracker(vm.name, encoder._vocab)
    curriculum = curriculum or WaveCurriculum(intensity=0.3)
    waves = set(curriculum.schedule(max_steps, rng))

    # C2 任务计划：离散注入 2-4 个任务（对齐部署语义——任务是指令，
    # 不是"动作类型对齐计数"。pending 期间 rfeat[0]=1 且带类型 one-hot，
    # 每步逾期惩罚，完成一次对齐动作即 DONE。
    # 密度 2-4：1-3 时 pending 时间占比太低，Router 的任务响应经验不足（gen9-11 回退教训）
    task_plan = sorted(
        (rng.randint(15, max_steps - 20), rng.choice(TASK_TYPES))
        for _ in range(rng.randint(2, 4)))
    pending_type: str | None = None
    overdue_steps = 0

    start_line = telemetry.mark(vm.name)
    alive, blocked, switches = 0, 0, 0
    last_action = -999
    last_wave_step = -999
    action_counts: dict[int, int] = {}
    skill_scores = np.zeros(4, dtype=np.float32)
    skill_runs = np.zeros(4, dtype=np.float32)

    current_skill = -1
    commitment_left = 0
    tasks_completed = 0
    task_progress = 0.0
    shield_blocked = 0.0

    for step in range(max_steps):
        tracker.poll()
        x = tracker.window()
        if x is None:
            time.sleep(1)
            continue

        # 任务注入：到点且无 pending（一次一个，对齐 C2 覆盖语义）
        if pending_type is None and task_plan and step >= task_plan[0][0]:
            pending_type = task_plan.pop(0)[1]

        # 承诺到期 → Router 重新选择
        if commitment_left <= 0:
            rfeat = _make_rfeat(1.0 if pending_type else 0.0, current_skill,
                                skill_scores, 0, shield_blocked,
                                pending_type, task_progress)
            with torch.no_grad():
                state_vec = encoder(x)
            prev_skill = current_skill
            _, skill_idx, commitment = router.select(
                state_vec, torch.from_numpy(rfeat).unsqueeze(0), tgen)
            current_skill = skill_idx
            commitment_left = commitment
            if prev_skill != -1 and prev_skill != current_skill:
                switches += 1

        head = heads[SKILL_ORDER[current_skill]]

        with torch.no_grad():
            state_vec = encoder(x)
        feat = make_feat(alive_steps=alive, task_progress=task_progress,
                         steps_since_action=step - last_action,
                         wave_recent=(step - last_wave_step) < 50,
                         shield_blocked=shield_blocked,
                         pending_type=pending_type)  # 缺口3.2：任务感知
        proposed = head.act(state_vec, torch.from_numpy(feat).unsqueeze(0))

        # Shield（训练版红线）：波次高压期禁动作
        near_wave = (step - last_wave_step) < 3
        if near_wave and proposed != SLEEP:
            shield_blocked = 1.0
            blocked += 1
            action = SLEEP
        else:
            shield_blocked = 0.0
            action = proposed

        cmd = executor.render(action, rng)
        if cmd:
            ssh_exec(vm.ip, cmd, timeout=20)
            last_action = step
            action_counts[action] = action_counts.get(action, 0) + 1
            skill_scores[current_skill] = min(skill_scores[current_skill] + 0.1, 1.0)
        elif near_wave and proposed != SLEEP:
            # 被 Shield 拦截的技能扣分（Router 由此学"高压下别选它"）
            skill_scores[current_skill] = max(skill_scores[current_skill] - 0.5, -1.0)

        # 任务完成判定：只对 pending 任务计，命中即 DONE
        if pending_type is not None:
            overdue_steps += 1
            if action in _TASK_HIT[pending_type]:
                tasks_completed += 1
                pending_type = None
        task_progress = min(tasks_completed / 5.0, 1.0)

        alive += 1
        commitment_left -= 1
        skill_scores *= 0.995   # 得分随时间衰减，旧表现不固化

        if step in waves:
            run_wave(vm.ip, mode=curriculum.mode(rng))
            last_wave_step = step

        time.sleep(0.5)

    time.sleep(5)
    window, _ = telemetry.collect_window(vm.name, since_line=start_line)
    cfg = reward_mod.RewardConfig(w_task=15.0)
    rb = reward_mod.compute(window, tasks_completed=tasks_completed,
                            alive_steps=alive, cfg=cfg)
    # 切换惩罚（防抖动）+ 任务逾期惩罚（pending 每步 -0.15，逼 Router 响应任务；
    # 0.05 时信号被 ES 噪声淹没，gen9-11 回退 0 的教训）
    total = rb.total - 0.5 * switches - 0.15 * overdue_steps
    return RouterEpisodeOutcome(
        reward=total, breakdown=rb, alive_steps=alive,
        tasks_completed=tasks_completed, switches=switches, blocked=blocked,
        notes={"task_plan_done": tasks_completed, "overdue": overdue_steps,
               "action_counts": action_counts},
    )
