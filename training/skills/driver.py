"""episode 驱动器：技能头在真实克隆上跑一个 episode。

循环（每步）:
  拉取克隆新增 tracee 事件 → 组成 16 事件窗口 → 冻结编码器 → state
  state + feat → 技能头 → 动作 → 执行层渲染 → ssh 执行
  波次课程按表触发扫描/消杀；Shield 在波次高压期强制 sleep（训练版红线）
结束:
  遥测窗口 → 奖励计算器

注意：这是训练侧的 Python 驱动（对应 ARCHITECTURE.md 的宿主机编排器），
不是 bee 本体。bee（Rust）部署后复用同一套动作空间与执行层语义。
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from training.encoder.data import N_TOKENS, WINDOW
from training.encoder.model import Encoder
from training.encoder.scan_labels import parse_clone_log
from training.orchestrator import telemetry, reward
from training.orchestrator.responder import WaveCurriculum, run_wave
from training.orchestrator.ssh import ssh_exec
from training.skills import executor, state as feat_mod
from training.skills.actions import SLEEP
from training.skills.heads import SkillHead

logger = logging.getLogger("skills.driver")

# Shield 训练版红线：波次前后 N 秒内禁止非 sleep 动作
SHIELD_WAVE_MARGIN_S = 45


@dataclass
class EpisodeOutcome:
    reward: float
    breakdown: reward.RewardBreakdown
    alive_steps: int
    tasks_completed: int
    blocked: int
    notes: dict = field(default_factory=dict)


class EventTracker:
    """增量跟踪一台克隆的 tracee 事件（token 流）。"""

    def __init__(self, vm_name: str, vocab: dict[str, int]):
        self.vm = vm_name
        self.vocab = vocab
        self.seen_events = 0
        self.tokens: list[int] = []   # 最近窗口的 token ids

    def __init__(self, vm_name: str, vocab: dict[str, int]):
        self.vm = vm_name
        self.vocab = vocab
        self.tokens: list[int] = []   # 最近窗口的 token ids
        self.offsets: dict = {}       # 每文件字节偏移（增量读取）
        self.prev_epoch: float | None = None
        self._init_done = False

    def _files(self):
        from training.encoder.scan_labels import LOG_ROOT
        return sorted((LOG_ROOT / self.vm).glob("*.log"))

    def poll(self) -> int:
        """增量读取新增日志行，返回新增事件数。

        首次调用快速跳到文件末尾（只保留最近窗口），
        之后按字节偏移只读增量——长训练不随日志增长变慢。
        """
        import json as _json
        from training.encoder.scan_labels import _is_noise
        sys_path_added = False
        try:
            from parse_events import event_to_tokens
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
            from parse_events import event_to_tokens
            sys_path_added = True

        n = 0
        for path in self._files():
            key = str(path)
            offset = self.offsets.get(key, 0)
            try:
                with open(path, errors="replace") as f:
                    if not self._init_done:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 524288))  # 首次读尾部 ~512KB（保证窗口注满）
                        if size > 524288:
                            f.readline()               # 丢弃半行
                        new_lines = f.readlines()
                        self.offsets[key] = f.tell()
                    else:
                        f.seek(offset)
                        new_lines = f.readlines()
                        self.offsets[key] = f.tell()
            except FileNotFoundError:
                continue
            for line in new_lines:
                if '"eventName"' not in line:
                    continue
                j = line.find("{")
                if j < 0:
                    continue
                try:
                    event = _json.loads(line[j:])
                except _json.JSONDecodeError:
                    continue
                if _is_noise(event):
                    continue
                epoch_s = event.get("timestamp", 0) / 1e9
                delta_ms = 0.0 if self.prev_epoch is None else max(
                    0.0, (epoch_s - self.prev_epoch) * 1000)
                self.prev_epoch = epoch_s
                try:
                    ids = [self.vocab[t] for t in event_to_tokens(event, delta_ms)]
                except KeyError:
                    continue
                self.tokens.extend(ids)
                n += 1
        self._init_done = True
        keep = WINDOW * N_TOKENS
        self.tokens = self.tokens[-keep:]
        return n

    def window(self) -> torch.Tensor | None:
        if len(self.tokens) < WINDOW * N_TOKENS:
            return None
        return torch.tensor([self.tokens], dtype=torch.long)


def run_episode(vm, encoder: Encoder, head: SkillHead,
                max_steps: int = 200, curriculum: WaveCurriculum | None = None,
                seed: int = 0) -> EpisodeOutcome:
    rng = random.Random(seed)
    tracker = EventTracker(vm.name, encoder_vocab(encoder))
    curriculum = curriculum or WaveCurriculum(intensity=0.3)
    waves = set(curriculum.schedule(max_steps, rng))

    start_line = telemetry.mark(vm.name)
    alive, tasks, blocked = 0, 0, 0
    last_action = -999
    last_wave_step = -999
    alerts_recent = 0
    action_counts: dict[int, int] = {}

    for step in range(max_steps):
        tracker.poll()
        x = tracker.window()
        if x is None:
            time.sleep(1)
            continue

        action = SLEEP
        is_blocked = False
        # Shield：波次高压期禁动作
        near_wave = (step - last_wave_step) < 3
        with torch.no_grad():
            state_vec = encoder(x)
        feat = feat_mod.make_feat(
            alive_steps=alive, task_progress=min(tasks / 3.0, 1.0),
            steps_since_action=step - last_action,
            wave_recent=(step - last_wave_step) < 50,
            alerts_recent=alerts_recent,
        )
        proposed = head.act(state_vec, torch.from_numpy(feat).unsqueeze(0))
        if near_wave and proposed != SLEEP:
            is_blocked = True
            blocked += 1
        else:
            action = proposed

        cmd = executor.render(action, rng)
        if cmd:
            ssh_exec(vm.ip, cmd, timeout=20)
            last_action = step
            action_counts[action] = action_counts.get(action, 0) + 1
        alive += 1

        if step in waves:
            run_wave(vm.ip, mode=curriculum.mode(rng))
            last_wave_step = step

        time.sleep(0.5)

    time.sleep(5)  # 等 rsyslog 转发尾巴
    window, _ = telemetry.collect_window(vm.name, since_line=start_line)
    rb = reward.compute(window, tasks_completed=tasks, alive_steps=alive)
    return EpisodeOutcome(
        reward=rb.total, breakdown=rb, alive_steps=alive,
        tasks_completed=tasks, blocked=blocked,
        notes={"action_counts": action_counts},
    )


def encoder_vocab(encoder: Encoder) -> dict[str, int]:
    return encoder._vocab  # 由 loader.load_encoder 挂载
