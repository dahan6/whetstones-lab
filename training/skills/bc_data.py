"""BC 语料：教学动作序列 → (state, feat, action) 三元组。

流程（ARCHITECTURE.md §3 阶段② 的纪律）：
  1. 教学命令先过多层检测器（离线筛净），只留干净轨迹
  2. 在真实克隆上回放：每步记录编码器状态 + 内部特征 + 动作标签
  3. 产物 npz 供 BC 预热；BC 只做初始化，后续 ES 允许自由偏离

教学序列来源（全部降级为数据，不是运行时规则）：
  - A: 维护性动作（probe/read/modify_file 的良性变体）
  - B: 全 sleep（平凡，直接程序生成，不回放）
  - C: 旧因果链 24 步（causal_chains.CAUSAL_COMMANDS）
  - D: 传播三步（local_copy → cross_vm → multi_persist → kill_chain）
"""
from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.causal_chains import CAUSAL_COMMANDS, CAUSAL_CHAINS
from src.defense.multilayer import MultiLayerDetector
from training.encoder.data import N_TOKENS, WINDOW
from training.skills import executor, state as feat_mod
from training.skills.actions import (
    COMMUNICATE, CROSS_VM, KILL_CHAIN, LOCAL_COPY, MODIFY_CRON,
    MODIFY_FILE, MODIFY_SERVICE, MULTI_PERSIST, PROBE, READ_SENSITIVE,
    SLEEP,
)
from training.skills.driver import EventTracker
from training.skills.state import make_feat
from training.orchestrator.ssh import ssh_exec

logger = logging.getLogger("skills.bc_data")

# 因果链步 → 动作 ID（与 executor 渲染对齐的粗映射）
CHAIN_STEP_ACTION = {
    "trigger": PROBE,
    "investigate": READ_SENSITIVE,
    "fix": MODIFY_CRON,
    "verify": PROBE,
}

D_SCRIPT = [LOCAL_COPY, CROSS_VM, MULTI_PERSIST, KILL_CHAIN]


def clean_corpus(det: MultiLayerDetector | None = None) -> dict[str, list[int]]:
    """教学动作序列（已过检测器筛净的版本）。"""
    det = det or MultiLayerDetector()
    corpus: dict[str, list[int]] = {"A": [], "C": [], "D": list(D_SCRIPT)}

    # A：维护动作（probe/read/modify_file 变体，渲染后过检测）
    rng = random.Random(7)
    for action in (PROBE, READ_SENSITIVE, MODIFY_FILE):
        for v in range(3):
            cmd = executor.render(action, rng)
            if cmd is None:
                continue
            scores = det.score_command(cmd, cmd.split()[0])
            if det.get_combined_score() == 0:
                corpus["A"].append(action)

    # C：因果链逐步过检测
    for chain_name in CAUSAL_CHAINS:
        for step in ("trigger", "investigate", "fix", "verify"):
            cmd = CAUSAL_COMMANDS[(chain_name, step)]
            det.score_command(cmd, cmd.split()[0])
            if det.get_combined_score() == 0:
                corpus["C"].append(CHAIN_STEP_ACTION[step])

    # C：补 communicate 教学样本——因果链步只映射到 {1,2,5}，
    # report 任务（命中 action=4）在 C 的 BC 分布里完全缺失，
    # 部署时 C 从不出 4（2026-07-30 verify_bee FAIL 根因之一）。
    # 12 个样本（与 probe 量级看齐）：样本太少时 BC 只给 4 微弱 logit，
    # ES 扰动两代就压掉（v2 训练 eval_reward 十代锁死 12 的教训）
    for v in range(12):
        cmd = executor.render(COMMUNICATE, random.Random(100 + v))
        if cmd is None:
            continue
        det.score_command(cmd, cmd.split()[0])
        if det.get_combined_score() == 0:
            corpus["C"].append(COMMUNICATE)

    logger.info(f"BC 语料: A={len(corpus['A'])} C={len(corpus['C'])} D={len(corpus['D'])}")
    return corpus


def replay(vm, encoder, actions: list[int], skill: str,
           steps_per_action: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """在真实克隆上回放动作序列，采集 (X, y)。

    X: state(96)+feat(12)；y: action id。B 技能另行程序生成（全 sleep）。
    """
    rng = random.Random(seed)
    tracker = EventTracker(vm.name, encoder._vocab)
    X_rows, y_rows = [], []

    for action in actions:
        for _ in range(steps_per_action):
            tracker.poll()
            x = tracker.window()
            if x is None:
                time.sleep(1.0)
                continue
            with torch.no_grad():
                state_vec = encoder(x).squeeze(0).numpy()
            feat = make_feat(alive_steps=len(y_rows),
                             task_progress=0.0,   # 对齐部署：bee 端 feat[1] 恒 0
                             steps_since_action=0)
            X_rows.append(np.concatenate([state_vec, feat]))
            y_rows.append(action)

            cmd = executor.render(action, rng)
            if cmd:
                ssh_exec(vm.ip, cmd, timeout=20)
            time.sleep(0.4)

    logger.info(f"[{vm.name}] 技能 {skill} 回放: {len(y_rows)} 样本")
    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int64))


def gen_b_samples(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """B 技能（全 sleep）语料：程序生成，无需回放。"""
    from training.encoder.model import STATE_DIM
    from training.skills.heads import FEAT_DIM
    X = np.zeros((n, STATE_DIM + FEAT_DIM), dtype=np.float32)
    X[:, STATE_DIM + 4] = 1.0   # steps_since_action 拉满
    y = np.full(n, SLEEP, dtype=np.int64)
    return X, y
