"""Router BC 预热：把"任务响应"的标准答案直接教给 Router，ES 只做精炼。

动机（v2-v4 四轮 ES 失败的根因，2026-07-30）：
  "pending 任务类型 → 对齐技能+短窗"是稀疏条件映射——pending 只占 episode
  1/3 时间，rfeat[11:15] 大部分为 0，ES 黑盒扰动在全参数空间找 4 个 one-hot
  维度的条件权重，信噪比极低。四轮纯 ES 最好成绩门3 均值 0.33。
  架构纪律允许 BC 初始化（"BC 只做初始化，ES 可自由偏离"）——
  把任务响应当教学数据，ES 随后学"什么时候打破规则"（高压时静默优先）。

标签规则（宏观动作 id = skill×4 + 窗档 idx，窗档 {10,30,100,300}）：
  shield_blocked=1      → B×10   （高压先静默短窗观望）
  pending=propagate     → D×30
  pending=report        → C×30
  pending=read_files    → A×30   （A 的 READ_SENSITIVE 命中 {2}，最低噪）
  pending=persist       → C×30
  无 pending            → B×300  （静默长窗=生存基线）

state 用真实 VM 采集的编码器状态（不是合成噪声——保证 BC 分布内）。
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.orchestrator.vm_pool import VMPool
from training.router.model import COMMITMENTS, N_MACRO, RouterNet
from training.skills.driver import EventTracker
from training.skills.loader import load_encoder

logger = logging.getLogger("router.bc")

SKILL_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
TASK_TYPES = ["read_files", "propagate", "report", "persist"]


def rule_label(pending_type: str | None, shield_blocked: float) -> int:
    """(pending, shield) → 宏观动作 id。"""
    if shield_blocked >= 0.5:
        return SKILL_IDX["B"] * 4 + 0          # B×10
    if pending_type is None:
        return SKILL_IDX["B"] * 4 + 1          # B×30（B 只出 SLEEP，长短窗行为相同；
                                               # 短窗=任务最坏等 24s，纯收益）
    skill = {"propagate": "D", "report": "C", "read_files": "C",
             "persist": "C"}[pending_type]
    return SKILL_IDX[skill] * 4 + 1            # 技能×30


def collect_states(vm, encoder, n: int = 400) -> np.ndarray:
    """真机采集编码器状态（覆盖不同时段/负载的自然分布）。"""
    tracker = EventTracker(vm.name, encoder._vocab)
    states = []
    while len(states) < n:
        tracker.poll()
        x = tracker.window()
        if x is None:
            time.sleep(0.5)
            continue
        with torch.no_grad():
            states.append(encoder(x).squeeze(0).numpy())
        time.sleep(0.15)
    return np.stack(states)


def gen_rfeat_batch(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """合成 rfeat(16) 与标签。类别：pending 4 类各 15%，无 pending 40%。"""
    X = np.zeros((n, 16), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        u = rng.random()
        ptype = None if u < 0.40 else TASK_TYPES[int((u - 0.40) / 0.15) % 4]
        shield = 1.0 if rng.random() < 0.10 else 0.0
        X[i, 0] = 1.0 if ptype else 0.0
        cur = int(rng.integers(0, 4))
        X[i, 1 + cur] = 1.0
        X[i, 5:9] = rng.uniform(-1, 1, 4)
        X[i, 9] = 0.0                          # 选择点：承诺剩余=0
        X[i, 10] = shield
        if ptype:
            X[i, 11 + TASK_TYPES.index(ptype)] = 1.0
        X[i, 15] = rng.uniform(0, 1)
        y[i] = rule_label(ptype, shield)
    return X, y


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/router_bc.pt")
    ap.add_argument("--n-states", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    encoder = load_encoder()
    pool = VMPool(size=1)
    vm = pool.existing()[0]
    vm.refresh_ip()
    logger.info(f"从 {vm.name} 采集真实状态 ×{args.n_states}...")
    states = collect_states(vm, encoder, args.n_states)

    rng = np.random.default_rng(7)
    rX, ry = gen_rfeat_batch(len(states), rng)
    Xs = torch.from_numpy(states).float()
    Xr = torch.from_numpy(rX).float()
    y = torch.from_numpy(ry)

    router = RouterNet()
    opt = torch.optim.Adam(router.parameters(), lr=2e-3)
    n = len(y)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 64):
            idx = perm[b:b + 64]
            logits = router(Xs[idx], Xr[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 40 == 0:
            with torch.no_grad():
                acc = (router(Xs, Xr).argmax(-1) == y).float().mean().item()
            logger.info(f"epoch {ep+1}/{args.epochs} loss={tot:.3f} acc={acc:.3f}")

    torch.save({"router_state": router.state_dict(), "phase": "bc"}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
