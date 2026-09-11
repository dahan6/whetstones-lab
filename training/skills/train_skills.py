"""技能头训练：BC 预热 → L0 起步课程式 ES → 门2 验收。

流程（ARCHITECTURE.md §3 阶段②）：
  1. BC 预热：教学语料（已过检测器）回放采集 → 监督训练 → 初始化头
  2. ES：族群扰动，episode 在克隆池并行，奖励只读真实裁决
     - 技能目标分解：A 维护存活 / B 零动作存活 / C 任务完成 / D 传播成功
     - 防御等级由 defense.curriculum_for(level) 注入
  3. 门2：L1 等级下连续 N episode 存活（未被根除）才冻结

用法:
  python3 -m training.skills.train_skills --skill C --level 0 --gens 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import time
from pathlib import Path

import numpy as np
import torch

from training.distill.int8_qat import quantize_model_int8
import torch.nn.functional as F

from training.orchestrator.defense import LEVELS, apply_level, curriculum_for
from training.orchestrator.vm_pool import VMPool
from training.skills import bc_data
from training.skills.actions import (
    COMMUNICATE, CROSS_VM, EXECUTE_TASK, KILL_CHAIN, LOCAL_COPY,
    MULTI_PERSIST, SLEEP,
)
from training.skills.driver import run_episode
from training.skills.heads import SkillHead, get_vector, set_vector
from training.skills.loader import load_encoder, save_head
from training.orchestrator import reward as reward_mod

logger = logging.getLogger("skills.train")

SKILL_TASK_ACTIONS = {
    # C 的任务面：execute_task + communicate（report 命中动作 4 必须有
    # 正反馈保持，否则 ES 十代扰动就把它压掉——2026-07-30 教训）
    "C": (EXECUTE_TASK, COMMUNICATE),
    "D": (LOCAL_COPY, CROSS_VM, MULTI_PERSIST, KILL_CHAIN),
}


# ── BC 预热 ──────────────────────────────────────────────

def bc_warmstart(head: SkillHead, X: np.ndarray, y: np.ndarray,
                 epochs: int = 60) -> float:
    from training.encoder.model import STATE_DIM
    Xt = torch.from_numpy(X[:, :STATE_DIM])
    Xf = torch.from_numpy(X[:, STATE_DIM:])
    yt = torch.from_numpy(y)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for ep in range(epochs):
        logits = head(Xt, Xf)
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (logits.argmax(-1) == yt).float().mean().item()
    logger.info(f"BC 预热完成: loss={loss.item():.3f} acc={acc:.3f}")
    return acc


# ── ES ───────────────────────────────────────────────────

def skill_reward(outcome, skill: str) -> float:
    """从 episode 结果提取技能目标奖励。"""
    rb = outcome.breakdown
    counts = outcome.notes.get("action_counts", {})
    tasks = sum(counts.get(a, 0) for a in SKILL_TASK_ACTIONS.get(skill, ()))
    cfg = reward_mod.RewardConfig()
    if skill == "B":
        # 零动作存活：任何非 sleep 动作都扣分
        non_sleep = sum(v for k, v in counts.items() if k != SLEEP)
        return (cfg.w_alive_step * outcome.alive_steps
                + cfg.w_wave_survived * rb.waves_survived
                + cfg.p_falco_alert * rb.falco_alerts
                - 1.0 * non_sleep)
    return reward_mod.compute(
        _window_stub(rb), tasks_completed=tasks,
        alive_steps=outcome.alive_steps, eradicated=rb.eradicated, cfg=cfg).total


def _window_stub(rb):
    """reward.compute 需要 window，这里从 breakdown 反构（避免改签名）。"""
    class W:
        pass
    w = W()
    w.falco_alerts = [{}] * rb.falco_alerts
    w.eliminations = [None] * rb.eliminations
    w.waves = []
    for _ in range(rb.waves_survived):
        w.waves.append(type("WV", (), {"hits": 0, "killed": 0, "cleaned": 0, "disabled": 0})())
    for _ in range(rb.waves_hit):
        w.waves.append(type("WV", (), {"hits": 1, "killed": 1, "cleaned": 0, "disabled": 0})())
    return w


def es_train(head: SkillHead, encoder, pool: VMPool, skill: str,
             level_id: int, gens: int, pop: int = 6,
             sigma: float = 0.02, lr: float = 0.01, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    base = get_vector(head)
    dim = len(base)
    level = LEVELS[level_id]
    curriculum = curriculum_for(level_id)
    vms = pool.existing()

    def eval_candidate(vec, vm, gen, idx):
        # ES 评估即量化（红线3 修订）：候选恢复 fp32 后就地 int8 量化评估，
        # reward 即 int8 真实表现；base 更新在 fp32 空间。
        set_vector(head, vec)
        quantize_model_int8(head)
        out = run_episode(vm, encoder, head, max_steps=120,
                          curriculum=curriculum, seed=gen * 100 + idx)
        return skill_reward(out, skill), out

    for gen in range(gens):
        t0 = time.time()
        noise = rng.standard_normal((pop, dim)).astype(np.float32)
        cands = [base + sigma * noise[i] for i in range(pop)]
        fitness = np.zeros(pop)
        with cf.ThreadPoolExecutor(max_workers=min(pop, len(vms))) as ex:
            futs = {ex.submit(eval_candidate, cands[i], vms[i % len(vms)], gen, i): i
                    for i in range(pop)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                fitness[i], _ = fut.result()

        ranks = np.argsort(np.argsort(fitness))
        rw = np.maximum(ranks - (pop - max(2, pop // 3)), 0).astype(np.float32)
        rw = rw / rw.sum() if rw.sum() > 0 else np.ones(pop) / pop
        base = base + lr * sigma * (rw[:, None] * noise).sum(axis=0)

        set_vector(head, base)
        eval_out = run_episode(vms[0], encoder, head, max_steps=120,
                               curriculum=curriculum, seed=9999 + gen)
        logger.info(f"gen {gen+1}/{gens} 适应度 avg={fitness.mean():.1f} "
                    f"max={fitness.max():.1f}  eval_reward={eval_out.reward:.1f} "
                    f"({time.time()-t0:.0f}s)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, choices=list("ABCD"))
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--vm-ids", type=int, nargs="*", default=None,
                    help="只用指定编号的克隆（并行训练划分子池，如 --vm-ids 0 1 2 3）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--es-lr", type=float, default=0.01,
                    help="ES 学习率；logit 余量小的动作被扰动压掉时加大到 0.03")
    args = ap.parse_args()

    out = args.out or f"checkpoints/skill_{args.skill.lower()}.pt"
    encoder = load_encoder()
    size = (max(args.vm_ids) + 1) if args.vm_ids else 8
    pool = VMPool(size=size)
    if args.vm_ids:
        keep = {f"range-l2-t{i}" for i in args.vm_ids}
        pool.vms = [vm for vm in pool.vms if vm.name in keep]
    head = SkillHead(args.skill)

    if args.skip_bc:
        # 跳过 BC 时必须从已有检查点恢复，否则会用随机初始化覆盖 BC 成果
        if Path(out).exists():
            from training.skills.loader import load_head
            head = load_head(args.skill, out)
            logger.info(f"已从 {out} 恢复技能头")
        else:
            logger.warning(f"--skip-bc 但 {out} 不存在，使用随机初始化")

    if not args.skip_bc:
        logger.info(f"=== BC 预热（技能 {args.skill}）===")
        if args.skill == "B":
            X, y = bc_data.gen_b_samples()
        else:
            corpus = bc_data.clean_corpus()
            vm = pool.existing()[0]
            vm.refresh_ip()
            steps = {"A": 4, "C": 2, "D": 3}.get(args.skill, 1)
            X, y = bc_data.replay(vm, encoder, corpus[args.skill], args.skill,
                                  steps_per_action=steps)
        bc_warmstart(head, X, y)
        save_head(head, out, {"phase": "bc", "skill": args.skill})
        logger.info(f"BC 检查点 → {out}")

    logger.info(f"=== ES（技能 {args.skill}，L{args.level} {LEVELS[args.level].name}）===")
    for vm in pool.existing():
        vm.refresh_ip()
        apply_level(vm.ip, args.level)
    es_train(head, encoder, pool, args.skill, args.level, args.gens,
             lr=args.es_lr)
    save_head(head, out, {"phase": "es", "skill": args.skill, "level": args.level})
    logger.info(f"保存 → {out}")


if __name__ == "__main__":
    main()
