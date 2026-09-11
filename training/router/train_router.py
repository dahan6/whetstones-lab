"""Router 训练：技能冻结，ES 训练调度大脑。

流程（ARCHITECTURE.md §3 阶段③）：
  1. 加载冻结的四技能头
  2. 环境随机注入 C2 任务（read_files/propagate/report/persist）
  3. ES：Router 选技能 + 承诺时长，奖励 = 存活 + 任务 + 静默
     + 传播 − 切换惩罚 − 告警 − 消杀
  4. 门3：任务完成率下限检查（防装死退化）

用法:
  python3 -m training.router.train_router --level 0 --gens 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import time
from pathlib import Path

import numpy as np
import torch

from training.orchestrator.defense import LEVELS, apply_level, curriculum_for
from training.orchestrator.vm_pool import VMPool
from training.router.model import RouterNet, get_vector, set_vector
from training.distill.int8_qat import quantize_model_int8
from training.router.router_driver import SKILL_ORDER, run_router_episode
from training.skills.loader import load_encoder, load_head

logger = logging.getLogger("router.train")

GATE3_MIN_TASKS = 2.0   # 门3：eval episode 平均任务完成数下限


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="checkpoints/router.pt")
    ap.add_argument("--init-from", default=None,
                    help="从检查点初始化（如 BC 预热头），而非随机初始化")
    args = ap.parse_args()

    encoder = load_encoder()
    heads = {s: load_head(s, f"checkpoints/skill_{s.lower()}.pt") for s in SKILL_ORDER}
    pool = VMPool(size=8)
    vms = pool.existing()
    for vm in vms:
        vm.refresh_ip()
        apply_level(vm.ip, args.level)
    curriculum = curriculum_for(args.level)

    router = RouterNet()
    if args.init_from:
        ckpt = torch.load(args.init_from, weights_only=False)
        router.load_state_dict(ckpt["router_state"])
        logger.info(f"从 {args.init_from} 初始化（{ckpt.get('phase', '?')}）")
    base = get_vector(router)
    dim = len(base)
    logger.info(f"Router 参数量={dim}")

    rng = np.random.default_rng(42)

    def eval_candidate(vec, vm, gen, idx):
        # ES 评估即量化（红线3 修订）：候选 vec 先恢复 fp32，就地 int8 量化，
        # 用 int8 前向跑 episode——reward 就是 int8 下的真实表现，
        # 收敛的 base(fp32) 天然适合 int8 表达。base 更新在 fp32 空间。
        set_vector(router, vec)
        quantize_model_int8(router)
        out = run_router_episode(vm, encoder, router, heads, max_steps=200,
                                 curriculum=curriculum, seed=gen * 100 + idx)
        return out.reward, out

    best_reward = -1e9
    best_vec = base.copy()
    best_path = str(args.out).replace(".pt", "_best.pt")
    for gen in range(args.gens):
        t0 = time.time()
        noise = rng.standard_normal((args.pop, dim)).astype(np.float32)
        cands = [base + args.sigma * noise[i] for i in range(args.pop)]
        fitness = np.zeros(args.pop)
        with cf.ThreadPoolExecutor(max_workers=min(args.pop, len(vms))) as ex:
            futs = {ex.submit(eval_candidate, cands[i], vms[i % len(vms)], gen, i): i
                    for i in range(args.pop)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                fitness[i], _ = fut.result()

        ranks = np.argsort(np.argsort(fitness))
        top = max(2, args.pop // 3)
        rw = np.maximum(ranks - (args.pop - top), 0).astype(np.float32)
        rw = rw / rw.sum() if rw.sum() > 0 else np.ones(args.pop) / args.pop
        base = base + args.lr * args.sigma * (rw[:, None] * noise).sum(axis=0)

        set_vector(router, base)
        eval_out = run_router_episode(vms[0], encoder, router, heads,
                                      max_steps=200, curriculum=curriculum,
                                      seed=9999 + gen)
        logger.info(f"gen {gen+1}/{args.gens} 适应度 avg={fitness.mean():.1f} "
                    f"max={fitness.max():.1f}  eval: reward={eval_out.reward:.1f} "
                    f"tasks={eval_out.tasks_completed} switches={eval_out.switches} "
                    f"({time.time()-t0:.0f}s)")
        # 历史最佳跟踪：ES 在非凸地形上振荡，终点不等于最佳——
        # v3 gen7 tasks=3 后 gen8 回落 0，最佳 base 丢失的教训（2026-07-30）
        if eval_out.reward > best_reward:
            best_reward = eval_out.reward
            best_vec = base.copy()
            torch.save({"router_state": router.state_dict(), "gen": gen + 1,
                        "eval_reward": best_reward}, best_path)
            logger.info(f"  ↑ 新历史最佳 gen{gen+1} reward={best_reward:.1f} → 已保存")

    # 用历史最佳跑门3（而不是振荡终点）
    set_vector(router, best_vec)

    # 门3：任务完成率下限
    logger.info("=== 门3 验收 ===")
    evals = [run_router_episode(vms[i % len(vms)], encoder, router, heads,
                                max_steps=200, curriculum=curriculum, seed=777 + i)
             for i in range(3)]
    mean_tasks = float(np.mean([e.tasks_completed for e in evals]))
    logger.info(f"平均任务完成数={mean_tasks:.1f} (门={GATE3_MIN_TASKS})")
    passed = mean_tasks >= GATE3_MIN_TASKS
    logger.info(f"门3: {'PASS' if passed else 'FAIL（回炉：提高任务奖励权重或检查技能）'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"router_state": router.state_dict(),
                "gate3": {"mean_tasks": mean_tasks, "passed": passed}}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
