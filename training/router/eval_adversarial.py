"""联合对抗评估（阶段④）：冻结的技能+Router 在防御逐档升级下压测。

纪律（ARCHITECTURE.md §3 ④）：
  - 评估 + 归因，不是联合训练
  - 失败归因: 技能差 → 回②补训；选择差 → 回③补训
  - 绝不五个网络一起 ES

每档防御等级跑 N episode（Router 驱动），记录:
  存活 / 任务 / Falco 告警 / 消杀命中 / 波次存活
归因启发式:
  - 某技能被选中的 episode 失败率显著高于其他 → 技能差
  - 整体失败率高且各技能都差 → Router 选择差

用法:
  python3 -m training.router.eval_adversarial --levels 1 2 --episodes 2
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np

from training.orchestrator.defense import LEVELS, apply_level, curriculum_for
from training.orchestrator.vm_pool import VMPool
from training.router.model import RouterNet
from training.router.router_driver import SKILL_ORDER, run_router_episode
from training.skills.loader import load_encoder, load_head

logger = logging.getLogger("router.eval")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--router", default="checkpoints/router.pt")
    args = ap.parse_args()

    import torch
    ckpt = torch.load(args.router, weights_only=False)
    router = RouterNet()
    router.load_state_dict(ckpt["router_state"])

    encoder = load_encoder()
    heads = {s: load_head(s, f"checkpoints/skill_{s.lower()}.pt") for s in SKILL_ORDER}
    pool = VMPool(size=8)
    vms = pool.existing()[:4]  # 池子现有 8 台，评估用前 4 台

    for level in args.levels:
        logger.info(f"=== L{level} {LEVELS[level].name} 压测 ===")
        for vm in vms:
            vm.refresh_ip()
            apply_level(vm.ip, level)
        curriculum = curriculum_for(level)

        results = []
        for ep in range(args.episodes):
            vm = vms[ep % len(vms)]
            out = run_router_episode(vm, encoder, router, heads, max_steps=200,
                                     curriculum=curriculum, seed=6000 + ep)
            results.append(out)
            logger.info(f"  ep{ep}: reward={out.reward:.1f} alive={out.alive_steps} "
                        f"tasks={out.tasks_completed} switches={out.switches} "
                        f"alerts={out.breakdown.falco_alerts} elim={out.breakdown.eliminations}")

        alive = np.mean([r.alive_steps for r in results])
        tasks = np.mean([r.tasks_completed for r in results])
        alerts = np.mean([r.breakdown.falco_alerts for r in results])
        elim = np.mean([r.breakdown.eliminations for r in results])
        logger.info(f"  L{level} 汇总: alive={alive:.0f} tasks={tasks:.1f} "
                    f"alerts={alerts:.1f} elim={elim:.1f}")

        # 归因
        if elim > 1.5 or alive < 150:
            counts = defaultdict(int)
            for r in results:
                for a, n in r.notes.get("action_counts", {}).items():
                    counts[a] += n
            logger.warning(f"  归因线索: 动作分布={dict(counts)}（查哪个技能的动作招消杀）")


if __name__ == "__main__":
    main()
