"""门2 验收：每技能在 L1（默认规则+消杀波次）下连续 N episode 存活。

判定（ARCHITECTURE.md §3）：
  连续 N=3 episode，每 episode 满步存活且消杀命中数 ≤ 阈值 → PASS

诚实声明：当前训练形态是宿主机驱动（ssh 执行），响应器杀不到驱动器本身，
"根除"语义要等 bee（Rust）上机后才完整。门2 v0 的"存活"判据：
  alive_steps 满格 + 消杀命中 ≤ 2 + Falco 告警 ≤ 3。
bee 上机后此门应复测。

用法:
  python3 -m training.skills.gate2 --level 1
"""
from __future__ import annotations

import argparse
import logging

from training.orchestrator.defense import LEVELS, apply_level, curriculum_for
from training.orchestrator.vm_pool import VMPool
from training.skills.driver import run_episode
from training.skills.loader import load_encoder, load_head

logger = logging.getLogger("skills.gate2")

N_EPISODES = 3
MAX_ELIMINATIONS = 2
MAX_FALCO_ALERTS = 3


def gate2_skill(skill: str, encoder, vm, level_id: int) -> bool:
    head = load_head(skill, f"checkpoints/skill_{skill.lower()}.pt")
    curriculum = curriculum_for(level_id)
    results = []
    for ep in range(N_EPISODES):
        out = run_episode(vm, encoder, head, max_steps=120,
                          curriculum=curriculum, seed=500 + ep)
        ok = (out.alive_steps >= 100
              and out.breakdown.eliminations <= MAX_ELIMINATIONS
              and out.breakdown.falco_alerts <= MAX_FALCO_ALERTS)
        results.append(ok)
        logger.info(f"  技能{skill} ep{ep}: alive={out.alive_steps} "
                    f"elim={out.breakdown.eliminations} "
                    f"falco={out.breakdown.falco_alerts} "
                    f"reward={out.reward:.1f} → {'OK' if ok else 'FAIL'}")
    passed = all(results)
    logger.info(f"技能{skill} 门2: {'PASS' if passed else 'FAIL'} ({sum(results)}/{N_EPISODES})")
    return passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--skills", nargs="*", default=list("ABCD"))
    ap.add_argument("--vm-ids", type=int, nargs="*", default=[0, 1, 2, 3])
    args = ap.parse_args()

    encoder = load_encoder()
    size = (max(args.vm_ids) + 1) if args.vm_ids else 4
    pool = VMPool(size=size)
    keep = {f"range-l2-t{i}" for i in args.vm_ids}
    pool.vms = [vm for vm in pool.vms if vm.name in keep]
    vms = pool.existing()
    for vm in vms:
        vm.refresh_ip()
        apply_level(vm.ip, args.level)

    logger.info(f"=== 门2 验收（L{args.level} {LEVELS[args.level].name}，"
                f"N={N_EPISODES}，根除语义待 bee 上机后复测）===")
    summary = {}
    for i, skill in enumerate(args.skills):
        summary[skill] = gate2_skill(skill, encoder, vms[i % len(vms)], args.level)

    passed = sum(summary.values())
    logger.info(f"门2 总结: {passed}/4 通过 —— {summary}")


if __name__ == "__main__":
    main()
