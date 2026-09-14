#!/usr/bin/env python3
"""蓝方进化驱动器 v0：眼基因组锦标赛选择。

与红方 toxoplasma/evolve.py 完全同构的进化机制，
但适应度 = 检出率 - 10×误报率（只读环境结局）。

用法:
  cd ~/defense-lab && /usr/bin/python3 -m detector.eye.blue_evolve --generations 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import time
from pathlib import Path

logger = logging.getLogger("eye.evolve")
ROOT = Path(__file__).parent

# 数值位：{key_path: (相对变异量, 下界, 上界)}
NUMERIC_LOCI = {
    "faces.P1_pattern.match_threshold": (0.10, 0.5, 1.0),
    "faces.P2_context.tau_percentile": (0.05, 95.0, 99.9),
    "faces.P4_temporal.cv_low": (0.10, 0.8, 2.5),
    "faces.P4_temporal.min_events": (0.15, 5, 50),
    "faces.P6_burst.threshold": (0.10, 0.3, 0.8),
    "faces.P7_seqood.threshold": (0.10, 3.0, 6.0),
    "faces.P8_anatomy.l1_threshold": (0.10, 0.5, 0.9),
    "faces.P8_anatomy.l2_sim_floor": (0.02, 0.95, 0.999),
    "faces.P8_anatomy.l3_min_files": (0.20, 2, 8),
    "faces.P8_anatomy.l3_window_s": (0.20, 20, 300),
    "models.meta.danger_threshold": (0.15, 0.01, 0.5),
    "models.threat_memory.distance_max": (0.10, 0.5, 5.0),
    "windows.hunt_period_s": (0.15, 30, 300),
    "windows.wave_period_s": (0.15, 120, 1800),
    "windows.rescan_delay_s": (0.15, 0, 300),
    "windows.chain_window_s": (0.15, 60, 600),
    "hunt.kill_wipe": (0.15, 0, 1),
    "hunt.freeze_quarantine": (0.15, 0, 1),
    "hunt.evict_landing": (0.15, 0, 1),
    "hunt.cooldown_s": (0.15, 60, 600),
}

# 离散位：面/器官开关
SWITCH_LOCI = [
    "faces.P0_adaptive.on", "faces.P1_pattern.on", "faces.P2_context.on",
    "faces.P3_rarity.on", "faces.P4_temporal.on", "faces.P5_adaptive_low.on",
    "faces.P6_burst.on", "faces.P7_seqood.on", "faces.P8_anatomy.on",
    "organs.skewness.on", "organs.ks_distance.on", "organs.cross_host.on",
    "models.threat_memory.on",
]


def get_nested(d, path):
    keys = path.split(".")
    for k in keys:
        d = d.get(k, {})
    return d


def set_nested(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def mutate(genome, rng):
    g = json.loads(json.dumps(genome))
    for path, (rel, lo, hi) in NUMERIC_LOCI.items():
        if rng.random() < 0.8:
            val = get_nested(g, path)
            if isinstance(val, (int, float)) and val is not None:
                new_val = val * (1 + rng.gauss(0, rel))
                if isinstance(val, int):
                    new_val = int(max(lo, min(hi, new_val)))
                else:
                    new_val = round(max(lo, min(hi, new_val)), 4)
                set_nested(g, path, new_val)
    for path in SWITCH_LOCI:
        if rng.random() < 0.10:
            val = get_nested(g, path)
            if isinstance(val, bool):
                set_nested(g, path, not val)
    return g


def fitness_from_results(detections, false_positives, total_deployments, total_benign_windows):
    det_rate = detections / max(1, total_deployments)
    fpr = false_positives / max(1, total_benign_windows)
    return det_rate - 10 * fpr


def run_generation(pop, rng, gen):
    """一代评估。当前 v0 是离线模拟——靶场轮次待接入。"""
    results = []
    for i, genome in enumerate(pop):
        # 离线评估（用威胁记忆库做代理）
        # 真实版需要接入靶场
        score = rng.gauss(0.5, 0.15)  # placeholder
        results.append((score, i, genome))
    results.sort(key=lambda x: -x[0])
    return results


def select_next_gen(results, pop, rng, pop_size):
    if not results:
        return pop
    best_score = results[0][0]
    tier_a = [pop[i] for score, i, _ in results if score >= 0.8 * best_score]
    tier_b = [pop[i] for score, i, _ in results if 0.5 * best_score <= score < 0.8 * best_score]
    parents = tier_a + tier_b if (tier_a or tier_b) else [pop[0]]
    nxt = [json.loads(json.dumps(pop[results[0][1]]))]  # elite
    i = 0
    while len(nxt) < pop_size:
        parent = parents[i % len(parents)]
        child = mutate(parent, rng)
        nxt.append(child)
        i += 1
    return nxt


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = json.load(open(ROOT / "eye.json"))
    pop = [json.loads(json.dumps(base)) for _ in range(args.pop_size)]
    # 第一代变异
    pop = [mutate(p, rng) if i > 0 else p for i, p in enumerate(pop)]

    for gen in range(1, args.generations + 1):
        logger.info(f"=== 第 {gen} 代: {len(pop)} 个体 ===")
        results = run_generation(pop, rng, gen)
        scores = [(i, round(s, 3)) for s, i, _ in results[:3]]
        logger.info(f"  Top 3: {scores}")
        pop = select_next_gen(results, pop, rng, args.pop_size)

    logger.info("=== 蓝方进化 v0 完成（离线模拟） ===")
    logger.info("注意: 当前是骨架版。接入靶场后 fitness 从环境结局读出。")


if __name__ == "__main__":
    main()
