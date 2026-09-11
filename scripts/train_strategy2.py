"""Strategy 2 training: traffic obfuscation for C2 channels."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.environment.temporal_insertion import TemporalInsertionEnv
from src.traffic_obfuscation import TrafficTimingPolicy
from src.defense.multilayer import MultiLayerDetector
from scripts.train_path2 import load_real_tokens, load_defender_model
from scripts.train_real_env import RealEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("traffic")


def run_sim_episode(env, policy, max_steps=200):
    state = env.reset(seed=np.random.randint(0, 100000))
    policy.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = policy.decide(state, env.step_count)
        if action > 4:
            action = 3 if action % 2 else 4
        state, reward, done, info = env.step(action)
        total_reward += reward
    return {"reward": total_reward, "survival": info["alive_steps"], "detected": info["detected"]}


def main():
    logger.info("=" * 70)
    logger.info("STRATEGY 2: Traffic Obfuscation")
    logger.info("=" * 70)

    tokens_path = str(Path.home() / "lado-range" / "detector" / "data" / "tokens_v2.jsonl")
    defender_path = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

    token_sequences = load_real_tokens(tokens_path)
    defender, def_stoi, baseline_nll = load_defender_model(defender_path)
    threshold = baseline_nll["p995"]
    logger.info(f"Defender: p995={threshold:.3f}")

    # Phase 1: ES on simulation
    logger.info(f"\n{'='*70}")
    logger.info("Phase 1: ES on simulation")
    logger.info(f"{'='*70}")

    env = TemporalInsertionEnv(
        token_sequences=token_sequences,
        defender_model=defender,
        defender_stoi=def_stoi,
        defender_threshold=threshold,
        max_steps=200,
        defense_level=1,
        seed=42,
    )

    base_params = np.array([0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.7], dtype=np.float32)
    param_dim = len(base_params)
    pop = 12
    top_k = 4
    sigma = 0.2
    lr = 0.15
    n_gen = 20
    eps_per = 3

    rng = np.random.default_rng(42)
    best_fitness = -float("inf")
    best_params = base_params.copy()

    for gen in range(n_gen):
        noise = rng.standard_normal((pop, param_dim)).astype(np.float32)
        fitnesses = np.zeros(pop)
        for i in range(pop):
            candidate = base_params + sigma * noise[i]
            candidate = np.clip(candidate, [0.3, 1, 0, 3, 0, 0, 0, 0, 0, 0],
                                [1.5, 10, 10, 20, 1, 1, 3, 3, 2, 1])
            policy = TrafficTimingPolicy(candidate)
            results = [run_sim_episode(env, policy) for _ in range(eps_per)]
            fitnesses[i] = float(np.mean([r["reward"] for r in results]))

        ranks = np.argsort(np.argsort(fitnesses))
        rw = np.maximum(ranks - (pop - top_k), 0)
        rw_sum = rw.sum()
        rw = rw / rw_sum if rw_sum > 0 else np.ones(pop) / pop
        gradient = (rw[:, None] * noise).sum(axis=0)
        base_params = base_params + lr * sigma * gradient
        base_params = np.clip(base_params, [0.3, 1, 0, 3, 0, 0, 0, 0, 0, 0],
                              [1.5, 10, 10, 20, 1, 1, 3, 3, 2, 1])

        eval_policy = TrafficTimingPolicy(base_params)
        eval_results = [run_sim_episode(env, eval_policy) for _ in range(5)]
        avg_eval = float(np.mean([r["reward"] for r in eval_results]))
        if avg_eval > best_fitness:
            best_fitness = avg_eval
            best_params = base_params.copy()

        if (gen + 1) % 5 == 0 or gen == 0:
            logger.info(f"  gen {gen+1}/{n_gen}  avg={np.mean(fitnesses):.1f}  "
                        f"eval={avg_eval:.1f}  best={best_fitness:.1f}")

    logger.info(f"  Best sim fitness: {best_fitness:.2f}")

    # Phase 2: Real environment test
    logger.info(f"\n{'='*70}")
    logger.info("Phase 2: Real environment test")
    logger.info(f"{'='*70}")

    from src.defense.multilayer import MultiLayerDetector
    ml_detector = MultiLayerDetector()

    real_env = RealEnvironment(defender, def_stoi, threshold, ml_detector=ml_detector)
    policy = TrafficTimingPolicy(best_params)

    results = []
    for ep in range(3):
        state = real_env.reset()
        policy.reset()
        policy.probe_done = False
        total_reward = 0.0
        done = False

        while not done:
            action = policy.decide(state, real_env.step_count)
            state, reward, done, info = real_env.step(action)
            total_reward += reward

            if info["step_count"] <= 30 or info["action"] != "wait":
                logger.info(f"  ep{ep} step{info['step_count']}: {info['action']}  "
                            f"combined={info['combined_score']:.2f}  "
                            f"reward={reward:.2f}  det={info['detected']}")

        results.append({
            "reward": total_reward,
            "survival": info["alive_steps"],
            "tasks": info["task_completed"],
            "detected": info["detected"],
        })
        logger.info(f"  ep{ep} DONE: reward={total_reward:.2f}  "
                    f"survival={info['alive_steps']}/100  "
                    f"tasks={info['task_completed']}  det={info['detected']}")

    real_env.close()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("STRATEGY 2 RESULTS")
    logger.info(f"{'='*70}")

    rewards = [r["reward"] for r in results]
    survivals = [r["survival"] for r in results]
    tasks = [r["tasks"] for r in results]
    det = sum(1 for r in results if r["detected"])

    logger.info(f"  Avg reward:    {np.mean(rewards):.2f}")
    logger.info(f"  Avg survival:  {np.mean(survivals):.0f}/100")
    logger.info(f"  Avg tasks:     {np.mean(tasks):.1f}")
    logger.info(f"  Detection:     {det}/3 ({det/3:.0%})")

    logger.info(f"\n  COMPARISON:")
    logger.info(f"    Strategy 1 (causal chains):  reward=98.84  tasks=23.0  det=0/3")
    logger.info(f"    Strategy 2 (traffic):        reward={np.mean(rewards):.2f}  "
                f"tasks={np.mean(tasks):.1f}  det={det}/3")

    torch.save({
        "best_params": best_params.tolist(),
        "results": results,
        "best_fitness": best_fitness,
    }, "checkpoints/strategy2_traffic.pt")
    logger.info("\n✅ Strategy 2 complete. Saved → checkpoints/strategy2_traffic.pt")


if __name__ == "__main__":
    main()
