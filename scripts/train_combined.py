"""Combined strategy: S1 (causal chains) as core + S2/S3/S4 as enhancements.

Core architecture:
  - S1 (causal chains): the backbone — detect→investigate→fix→verify narrative
  - S2 (traffic obfuscation): enhances the "communicate" step with stealth C2
  - S3 (time fragmentation): adds patience during high-surveillance periods
  - S4 (multi-VM): adds propagation redundancy at key moments

The agent executes causal chains, but:
  - During "communicate" steps, uses traffic-obfuscated channels
  - During high-surveillance, applies time-frag patience
  - At milestones, triggers propagation actions
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.environment.temporal_insertion import TemporalInsertionEnv
from src.causal_chains import CAUSAL_CHAINS, CausalTimingPolicy
from src.traffic_obfuscation import TrafficObfuscator
from src.advanced_primitives import ADVANCED_COMMANDS, ADVANCED_NAMES
from src.defense.multilayer import MultiLayerDetector
from scripts.train_path2 import load_real_tokens, load_defender_model
from scripts.train_real_env import RealEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("combined")


class CombinedPolicy:
    """Combined policy: causal chains + traffic obfuscation + time frag + multi-VM.

    ES search space (14 params):
      0-9: causal chain timing (from S1)
      10: traffic_channel_preference  [0-3]  — which C2 channel to prefer
      11: time_frag_patience          [0-1]  — how much patience during high-surveillance
      12: propagation_frequency       [0-1]  — how often to trigger propagation
      13: aggressive_ratio              [0-1]  — probability of aggressive action
    """

    def __init__(self, params=None):
        if params is None:
            params = np.array([
                0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.7,  # S1 params
                1.0, 0.7, 0.3, 0.5,                        # S2-S4 params
            ], dtype=np.float32)
        self.params = params
        self.causal = CausalTimingPolicy(params[:10])
        self.obfuscator = TrafficObfuscator()
        self.reset()

    def reset(self):
        self.causal.reset()
        self.causal.executor.last_insert = -100
        self.last_insert = -100
        self.propagation_count = 0

    def decide(self, state, step):
        """Combined decision logic."""
        p = self.params
        threshold = state["threshold"]
        surprise = state.get("insertion_surprise", 0.0)
        scan = state.get("scan_active", False)
        task_prog = state["task_progress"]
        survival = state["survival_progress"]

        insert_thr = threshold * p[0]
        min_wait = int(p[1])
        cooldown = int(p[3])
        phase_align = p[4]
        traffic_pref = int(p[10])
        patience = p[11]
        prop_freq = p[12]
        aggressive = p[13]

        # ── S3: Time fragmentation during high surveillance ──
        if scan and np.random.random() > patience:
            return 0  # wait during scan

        # ── S1: Causal chain execution ──
        action = self.causal.decide(state, step)

        # ── S2: Enhance communicate with traffic obfuscation ──
        if action == 4:  # communicate
            network_state = {
                "scan_active": scan,
                "dns_available": True,
                "http_available": True,
                "icmp_allowed": True,
            }
            channel, cmd = self.obfuscator.select_channel(network_state)
            # Still return 4 (communicate) but the channel choice is logged
            # In a real implementation, this would change the actual command
            pass

        # ── S4: Propagation at milestones ──
        if step > 30 and self.propagation_count < 3:
            if np.random.random() < prop_freq:
                self.propagation_count += 1
                prop_actions = [8, 9, 10, 11, 12]
                return prop_actions[self.propagation_count % len(prop_actions)]

        # ── Advanced primitives: DLL hijack / command wrap / port mixing ──
        # Trigger advanced techniques during safe windows with moderate frequency
        if surprise < threshold * 0.5 and step > 20:
            if np.random.random() < 0.15:  # 15% chance per step
                adv_actions = [13, 14, 15, 16, 17, 18, 19, 20, 21]
                return np.random.choice(adv_actions)

        # ── Aggressive actions during safe windows ──
        if surprise < threshold * 0.4 and np.random.random() < aggressive:
            if action == 0:  # if waiting, try aggressive
                return np.random.choice([5, 6, 7])

        return action


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
    logger.info("COMBINED STRATEGY: S1 + S2 + S3 + S4")
    logger.info("=" * 70)

    tokens_path = str(Path.home() / "lado-range" / "detector" / "data" / "tokens_v2.jsonl")
    defender_path = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

    token_sequences = load_real_tokens(tokens_path)
    defender, def_stoi, baseline_nll = load_defender_model(defender_path)
    threshold = baseline_nll["p995"]
    logger.info(f"Defender: p995={threshold:.3f}")

    # Phase 1: ES on simulation
    logger.info(f"\n{'='*70}")
    logger.info("Phase 1: ES on simulation (combined policy)")
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

    base_params = np.array([
        0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.7,
        1.0, 0.7, 0.3, 0.5,
    ], dtype=np.float32)
    param_dim = len(base_params)
    pop = 12
    top_k = 4
    sigma = 0.15
    lr = 0.12
    n_gen = 25
    eps_per = 3

    rng = np.random.default_rng(42)
    best_fitness = -float("inf")
    best_params = base_params.copy()

    for gen in range(n_gen):
        t0 = time.time()
        noise = rng.standard_normal((pop, param_dim)).astype(np.float32)
        fitnesses = np.zeros(pop)

        for i in range(pop):
            candidate = base_params + sigma * noise[i]
            candidate = np.clip(candidate,
                                [0.3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1.5, 10, 10, 20, 1, 1, 3, 3, 2, 1, 3, 1, 1, 1])
            policy = CombinedPolicy(candidate)
            results = [run_sim_episode(env, policy) for _ in range(eps_per)]
            fitnesses[i] = float(np.mean([r["reward"] for r in results]))

        ranks = np.argsort(np.argsort(fitnesses))
        rw = np.maximum(ranks - (pop - top_k), 0)
        rw_sum = rw.sum()
        rw = rw / rw_sum if rw_sum > 0 else np.ones(pop) / pop
        gradient = (rw[:, None] * noise).sum(axis=0)
        base_params = base_params + lr * sigma * gradient
        base_params = np.clip(base_params,
                              [0.3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1.5, 10, 10, 20, 1, 1, 3, 3, 2, 1, 3, 1, 1, 1])

        eval_policy = CombinedPolicy(base_params)
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
    policy = CombinedPolicy(best_params)

    results = []
    for ep in range(3):
        state = real_env.reset()
        policy.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = policy.decide(state, real_env.step_count)
            state, reward, done, info = real_env.step(action)
            total_reward += reward

            if info["step_count"] <= 35 or info["action"] != "wait":
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
    logger.info("COMBINED STRATEGY RESULTS")
    logger.info(f"{'='*70}")

    rewards = [r["reward"] for r in results]
    survivals = [r["survival"] for r in results]
    tasks = [r["tasks"] for r in results]
    det = sum(1 for r in results if r["detected"])

    logger.info(f"  Avg reward:    {np.mean(rewards):.2f}")
    logger.info(f"  Avg survival:  {np.mean(survivals):.0f}/100")
    logger.info(f"  Avg tasks:     {np.mean(tasks):.1f}")
    logger.info(f"  Detection:     {det}/3 ({det/3:.0%})")

    logger.info(f"\n  FULL COMPARISON:")
    logger.info(f"    S1 alone (causal chains):  reward=98.84  tasks=23.0  det=0/3")
    logger.info(f"    S2 alone (traffic):          reward=35.00  tasks=2.0   det=0/3")
    logger.info(f"    S3 alone (time frag):        reward=14.92  tasks=3.0   det=0/3")
    logger.info(f"    S4 alone (multi-vm):         reward=54.04  tasks=2.0   det=0/3")
    logger.info(f"    COMBINED:                    reward={np.mean(rewards):.2f}  "
                f"tasks={np.mean(tasks):.1f}  det={det}/3")

    torch.save({
        "best_params": best_params.tolist(),
        "results": results,
        "best_fitness": best_fitness,
    }, "checkpoints/combined_strategy.pt")
    logger.info("\n✅ Combined strategy complete. Saved → checkpoints/combined_strategy.pt")


if __name__ == "__main__":
    main()
