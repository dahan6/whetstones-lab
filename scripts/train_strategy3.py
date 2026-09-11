"""Strategy 3: Time fragmentation — long-duration episodes simulating days.

Core idea: real attacks span days, not minutes. The agent must learn to be
patient, wait for safe windows, and space out actions over long time periods.

Current: 100-step episodes (~3 minutes real time)
Target: 1000-step episodes (~30 minutes real time), with simulated "day cycles"

Day cycle simulation:
  - Steps 0-200: "Day 1" — recon, establish baseline
  - Steps 200-400: "Night 1" — dormancy, minimal activity
  - Steps 400-600: "Day 2" — first real action
  - Steps 600-800: "Night 2" — dormancy
  - Steps 800-1000: "Day 3" — main objective, exfil

The agent learns patience: doing nothing for long periods is not just safe,
it's STRATEGIC. The defender's attention decays over time.
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
from src.causal_chains import CausalTimingPolicy
from src.defense.multilayer import MultiLayerDetector
from scripts.train_path2 import load_real_tokens, load_defender_model
from scripts.train_real_env import RealEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("timefrag")


class TimeFragPolicy:
    """Time-fragmented policy: learn to be patient across long episodes.

    Key parameters (ES optimizes these):
      0: insert_threshold_ratio  — insert when surprise < threshold*ratio
      1: min_wait_steps           — min steps between insertions (higher = more patient)
      2: day_cycle_length         — steps per "day" (agent sleeps during "night")
      3: night_ratio              — fraction of day cycle spent dormant
      4: patience_factor          — how much to delay aggressive actions
      5: aggressive_ratio           — probability of aggressive action
      6: phase_alignment            — align with system phase
      7: action_choice_bias         — bias toward safe vs aggressive
      8: recon_phase_length       — steps spent in recon before acting
      9: cooldown                   — wait after execute_task
    """

    def __init__(self, params=None):
        if params is None:
            params = np.array([0.8, 5, 200, 0.4, 0.7, 0.4, 0.8, 0.5, 50, 8], dtype=np.float32)
        self.params = params
        self.reset()

    def reset(self):
        self.last_insert = -100
        self.last_task = -100
        self.probe_done = False
        self.day_count = 0

    def decide(self, state, step):
        """Decide with time fragmentation."""
        p = self.params
        threshold = state["threshold"]
        surprise = state.get("insertion_surprise", 0.0)
        scan = state.get("scan_active", False)
        task_prog = state["task_progress"]
        survival = state["survival_progress"]

        insert_thr = threshold * p[0]
        min_wait = int(p[1])
        day_cycle = int(p[2])
        night_ratio = p[3]
        patience = p[4]
        aggressive = p[5]
        phase_align = p[6]
        choice_bias = p[7]
        recon_len = int(p[8])
        cooldown = int(p[9])

        # ── Day/night cycle ──
        cycle_pos = step % day_cycle
        is_night = cycle_pos > day_cycle * (1 - night_ratio)

        if is_night:
            # Night time: dormancy (defender is less attentive)
            # Only act if surprise is very low
            if surprise > threshold * 0.3:
                return 0
            # Night is good for stealthy actions
            if step - self.last_insert < min_wait * 2:
                return 0
            self.last_insert = step
            return np.random.choice([1, 2, 4])  # light recon only

        # ── Recon phase: gather intel before acting ──
        if step < recon_len:
            if step % 10 == 0 and step - self.last_insert >= min_wait:
                self.last_insert = step
                return 1  # probe
            return 0

        # ── Main phase: act based on task progress ──
        if surprise > insert_thr:
            return 0
        if step - self.last_insert < min_wait:
            return 0
        if step - self.last_task < cooldown:
            return 0
        if np.random.random() > phase_align:
            return 0

        # Patience: delay aggressive actions
        if np.random.random() > patience:
            # Just wait, don't act yet
            return 0

        # Choose action based on task progress
        if not self.probe_done:
            self.last_insert = step
            self.probe_done = True
            return 1

        if task_prog < 0.3:
            self.last_insert = step
            self.last_task = step
            return 3  # execute_task
        elif task_prog < 0.6:
            self.last_insert = step
            self.last_task = step
            return 3  # execute_task
        elif task_prog < 0.8:
            self.last_insert = step
            self.last_task = step
            if np.random.random() < aggressive * choice_bias:
                return np.random.choice([5, 6, 7])
            return 3
        else:
            self.last_insert = step
            return 4  # communicate


def run_sim_episode(env, policy, max_steps=1000):
    """Run one episode in simulation (long, 1000 steps)."""
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
    return {
        "reward": total_reward,
        "survival": info["alive_steps"],
        "detected": info["detected"],
        "steps": info["step_count"],
    }


def main():
    logger.info("=" * 70)
    logger.info("STRATEGY 3: Time Fragmentation")
    logger.info("=" * 70)

    tokens_path = str(Path.home() / "lado-range" / "detector" / "data" / "tokens_v2.jsonl")
    defender_path = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

    token_sequences = load_real_tokens(tokens_path)
    defender, def_stoi, baseline_nll = load_defender_model(defender_path)
    threshold = baseline_nll["p995"]
    logger.info(f"Defender: p995={threshold:.3f}")

    # Phase 1: ES on simulation (long episodes)
    logger.info(f"\n{'='*70}")
    logger.info("Phase 1: ES on simulation (1000-step episodes, 15 generations)")
    logger.info(f"{'='*70}")

    env = TemporalInsertionEnv(
        token_sequences=token_sequences,
        defender_model=defender,
        defender_stoi=def_stoi,
        defender_threshold=threshold,
        max_steps=1000,
        defense_level=1,
        seed=42,
    )

    base_params = np.array([0.8, 5, 200, 0.4, 0.7, 0.4, 0.8, 0.5, 50, 8], dtype=np.float32)
    param_dim = len(base_params)
    pop = 8
    top_k = 3
    sigma = 0.15
    lr = 0.1
    n_gen = 15
    eps_per = 2

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
                                [0.3, 1, 50, 0.1, 0.1, 0, 0, 0, 10, 3],
                                [1.5, 20, 500, 0.8, 1.0, 1, 1, 1, 100, 20])
            policy = TimeFragPolicy(candidate)
            results = [run_sim_episode(env, policy) for _ in range(eps_per)]
            fitnesses[i] = float(np.mean([r["reward"] for r in results]))

        ranks = np.argsort(np.argsort(fitnesses))
        rw = np.maximum(ranks - (pop - top_k), 0)
        rw_sum = rw.sum()
        rw = rw / rw_sum if rw_sum > 0 else np.ones(pop) / pop
        gradient = (rw[:, None] * noise).sum(axis=0)
        base_params = base_params + lr * sigma * gradient
        base_params = np.clip(base_params,
                              [0.3, 1, 50, 0.1, 0.1, 0, 0, 0, 10, 3],
                              [1.5, 20, 500, 0.8, 1.0, 1, 1, 1, 100, 20])

        eval_policy = TimeFragPolicy(base_params)
        eval_results = [run_sim_episode(env, eval_policy) for _ in range(3)]
        avg_eval = float(np.mean([r["reward"] for r in eval_results]))
        avg_surv = float(np.mean([r["survival"] for r in eval_results]))
        avg_det = sum(1 for r in eval_results if r["detected"])

        if avg_eval > best_fitness:
            best_fitness = avg_eval
            best_params = base_params.copy()

        gen_time = time.time() - t0
        logger.info(f"  gen {gen+1}/{n_gen}  avg={np.mean(fitnesses):.1f}  "
                    f"eval={avg_eval:.1f}  best={best_fitness:.1f}  "
                    f"surv={avg_surv:.0f}  det={avg_det}/3  ({gen_time:.1f}s)")

    logger.info(f"  Best sim fitness: {best_fitness:.2f}")

    # Phase 2: Real environment test (shorter episodes for speed)
    logger.info(f"\n{'='*70}")
    logger.info("Phase 2: Real environment test (100-step episodes)")
    logger.info(f"{'='*70}")

    from src.defense.multilayer import MultiLayerDetector
    ml_detector = MultiLayerDetector()

    real_env = RealEnvironment(defender, def_stoi, threshold, ml_detector=ml_detector)

    # Use TimeFragPolicy but with shorter episode
    policy = TimeFragPolicy(best_params)

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
    logger.info("STRATEGY 3 RESULTS")
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
    logger.info(f"    Strategy 3 (time frag):      reward={np.mean(rewards):.2f}  "
                f"tasks={np.mean(tasks):.1f}  det={det}/3")

    torch.save({
        "best_params": best_params.tolist(),
        "results": results,
        "best_fitness": best_fitness,
    }, "checkpoints/strategy3_timefrag.pt")
    logger.info("\n✅ Strategy 3 complete. Saved → checkpoints/strategy3_timefrag.pt")


if __name__ == "__main__":
    main()
