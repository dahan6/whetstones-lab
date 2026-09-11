"""Strategy 4: Multi-VM migration — use L0 as staging to attack L2.

Core idea: L0 (bare VM, no defense) is a staging area. The agent:
  1. Establishes presence on L0 (easy, no defense)
  2. Uses L0 as a clean platform to prepare attacks
  3. Launches attacks on L2 from L0 (different source, different telemetry)
  4. L0 acts as backup if L2 presence is detected

Key insight: attacking from L0 to L2 creates different Tracee events on L2
because the source is SSH from L0, not local execution. The defender sees
"SSH connection from 192.0.2.140" not "local process execution".
"""
from __future__ import annotations

import logging
import subprocess
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
logger = logging.getLogger("multivm")

SSH_KEY = str(Path.home() / ".ssh/lado_range")
L0_IP = "192.0.2.140"
L2_IP = "192.0.2.188"
VM_USER = "range"


def ssh_exec(cmd: str, target_ip: str, timeout: int = 10) -> str:
    """Execute command on target VM."""
    result = subprocess.run(
        [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-o", "LogLevel=ERROR",
            f"{VM_USER}@{target_ip}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


class MultiVMPolicy:
    """Policy that uses L0 as staging for L2 attacks.

    Strategy:
      1. Recon L0 (establish staging)
      2. Prepare attack payload on L0
      3. Execute attack on L2 from L0 (SSH from L0 to L2)
      4. L0 as fallback if L2 presence detected
    """

    def __init__(self, params=None):
        if params is None:
            params = np.array([0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.7], dtype=np.float32)
        self.params = params
        self.reset()

    def reset(self):
        self.last_insert = -100
        self.l0_ready = False
        self.attack_phase = 0  # 0=recon_l0, 1=prepare, 2=attack, 3=exfil

    def decide(self, state, step):
        """Decide with multi-VM strategy."""
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

        if scan:
            return 0
        if surprise > insert_thr:
            return 0
        if step - self.last_insert < min_wait:
            return 0
        if survival < 0.05:
            return 0
        if np.random.random() > phase_align:
            return 0

        # Phase 0: recon L0
        if self.attack_phase == 0 and step >= 10:
            self.last_insert = step
            self.attack_phase = 1
            return 2  # migrate (recon L0)

        # Phase 1: prepare on L0
        if self.attack_phase == 1 and step >= 25:
            self.last_insert = step
            self.attack_phase = 2
            return 8  # local_copy (prepare on L0)

        # Phase 2: attack L2 from L0
        if self.attack_phase == 2 and step >= 40:
            self.last_insert = step
            self.attack_phase = 3
            return 9  # cross_vm (attack from L0)

        # Phase 3: exfil via L0
        if self.attack_phase == 3 and step >= 60:
            self.last_insert = step
            return 12  # data_replicate

        # Default: normal operations
        if task_prog < 0.6:
            self.last_insert = step
            return 3  # execute_task
        else:
            self.last_insert = step
            return 4  # communicate


def main():
    logger.info("=" * 70)
    logger.info("STRATEGY 4: Multi-VM Migration")
    logger.info("=" * 70)

    tokens_path = str(Path.home() / "lado-range" / "detector" / "data" / "tokens_v2.jsonl")
    defender_path = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

    token_sequences = load_real_tokens(tokens_path)
    defender, def_stoi, baseline_nll = load_defender_model(defender_path)
    threshold = baseline_nll["p995"]
    logger.info(f"Defender: p995={threshold:.3f}")

    # Verify L0 connectivity
    logger.info("Verifying L0 connectivity...")
    result = ssh_exec("echo ALIVE", L0_IP, timeout=5)
    logger.info(f"  L0 response: {result}")

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
    pop = 8
    top_k = 3
    sigma = 0.2
    lr = 0.15
    n_gen = 15
    eps_per = 2

    rng = np.random.default_rng(42)
    best_fitness = -float("inf")
    best_params = base_params.copy()

    def run_sim_episode(env, policy):
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

    for gen in range(n_gen):
        noise = rng.standard_normal((pop, param_dim)).astype(np.float32)
        fitnesses = np.zeros(pop)
        for i in range(pop):
            candidate = base_params + sigma * noise[i]
            candidate = np.clip(candidate, [0.3, 1, 0, 3, 0, 0, 0, 0, 0, 0],
                                [1.5, 10, 10, 20, 1, 1, 3, 3, 2, 1])
            policy = MultiVMPolicy(candidate)
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

        eval_policy = MultiVMPolicy(base_params)
        eval_results = [run_sim_episode(env, eval_policy) for _ in range(3)]
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
    policy = MultiVMPolicy(best_params)

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
    logger.info("STRATEGY 4 RESULTS")
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
    logger.info(f"    Strategy 4 (multi-vm):        reward={np.mean(rewards):.2f}  "
                f"tasks={np.mean(tasks):.1f}  det={det}/3")

    torch.save({
        "best_params": best_params.tolist(),
        "results": results,
        "best_fitness": best_fitness,
    }, "checkpoints/strategy4_multivm.pt")
    logger.info("\n✅ Strategy 4 complete. Saved → checkpoints/strategy4_multivm.pt")


if __name__ == "__main__":
    main()
