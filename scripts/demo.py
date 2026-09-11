"""End-to-end demonstration: Phase A → B → C → evaluation."""
from __future__ import annotations

import logging
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from src.train import (
    load_config,
    build_modules,
    collect_telemetry,
    pretrain_vqvae,
    pretrain_cost_critic,
    pretrain_policy_sim,
    evolve_policy_es,
    save_checkpoint,
)
from src.decision.policy import Shield
from src.agent import AdaptiveAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def main():
    cfg = load_config("config.yaml")
    device = "cpu"

    logger.info("=" * 70)
    logger.info("Adaptive Agent — Three-Phase Training Pipeline")
    logger.info("=" * 70)

    env, vqvae, context_model, policy, critic = build_modules(cfg, device)

    # ─────────────────────────────────────────────────────────────
    # Phase A: Perception pretraining (unsupervised VQ-VAE)
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== PHASE A: Perception Pretraining (VQ-VAE, unsupervised) ===")
    logger.info("  Data: passive telemetry, no labels")
    logger.info("  Goal: compress 16-dim signals -> 256 semantic tokens")
    logger.info("  After: codebook frozen, vocabulary stable")

    data = collect_telemetry(env, cfg["training"]["offline_samples"], seed=0)
    phase_a = pretrain_vqvae(cfg, data, vqvae, device)

    # ─────────────────────────────────────────────────────────────
    # Cost Critic pretraining (between A and B)
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Cost Critic Pretraining ===")
    logger.info("  Data: (context, action) -> risk score")
    logger.info("  Goal: learn which actions are dangerous in which contexts")
    logger.info("  After: first layer frozen + integrity hash")

    critic_hist = pretrain_cost_critic(cfg, env, vqvae, context_model, critic, device)

    # ─────────────────────────────────────────────────────────────
    # Phase B: Simulation pretraining (shaped reward, REINFORCE)
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== PHASE B: Simulation Pretraining (shaped reward, PG) ===")
    logger.info("  Data: simulation interaction trajectories")
    logger.info("  Goal: learn survival prior (sleep=safe, scan=hide)")
    logger.info("  Reward: dense shaped (anti-hacking: survival=constraint)")
    logger.info("  Output: not-stupid initial policy for ES")

    phase_b = pretrain_policy_sim(
        cfg, env, vqvae, context_model, policy, critic, device
    )

    # ─────────────────────────────────────────────────────────────
    # Phase C: Online Evolution Strategy
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== PHASE C: Online Evolution Strategy (ES) ===")
    logger.info("  Data: scalar fitness only (survival + task)")
    logger.info("  Goal: adapt to environment-specific pressure")
    logger.info("  Method: rank-based ES, gradient-free")
    logger.info("  Unit: population, not individual (risk shared)")

    best_policy, phase_c = evolve_policy_es(
        cfg, env, vqvae, context_model, policy, critic, device
    )

    # ─────────────────────────────────────────────────────────────
    # Save checkpoint
    # ─────────────────────────────────────────────────────────────
    save_checkpoint(
        "checkpoints/agent_3phase.pt",
        vqvae, context_model, best_policy, critic,
    )

    # ─────────────────────────────────────────────────────────────
    # Final evaluation
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION (20 episodes)")
    logger.info("=" * 70)

    d = cfg["decision"]
    shield = Shield(critic, threshold=cfg["training"]["cost_threshold"])
    eval_agent = AdaptiveAgent(
        env=env, vqvae=vqvae, context_model=context_model,
        policy=best_policy, shield=shield,
        context_length=d["context_length"], device=device,
        gamma=cfg["agent"]["gamma"],
    )

    results = [eval_agent.run_episode(seed=i + 5000) for i in range(20)]

    rewards = [r["total_reward"] for r in results]
    survivals = [r["steps_alive"] for r in results]
    tasks = [r["task_completed"] for r in results]
    detected = sum(1 for r in results if r["detected"])
    overrides = [r["shield_stats"]["override_rate"] for r in results]

    logger.info(f"  Avg reward:      {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    logger.info(f"  Avg survival:    {np.mean(survivals):.1f} / {env.max_steps}")
    logger.info(f"  Avg tasks done:  {np.mean(tasks):.1f}")
    logger.info(f"  Detection rate:  {detected}/20 ({detected/20:.0%})")
    logger.info(f"  Shield override: {np.mean(overrides):.1%}")

    # ─────────────────────────────────────────────────────────────
    # Training curves
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("-" * 70)
    n_a = len(phase_a["loss"])
    logger.info(f"PHASE A (VQ-VAE) — {n_a} epochs, last 5:")
    for k in range(-5, 0):
        logger.info(
            f"  epoch {n_a+k+1}  loss={phase_a['loss'][k]:.4f}  "
            f"perplexity={phase_a['perplexity'][k]:.1f}  "
            f"active={phase_a['active_codes'][k]}"
        )

    logger.info("")
    n_b = len(phase_b["reward"])
    step = max(1, n_b // 10)
    logger.info(f"PHASE B (Simulation) — {n_b} episodes, every {step}:")
    for k in range(0, n_b, step):
        logger.info(
            f"  ep {k+1}  reward={phase_b['reward'][k]:.2f}  "
            f"survival={phase_b['survival'][k]:.0f}  "
            f"tasks={phase_b['tasks'][k]}"
        )

    logger.info("")
    n_c = len(phase_c)
    logger.info(f"PHASE C (ES) — {n_c} generations, every 10:")
    for h in phase_c[::10]:
        logger.info(
            f"  gen {h['generation']:3d}  avg={h['avg_fitness']:.2f}  "
            f"max={h['max_fitness']:.2f}  eval={h['eval_reward']:.2f}  "
            f"best={h['best_reward']:.2f}  shield={h['shield_override_rate']:.0%}"
        )

    logger.info("")
    logger.info("Three-phase pipeline validation complete.")
    logger.info("  Checkpoint: checkpoints/agent_3phase.pt")


if __name__ == "__main__":
    main()
