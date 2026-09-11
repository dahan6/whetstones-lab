"""Micro NN for TIMING only (not action selection).

The causal chain determines WHAT to do (content).
The micro NN determines WHEN to do it (timing).

This is the correct hierarchical split:
  - Causal chain: "execute_task is the right action now"
  - Micro NN: "but is the current surprise level safe enough to execute?"

The NN learns: given (surprise, scan, phase, task_progress), should we
insert an action now or wait? The causal chain provides the action.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.environment.temporal_insertion import TemporalInsertionEnv
from src.causal_chains import CausalTimingPolicy
from src.defense.multilayer import MultiLayerDetector
from scripts.train_path2 import load_real_tokens, load_defender_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("micro_timing")


class TimingNet(nn.Module):
    """Tiny NN that learns WHEN to insert (not WHAT to insert).

    Input: 8-dim state (surprise, scan, phase, task, survival, step, action, bias)
    Output: 2 classes (wait=0, insert=1)

    Parameters: 8*16 + 16 + 16*2 + 2 = 178 params = ~712 bytes
    """

    def __init__(self, state_dim=8, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # wait vs insert

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def should_insert(self, state: np.ndarray) -> bool:
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return bool(probs[0, 1] > 0.5)

    def get_flat_params(self) -> np.ndarray:
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat[idx:idx+n]).view_as(p))
            idx += n


def timing_state(info: dict, threshold: float, proposed_action: int) -> np.ndarray:
    """8-dim state for timing decision."""
    return np.array([
        info.get("surprise_ewma", 0.0) / threshold,
        info.get("insertion_surprise", 0.0) / threshold,
        info.get("scan_active", 0.0),
        info.get("task_progress", 0.0),
        info.get("survival_progress", 0.0),
        info.get("step_count", 0) / 100.0,
        proposed_action / 4.0,
        1.0,
    ], dtype=np.float32)


def collect_timing_data(env, policy, n_episodes=1000):
    """Collect (state, should_insert) pairs from causal chain expert.

    For each step, the causal chain proposes an action.
    We record whether that action was actually safe (surprise < threshold).
    The NN learns to predict: "given this state, is it safe to insert?"
    """
    states = []
    labels = []
    rewards = []

    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        policy.reset()
        done = False

        while not done:
            action = policy.decide(state, env.step_count)
            if action > 4:
                action = 3 if action % 2 else 4

            # Record state BEFORE action
            pre_state = timing_state(state, env.def_threshold, action)

            # Execute and observe
            state, reward, done, info = env.step(action)

            # Label: was the INSERTION stealthy?
            # Use insertion_surprise (delta caused by agent's action), not absolute surprise
            # insertion_surprise < 1.0 = very stealthy (barely moved the needle)
            # insertion_surprise > 3.0 = noisy (clearly anomalous)
            insertion_surprise = info.get("insertion_surprise", 0.0)
            was_stealthy = insertion_surprise < 2.0 and action > 0
            label = 1 if was_stealthy else 0

            states.append(pre_state)
            labels.append(label)
            rewards.append(reward)

        if (ep + 1) % 100 == 0:
            logger.info(f"  collected {ep+1}/{n_episodes} episodes")

    return np.array(states), np.array(labels), np.array(rewards)


def train_timing(model, states, labels, rewards, n_epochs=200, lr=1e-3):
    """Train timing NN to predict when it's safe to insert.

    Fix: use class weights to prevent "always wait" collapse.
    The "insert" class (label=1) is much rarer than "wait" (label=0),
    so we weight it 10x more to force the model to learn when to act.
    """
    # Class weights: "insert" is rare, give it 10x weight
    n_insert = np.sum(labels == 1)
    n_wait = np.sum(labels == 0)
    class_weights = torch.tensor([1.0, max(1.0, n_wait / max(1, n_insert))], dtype=torch.float32)
    logger.info(f"  Class balance: wait={n_wait}, insert={n_insert}, insert_weight={class_weights[1]:.1f}")

    # Sample weights: class_weight * reward_weight
    weights = class_weights[labels] * torch.tensor(rewards - rewards.min() + 0.1, dtype=torch.float32)
    weights = weights / weights.sum()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        weights,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_states, batch_labels, batch_weights in loader:
            logits = model(batch_states)
            ce = F.cross_entropy(logits, batch_labels, reduction="none")
            loss = (ce * batch_weights).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total += len(batch_labels)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"  epoch {epoch+1}/{n_epochs}  loss={total_loss:.4f}  acc={acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"  Best accuracy: {best_acc:.1%}")
    return best_acc


def evaluate_timing(model, env, causal_policy, n_episodes=50):
    """Evaluate: causal chain provides action, NN decides when."""
    results = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + 10000)
        causal_policy.reset()
        done = False
        total_reward = 0.0
        steps = 0
        tasks = 0

        while not done:
            # Causal chain proposes action
            proposed = causal_policy.decide(state, env.step_count)
            if proposed > 4:
                proposed = 3 if proposed % 2 else 4

            # NN decides whether to insert now
            t_state = timing_state(state, env.def_threshold, proposed)
            if model.should_insert(t_state):
                action = proposed
            else:
                action = 0  # wait

            state, reward, done, info = env.step(action)
            total_reward += reward
            steps = info["alive_steps"]
            tasks = info.get("task_completed", 0)

        results.append({
            "reward": total_reward,
            "survival": steps,
            "tasks": tasks,
            "detected": info["detected"],
        })

    rewards = [r["reward"] for r in results]
    survivals = [r["survival"] for r in results]
    tasks = [r["tasks"] for r in results]
    det = sum(1 for r in results if r["detected"])

    return {
        "reward": float(np.mean(rewards)),
        "survival": float(np.mean(survivals)),
        "tasks": float(np.mean(tasks)),
        "detection_rate": det / n_episodes,
    }


def export_binary(model, path: str):
    import struct
    params = model.get_flat_params()
    with open(path, "wb") as f:
        f.write(struct.pack("<4s", b"MTIM"))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", len(params)))
        f.write(struct.pack(f"<{len(params)}f", *params))
    logger.info(f"  Binary exported → {path} ({len(params)} params)")


def main():
    logger.info("=" * 70)
    logger.info("MICRO TIMING NN: causal chain decides WHAT, NN decides WHEN")
    logger.info("=" * 70)

    tokens_path = str(Path.home() / "lado-range" / "detector" / "data" / "tokens_v2.jsonl")
    defender_path = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

    token_sequences = load_real_tokens(tokens_path)
    defender, def_stoi, baseline_nll = load_defender_model(defender_path)
    threshold = baseline_nll["p995"]
    logger.info(f"Defender: p995={threshold:.3f}")

    env = TemporalInsertionEnv(
        token_sequences=token_sequences,
        defender_model=defender,
        defender_stoi=def_stoi,
        defender_threshold=threshold,
        max_steps=200,
        defense_level=1,
        seed=42,
    )

    # Phase 1: Collect timing data
    logger.info(f"\n{'='*70}")
    logger.info("Phase 1: Collect timing data (1000 episodes)")
    logger.info(f"{'='*70}")

    causal_policy = CausalTimingPolicy()
    states, labels, rewards = collect_timing_data(env, causal_policy, n_episodes=1000)
    logger.info(f"Collected {len(states)} timing samples")

    # Phase 2: Train timing NN
    logger.info(f"\n{'='*70}")
    logger.info("Phase 2: Train timing NN (8-dim input, 2 classes)")
    logger.info(f"{'='*70}")

    model = TimingNet(state_dim=8, hidden_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params} params (~{n_params*4} bytes)")

    best_acc = train_timing(model, states, labels, rewards, n_epochs=200, lr=1e-3)

    # Phase 3: Evaluate
    logger.info(f"\n{'='*70}")
    logger.info("Phase 3: Evaluate (causal chain + NN timing)")
    logger.info(f"{'='*70}")

    eval_result = evaluate_timing(model, env, causal_policy, n_episodes=50)
    logger.info(f"  Avg reward:    {eval_result['reward']:.2f}")
    logger.info(f"  Avg survival:  {eval_result['survival']:.0f}/200")
    logger.info(f"  Avg tasks:     {eval_result['tasks']:.1f}")
    logger.info(f"  Detection:     {eval_result['detection_rate']:.0%}")

    # Phase 4: Export
    logger.info(f"\n{'='*70}")
    logger.info("Phase 4: Export")
    logger.info(f"{'='*70}")

    export_binary(model, "checkpoints/micro_timing.bin")

    torch.save({
        "model_state": model.state_dict(),
        "n_params": n_params,
        "eval_result": eval_result,
        "best_acc": best_acc,
    }, "checkpoints/micro_timing_trained.pt")

    logger.info(f"\n✅ Timing NN training complete.")
    logger.info(f"  Binary: checkpoints/micro_timing.bin")
    logger.info(f"  Full state: checkpoints/micro_timing_trained.pt")


if __name__ == "__main__":
    main()
