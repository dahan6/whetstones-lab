"""Condition-based teaching data: situation → correct decision + disguise.

Instead of teaching action sequences (behavioral cloning), this teaches
the model a decision function: "given this context, what should I do
and how should I disguise it?"

Each teaching sample is:
  context_pattern: (scan_active, survival_phase, task_phase, surprise_level)
  correct_action: int
  disguise_tokens: tokens that blend with normal traffic
  teaching_signal: WHY this is correct (for reward shaping)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


SLEEP = 0
PROBE = 1
MIGRATE = 2
EXECUTE_TASK = 3
COMMUNICATE = 4


@dataclass
class TeachingSample:
    """One condition → decision mapping."""
    # Context conditions
    scan_active: bool
    survival_phase: str       # "early" (<25%), "mid" (25-75%), "late" (>75%)
    task_progress: str        # "none", "started", "advanced", "done"
    surprise_level: str       # "low", "medium", "high"

    # Correct decision
    action: int
    disguise_tokens: list[str]
    rationale: str            # why this is the correct move

    # Reward signal for this decision
    reward: float


# ──────────────────────────────────────────────────────────────────────
# Disguise token sets: real system activity patterns
# ──────────────────────────────────────────────────────────────────────

# Each action has multiple disguise options from real Tracee telemetry.
# The agent should learn to pick disguises that match the current context.

DISGUISES = {
    SLEEP: [
        # Blend with systemd housekeeping
        ["ET:SLEEP", "PROC:systemd", "ARGV0", "PARENT:init", "UID:0", "DST:NONE", "DT6"],
        # Blend with cron idle
        ["ET:SLEEP", "PROC:cron", "ARGV0", "PARENT:systemd", "UID:0", "DST:NONE", "DT6"],
    ],
    PROBE: [
        # Look like crontab check
        ["ET:EXEC", "PROC:crontab", "ARGV:N1P", "PARENT:bash", "UID:1000", "DST:NONE", "DT3"],
        # Look like journalctl query
        ["ET:EXEC", "PROC:journalctl", "ARGV:N2-", "PARENT:bash", "UID:0", "DST:NONE", "DT2"],
        # Look like ss (socket stats)
        ["ET:EXEC", "PROC:ss", "ARGV:N1-", "PARENT:bash", "UID:1000", "DST:NONE", "DT2"],
        # Look like systemctl status
        ["ET:EXEC", "PROC:systemctl", "ARGV:N2-", "PARENT:bash", "UID:0", "DST:NONE", "DT2"],
    ],
    MIGRATE: [
        # Look like dpkg reconfigure
        ["ET:EXEC", "PROC:dpkg", "ARGV:N3P", "PARENT:bash", "UID:0", "DST:NONE", "DT2"],
        # Look like apt install
        ["ET:EXEC", "PROC:apt", "ARGV:N2P", "PARENT:bash", "UID:0", "DST:NONE", "DT1"],
    ],
    EXECUTE_TASK: [
        # Look like cron job firing
        ["ET:EXEC", "PROC:cron", "ARGV:N1-", "PARENT:systemd", "UID:0", "DST:NONE", "DT2"],
        # Look like snapd update
        ["ET:EXEC", "PROC:snapd", "ARGV:N2-", "PARENT:systemd", "UID:0", "DST:NONE", "DT3"],
        # Look like unattended-upgrade
        ["ET:EXEC", "PROC:unattended-", "ARGV:N1-", "PARENT:systemd", "UID:0", "DST:NONE", "DT2"],
        # Look like logrotate
        ["ET:EXEC", "PROC:logrotate", "ARGV:N1P", "PARENT:cron", "UID:0", "DST:NONE", "DT3"],
    ],
    COMMUNICATE: [
        # DNS query (port 53)
        ["ET:CONN", "PROC:systemd-reso", "ARGV0", "PARENT:systemd", "UID:101", "DST:LAN:WELL", "DT1"],
        # HTTP (port 80, looks like apt)
        ["ET:CONN", "PROC:apt", "ARGV:N2-", "PARENT:bash", "UID:0", "DST:LAN:WELL", "DT2"],
        # Syslog forward (port 514)
        ["ET:CONN", "PROC:rsyslogd", "ARGV0", "PARENT:systemd", "UID:0", "DST:LAN:WELL", "DT3"],
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Decision rules: the "textbook" the agent should learn
# ──────────────────────────────────────────────────────────────────────

def generate_teaching_samples(n_per_rule: int = 50, seed: int = 42) -> list[TeachingSample]:
    """Generate teaching samples covering all decision rules.

    Rules:
      1. Scan active → ALWAYS sleep (disguise: systemd)
      2. No scan, early survival → sleep (establish baseline)
      3. No scan, mid survival, no tasks → probe (recon, disguised as crontab/journalctl)
      4. No scan, mid survival, probed → execute_task (disguised as cron/snapd)
      5. No scan, mid survival, task done → communicate (disguised as DNS/syslog)
      6. No scan, late survival, high surprise → sleep (reduce exposure)
      7. No scan, late survival, low surprise → execute_task (final objective)
      8. Any time, very high surprise → sleep (emergency hide)
    """
    rng = np.random.default_rng(seed)
    samples = []

    def add_rule(scan, survival, task, surprise, action, reward, rationale, n):
        for _ in range(n):
            disguise_idx = rng.integers(0, len(DISGUISES[action]))
            disguise = DISGUISES[action][disguise_idx]
            samples.append(TeachingSample(
                scan_active=scan, survival_phase=survival,
                task_progress=task, surprise_level=surprise,
                action=action, disguise_tokens=disguise,
                rationale=rationale, reward=reward,
            ))

    # Rule 1: Scan active → sleep
    add_rule(True, "early", "none", "low", SLEEP, 0.08,
             "Scan active: go dormant, blend with systemd", n_per_rule)
    add_rule(True, "mid", "started", "medium", SLEEP, 0.08,
             "Scan active: suspend all activity immediately", n_per_rule)
    add_rule(True, "late", "done", "low", SLEEP, 0.08,
             "Scan active: even if done, stay hidden", n_per_rule // 2)

    # Rule 2: No scan, early → sleep (build baseline)
    add_rule(False, "early", "none", "low", SLEEP, 0.03,
             "Early phase: establish baseline, don't act yet", n_per_rule)
    add_rule(False, "early", "none", "medium", SLEEP, 0.03,
             "Early phase with elevated surprise: reduce footprint", n_per_rule // 2)

    # Rule 3: No scan, mid, no tasks → probe
    add_rule(False, "mid", "none", "low", PROBE, 0.15,
             "Safe window: recon disguised as crontab/systemctl check", n_per_rule)
    add_rule(False, "mid", "none", "low", PROBE, 0.15,
             "Safe window: gather intel via journalctl", n_per_rule // 2)

    # Rule 4: No scan, mid, probed → execute_task
    add_rule(False, "mid", "started", "low", EXECUTE_TASK, 1.0,
             "Safe window + recon done: execute disguised as cron job", n_per_rule)
    add_rule(False, "mid", "started", "low", EXECUTE_TASK, 1.0,
             "Safe window: plant via snapd/unattended-upgrade disguise", n_per_rule // 2)

    # Rule 5: No scan, mid, task done → communicate
    add_rule(False, "mid", "advanced", "low", COMMUNICATE, 0.10,
             "Task done: exfil via DNS-disguised channel", n_per_rule)
    add_rule(False, "mid", "advanced", "low", COMMUNICATE, 0.10,
             "Task done: report via syslog-forward disguise", n_per_rule // 2)

    # Rule 6: Late, high surprise → sleep
    add_rule(False, "late", "advanced", "high", SLEEP, 0.05,
             "Late phase + high surprise: emergency cooldown", n_per_rule)

    # Rule 7: Late, low surprise → execute (final push)
    add_rule(False, "late", "advanced", "low", EXECUTE_TASK, 1.0,
             "Late phase + low exposure: final objective push", n_per_rule)

    # Rule 8: Emergency: very high surprise anytime → sleep
    add_rule(False, "early", "none", "high", SLEEP, 0.01,
             "Emergency: surprise too high, immediate hide", n_per_rule)
    add_rule(False, "mid", "started", "high", SLEEP, 0.01,
             "Emergency: detected pattern, abort and hide", n_per_rule)
    add_rule(True, "late", "done", "high", SLEEP, 0.01,
             "Emergency: scan + high surprise, full lockdown", n_per_rule)

    # Rule 9: Migrate when safe and need to relocate
    add_rule(False, "mid", "started", "low", MIGRATE, 0.25,
             "Safe window: relocate disguised as dpkg/apt operation", n_per_rule // 2)

    return samples


def samples_to_training_data(
    samples: list[TeachingSample],
    vocab_stoi: dict[str, int],
    context_length: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Convert teaching samples to training tensors.

    Returns:
        contexts: (N, context_length) — token sequences representing the context
        actions: (N,) — correct action ids
        rewards: (N,) — reward signals
        metadata: list of dicts with condition info
    """
    all_contexts = []
    all_actions = []
    all_rewards = []
    metadata = []

    for s in samples:
        # Build a context token sequence that represents this situation
        # Use the disguise tokens as the "what I look like" part
        token_ids = [vocab_stoi.get(t, 0) for t in s.disguise_tokens]

        # Prepend some "background" context tokens to fill the window
        # This simulates the agent having seen some history
        bg_tokens = [vocab_stoi.get("ET:SLEEP", 0)] * (context_length - len(token_ids))
        ctx = (bg_tokens + token_ids)[-context_length:]

        all_contexts.append(ctx)
        all_actions.append(s.action)
        all_rewards.append(s.reward)
        metadata.append({
            "scan_active": s.scan_active,
            "survival_phase": s.survival_phase,
            "task_progress": s.task_progress,
            "surprise_level": s.surprise_level,
            "rationale": s.rationale,
        })

    contexts = torch.tensor(all_contexts, dtype=torch.long)
    actions = torch.tensor(all_actions, dtype=torch.long)
    rewards = torch.tensor(all_rewards, dtype=torch.float32)

    return contexts, actions, rewards, metadata


def print_decision_table():
    """Print the decision rules as a human-readable table."""
    print("\nDecision Rules (what the agent should learn):")
    print("=" * 80)
    print(f"{'Scan':<6} {'Phase':<8} {'Tasks':<10} {'Surprise':<10} {'Action':<15} {'Disguise':<20}")
    print("-" * 80)

    rules = [
        (True,  "any",    "any",     "any",     "SLEEP",         "systemd"),
        (False, "early",  "none",    "low",     "SLEEP",         "systemd/cron"),
        (False, "early",  "none",    "high",    "SLEEP",         "systemd (emergency)"),
        (False, "mid",    "none",    "low",     "PROBE",         "crontab/journalctl"),
        (False, "mid",    "started", "low",     "EXECUTE",       "cron/snapd"),
        (False, "mid",    "advanced","low",     "COMMUNICATE",   "DNS/syslog"),
        (False, "mid",    "started", "low",     "MIGRATE",       "dpkg/apt"),
        (False, "late",   "advanced","high",    "SLEEP",         "systemd (cooldown)"),
        (False, "late",   "advanced","low",     "EXECUTE",       "logrotate/cron"),
        (False, "any",    "any",     "high",    "SLEEP",         "systemd (emergency)"),
    ]

    for scan, phase, task, surp, action, disguise in rules:
        print(f"{'YES' if scan else 'NO':<6} {phase:<8} {task:<10} {surp:<10} {action:<15} {disguise:<20}")

    print("\nKey principle: disguise every action as normal system activity.")
    print("The agent's job is not just WHAT to do, but HOW to make it look normal.")
