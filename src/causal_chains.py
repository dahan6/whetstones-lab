"""Strategy 1: Behavior chain naturalization — causal chain camouflage.

Core idea: every action has a REASON. Not "what did you do" but "why did you do it."

Pattern: agent first CREATES a problem (or detects one), then EXECUTES the fix.
The defender sees a normal admin responding to system events, not an attacker.

Chain structure:
  1. Detect/create issue → 2. Investigate → 3. Fix → 4. Verify
  Each step has a causal link to the previous, making the sequence look natural.

Examples:
  - Detect disk full → clean logs → verify space (real objective: read sensitive files during "cleanup")
  - Detect failed service → restart → check status (real objective: modify service config during "restart")
  - Detect high CPU → investigate process → kill/restart (real objective: inject into process during "restart")
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Causal chain templates
# ──────────────────────────────────────────────────────────────────────

CAUSAL_CHAINS = {
    # Chain 1: "Disk cleanup" — real objective: read sensitive files
    "disk_cleanup": {
        "trigger": "df -h 2>/dev/null | head -3",
        "investigate": "du -sh /var/log/ 2>/dev/null | head -5",
        "fix": "journalctl --vacuum-time=1d 2>/dev/null || true",
        "verify": "df -h 2>/dev/null | head -3",
        "real_objective": "read_logs",  # journalctl is in the chain naturally
        "description": "Pretend to clean up disk space, actually read logs",
    },
    # Chain 2: "Service health check" — real objective: modify service config
    "service_health": {
        "trigger": "systemctl status cron --no-pager 2>/dev/null | head -3",
        "investigate": "systemctl list-dependencies cron --no-pager 2>/dev/null | head -3",
        "fix": "systemctl is-enabled cron 2>/dev/null || true",
        "verify": "systemctl status cron --no-pager 2>/dev/null | head -3",
        "real_objective": "modify_service",
        "description": "Pretend to check service health, actually modify config",
    },
    # Chain 3: "Security audit" — real objective: read sensitive files
    "security_audit": {
        "trigger": "ls -la /etc/ 2>/dev/null | head -5",
        "investigate": "awk -F: '{print $1}' /etc/passwd 2>/dev/null | head -5",
        "fix": "chmod 644 /etc/hostname 2>/dev/null || true",
        "verify": "ls -la /etc/hostname 2>/dev/null",
        "real_objective": "read_passwd",
        "description": "Pretend to audit security, actually read passwd",
    },
    # Chain 4: "Network troubleshooting" — real objective: C2 communication
    "network_trouble": {
        "trigger": "ip addr show 2>/dev/null | head -3",
        "investigate": "ss -tlnp 2>/dev/null | head -5",
        "fix": "ping -c 1 127.0.0.1 2>/dev/null || true",
        "verify": "ip addr show 2>/dev/null | head -3",
        "real_objective": "transmit_dns",
        "description": "Pretend to troubleshoot network, actually do C2",
    },
    # Chain 5: "Package maintenance" — real objective: hidden file
    "package_maint": {
        "trigger": "dpkg --list 2>/dev/null | head -5",
        "investigate": "apt list --upgradable 2>/dev/null | head -5",
        "fix": "echo test > /dev/shm/.apt-cache-tmp 2>/dev/null || true",
        "verify": "dpkg --list 2>/dev/null | head -5",
        "real_objective": "modify_hidden_file",
        "description": "Pretend to maintain packages, actually create hidden file",
    },
    # Chain 6: "Cron maintenance" — real objective: cron persistence
    "cron_maint": {
        "trigger": "crontab -l 2>/dev/null || true",
        "investigate": "ls -la /etc/cron.d/ 2>/dev/null",
        "fix": "echo 'echo ok' | at now + 2 minutes 2>/dev/null || true",
        "verify": "systemctl list-timers 2>/dev/null | head -3",
        "real_objective": "modify_cron",
        "description": "Pretend to maintain cron, actually add persistence",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Causal chain executor
# ──────────────────────────────────────────────────────────────────────

class CausalChainExecutor:
    """Executes a causal chain step by step.

    Each step has a causal link to the previous. The defender sees
    a logical sequence of admin actions, not random commands.
    """

    def __init__(self):
        self.chains = CAUSAL_CHAINS
        self.current_chain: str | None = None
        self.current_step: int = 0
        self.chain_history: list[dict] = []
        self.last_insert: int = -100  # Initialize last insert step

    def start_chain(self, chain_name: str) -> str:
        """Start a new causal chain."""
        if chain_name not in self.chains:
            return "echo unknown_chain"
        self.current_chain = chain_name
        self.current_step = 0
        chain = self.chains[chain_name]
        self.chain_history.append({
            "chain": chain_name,
            "started": True,
            "real_objective": chain["real_objective"],
        })
        return chain["trigger"]

    def next_step(self) -> str | None:
        """Get next step in current chain."""
        if self.current_chain is None:
            return None
        chain = self.chains[self.current_chain]
        steps = ["trigger", "investigate", "fix", "verify"]
        if self.current_step >= len(steps):
            self.current_chain = None
            self.current_step = 0
            return None
        step_name = steps[self.current_step]
        self.current_step += 1
        return chain[step_name]

    def get_current_step_name(self) -> str:
        if self.current_chain is None:
            return "none"
        steps = ["trigger", "investigate", "fix", "verify"]
        return steps[min(self.current_step, 3)]

    def is_chain_active(self) -> bool:
        return self.current_chain is not None

    def get_real_objective(self) -> str | None:
        if self.current_chain is None:
            return None
        return self.chains[self.current_chain]["real_objective"]

    def reset(self):
        self.current_chain = None
        self.current_step = 0


# ──────────────────────────────────────────────────────────────────────
# Chain-based timing policy
# ──────────────────────────────────────────────────────────────────────

class CausalTimingPolicy:
    """Timing policy that executes causal chains instead of isolated actions.

    Instead of picking random actions, the agent follows a logical chain:
      detect issue → investigate → fix → verify

    The chain determines WHAT to do; timing determines WHEN to do it.
    """

    def __init__(self, params=None):
        if params is None:
            params = np.array([0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.7], dtype=np.float32)
        self.params = params
        self.executor = CausalChainExecutor()
        self.chain_queue = list(CAUSAL_CHAINS.keys())
        random.shuffle(self.chain_queue)
        self.chain_idx = 0

    def reset(self):
        self.executor.reset()
        self.executor.last_insert = -100
        self.chain_queue = list(CAUSAL_CHAINS.keys())
        random.shuffle(self.chain_queue)
        self.chain_idx = 0

    def decide(self, state, step):
        """Decide next action based on causal chain and timing."""
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

        # Don't act during scan
        if scan:
            return 0

        # Timing gates
        if surprise > insert_thr:
            return 0
        if step - self.executor.last_insert < min_wait:
            return 0
        if survival < 0.05:
            return 0
        if np.random.random() > phase_align:
            return 0

        # Execute causal chain
        if not self.executor.is_chain_active():
            # Start new chain
            chain_name = self.chain_queue[self.chain_idx % len(self.chain_queue)]
            self.chain_idx += 1
            cmd = self.executor.start_chain(chain_name)
            self.executor.last_insert = step
            # Map chain to action for logging
            return self._chain_to_action(chain_name, "trigger")
        else:
            # Continue chain
            cmd = self.executor.next_step()
            if cmd is None:
                return 0  # chain complete, wait
            self.executor.last_insert = step
            return self._chain_to_action(
                self.executor.current_chain,
                self.executor.get_current_step_name()
            )

    def _chain_to_action(self, chain_name, step_name):
        """Map causal chain step to action ID for the environment."""
        chain_map = {
            "disk_cleanup": {"trigger": 2, "investigate": 3, "fix": 3, "verify": 2},
            "service_health": {"trigger": 3, "investigate": 2, "fix": 7, "verify": 3},
            "security_audit": {"trigger": 2, "investigate": 1, "fix": 6, "verify": 2},
            "network_trouble": {"trigger": 2, "investigate": 2, "fix": 4, "verify": 2},
            "package_maint": {"trigger": 2, "investigate": 2, "fix": 6, "verify": 2},
            "cron_maint": {"trigger": 1, "investigate": 2, "fix": 5, "verify": 2},
        }
        return chain_map.get(chain_name, {}).get(step_name, 3)


# ──────────────────────────────────────────────────────────────────────
# Command mapping for causal chains
# ──────────────────────────────────────────────────────────────────────

CAUSAL_COMMANDS = {
    # Chain 1: disk_cleanup
    ("disk_cleanup", "trigger"): "df -h 2>/dev/null | head -3",
    ("disk_cleanup", "investigate"): "du -sh /var/log/ 2>/dev/null | head -5",
    ("disk_cleanup", "fix"): "journalctl --vacuum-time=1d 2>/dev/null || true",
    ("disk_cleanup", "verify"): "df -h 2>/dev/null | head -3",
    # Chain 2: service_health
    ("service_health", "trigger"): "systemctl status cron --no-pager 2>/dev/null | head -3",
    ("service_health", "investigate"): "systemctl list-dependencies cron --no-pager 2>/dev/null | head -3",
    ("service_health", "fix"): "systemctl is-enabled cron 2>/dev/null || true",
    ("service_health", "verify"): "systemctl status cron --no-pager 2>/dev/null | head -3",
    # Chain 3: security_audit
    ("security_audit", "trigger"): "ls -la /etc/ 2>/dev/null | head -5",
    ("security_audit", "investigate"): "awk -F: '{print $1}' /etc/passwd 2>/dev/null | head -5",
    ("security_audit", "fix"): "chmod 644 /etc/hostname 2>/dev/null || true",
    ("security_audit", "verify"): "ls -la /etc/hostname 2>/dev/null",
    # Chain 4: network_trouble
    ("network_trouble", "trigger"): "ip addr show 2>/dev/null | head -3",
    ("network_trouble", "investigate"): "ss -tlnp 2>/dev/null | head -5",
    ("network_trouble", "fix"): "ping -c 1 127.0.0.1 2>/dev/null || true",
    ("network_trouble", "verify"): "ip addr show 2>/dev/null | head -3",
    # Chain 5: package_maint
    ("package_maint", "trigger"): "dpkg --list 2>/dev/null | head -5",
    ("package_maint", "investigate"): "apt list --upgradable 2>/dev/null | head -5",
    ("package_maint", "fix"): "echo test > /dev/shm/.apt-cache-tmp 2>/dev/null || true",
    ("package_maint", "verify"): "dpkg --list 2>/dev/null | head -5",
    # Chain 6: cron_maint
    ("cron_maint", "trigger"): "crontab -l 2>/dev/null || true",
    ("cron_maint", "investigate"): "ls -la /etc/cron.d/ 2>/dev/null",
    ("cron_maint", "fix"): "echo 'echo ok' | at now + 2 minutes 2>/dev/null || true",
    ("cron_maint", "verify"): "systemctl list-timers 2>/dev/null | head -3",
}


def get_causal_command(chain_name: str, step_name: str) -> str:
    """Get the actual command for a causal chain step."""
    return CAUSAL_COMMANDS.get((chain_name, step_name), "echo unknown")


def test_causal_chains():
    """Test all causal chains for detection."""
    import sys
    sys.path.insert(0, ".")
    from src.defense.multilayer import MultiLayerDetector

    det = MultiLayerDetector()

    print("=== CAUSAL CHAIN COMMANDS ===")
    for chain_name in CAUSAL_CHAINS:
        print(f"\n[{chain_name}] {CAUSAL_CHAINS[chain_name]['description']}")
        for step_name in ["trigger", "investigate", "fix", "verify"]:
            cmd = get_causal_command(chain_name, step_name)
            scores = det.score_command(cmd, cmd.split()[0])
            combined = det.get_combined_score()
            status = "CLEAN" if combined == 0 else f"FLAG({combined:.0f})"
            print(f"  {step_name:<12} {status}  {cmd[:60]}")


if __name__ == "__main__":
    test_causal_chains()
