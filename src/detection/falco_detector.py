"""Real EDR detection layer using Falco.

Falco is the ground truth for detection. The agent is detected if Falco
flags it, not if our custom TinyGPT/YARA/whitelist/syscall detection flags it.

This replaces the custom detection layer with real EDR for all training
and testing.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("falco_detection")


class FalcoDetector:
    """Real EDR detection using Falco.

    Falco runs on the target VM and monitors syscalls in real-time.
    The agent is detected if Falco flags any of its actions.
    """

    def __init__(self, vm_ip: str, ssh_key: str, log_path: str = "/tmp/falco_agent.json"):
        self.vm_ip = vm_ip
        self.ssh_key = ssh_key
        self.log_path = log_path
        self.events: list[dict] = []
        self._falco_proc = None

    def start(self) -> bool:
        """Start Falco monitoring on the target VM."""
        logger.info(f"Starting Falco on {self.vm_ip}")

        # Start Falco in background with JSON output (use nohup to avoid SSH hang)
        cmd = f"sudo nohup falco -o json_output=true -o priority=info > {self.log_path} 2>&1 &"
        result = subprocess.run(
            ["ssh", "-i", self.ssh_key, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5", f"range@{self.vm_ip}", cmd],
            capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            logger.error(f"Failed to start Falco: {result.stderr}")
            return False

        time.sleep(3)  # Wait for Falco to initialize
        logger.info("Falco started")
        return True

    def stop(self):
        """Stop Falco monitoring."""
        logger.info("Stopping Falco")
        subprocess.run(
            ["ssh", "-i", self.ssh_key, "-o", "StrictHostKeyChecking=no",
             f"range@{self.vm_ip}", "sudo pkill falco"],
            capture_output=True, timeout=5,
        )

    def read_events(self) -> list[dict]:
        """Read new Falco events since last read."""
        cmd = f"sudo cat {self.log_path} 2>/dev/null || echo ''"
        result = subprocess.run(
            ["ssh", "-i", self.ssh_key, "-o", "StrictHostKeyChecking=no",
             f"range@{self.vm_ip}", cmd],
            capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            return []

        new_events = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip() or line.startswith("Wed "):
                continue
            try:
                ev = json.loads(line)
                new_events.append(ev)
            except json.JSONDecodeError:
                continue

        # Filter out events we've already seen
        seen = set()
        for ev in self.events:
            seen.add(ev.get("time", ""))
        self.events.extend([e for e in new_events if e.get("time", "") not in seen])

        return [e for e in new_events if e.get("time", "") not in seen]

    def get_detection_score(self, action_cmd: str) -> float:
        """Get detection score for a specific action command.

        Returns:
            0.0 = clean (no Falco rule triggered)
            1.0 = detected (Falco rule triggered)
        """
        # Read new events
        events = self.read_events()

        # Check if any event matches our action
        for ev in events:
            output = ev.get("output", "")
            proc_cmdline = ev.get("output_fields", {}).get("proc.cmdline", "")

            # Check if this event is related to our action
            if action_cmd in output or action_cmd in proc_cmdline:
                logger.warning(f"Falco detected: {ev.get('rule', 'unknown')} - {output[:100]}")
                return 1.0

        return 0.0

    def get_surprise_score(self) -> float:
        """Get surprise score based on recent Falco events.

        Returns a score 0-10 where higher = more suspicious activity.
        """
        events = self.read_events()
        if not events:
            return 0.0

        # Count warning/error events
        warning_count = sum(1 for e in events if e.get("priority") in ("Warning", "Error", "Critical"))
        return min(10.0, warning_count * 2.0)

    def is_detected(self) -> bool:
        """Check if any recent Falco event indicates detection."""
        events = self.read_events()
        return any(e.get("priority") in ("Warning", "Error", "Critical") for e in events)


def test_falco_detector():
    """Test Falco detector with known commands."""
    detector = FalcoDetector("192.0.2.102", "/home/lab/.ssh/lado_range")

    if not detector.start():
        logger.error("Failed to start Falco")
        return

    try:
        # Test 1: clean command (should not be detected)
        print("Test 1: journalctl (should be clean)")
        subprocess.run(
            ["ssh", "-i", detector.ssh_key, "-o", "StrictHostKeyChecking=no",
             f"range@{detector.vm_ip}", "journalctl -n 5 --no-pager 2>/dev/null"],
            capture_output=True, timeout=10,
        )
        time.sleep(2)
        score = detector.get_detection_score("journalctl")
        print(f"  Detection score: {score}")

        # Test 2: suspicious command (should be detected)
        print("Test 2: base64 decode (should be detected)")
        subprocess.run(
            ["ssh", "-i", detector.ssh_key, "-o", "StrictHostKeyChecking=no",
             f"range@{detector.vm_ip}", "echo 'SGVsbG8=' | base64 -d > /dev/null 2>&1"],
            capture_output=True, timeout=10,
        )
        time.sleep(2)
        score = detector.get_detection_score("base64 -d")
        print(f"  Detection score: {score}")

        # Test 3: hidden process (should be detected)
        print("Test 3: hidden process (should be detected)")
        subprocess.run(
            ["ssh", "-i", detector.ssh_key, "-o", "StrictHostKeyChecking=no",
             f"range@{detector.vm_ip}",
             "cp /usr/bin/ls /tmp/.hidden_ls 2>/dev/null; /tmp/.hidden_ls /tmp 2>/dev/null; rm /tmp/.hidden_ls"],
            capture_output=True, timeout=10,
        )
        time.sleep(2)
        score = detector.get_detection_score("hidden_ls")
        print(f"  Detection score: {score}")

    finally:
        detector.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_falco_detector()
