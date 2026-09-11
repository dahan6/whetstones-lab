"""SSH/SCP 助手：编排器对 VM 的唯一操作通道。"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


def ssh_exec(ip: str, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """在 VM 上执行命令，返回 (returncode, stdout, stderr)。"""
    r = subprocess.run(
        [
            "ssh", "-i", str(config.SSH_KEY),
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={config.SSH_CONNECT_TIMEOUT}",
            "-o", "LogLevel=ERROR",
            "-o", "BatchMode=yes",
            f"{config.VM_USER}@{ip}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def scp_push(ip: str, local: Path, remote: str, timeout: int = 120) -> None:
    """推送文件到 VM。"""
    subprocess.run(
        [
            "scp", "-i", str(config.SSH_KEY),
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={config.SSH_CONNECT_TIMEOUT}",
            "-o", "LogLevel=ERROR",
            str(local), f"{config.VM_USER}@{ip}:{remote}",
        ],
        check=True, capture_output=True, timeout=timeout,
    )


def ssh_alive(ip: str) -> bool:
    """探测 VM ssh 是否就绪。"""
    try:
        rc, out, _ = ssh_exec(ip, "echo ALIVE", timeout=10)
        return rc == 0 and "ALIVE" in out
    except (subprocess.TimeoutExpired, OSError):
        return False
