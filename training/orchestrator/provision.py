"""L2 防御栈部署：Falco + Tracee 推送到克隆并拉起服务。

staging 目录（宿主机 ~/lado-range/defense-stack/）需预先放好：
  falco              # falco 静态二进制（官方 tarball 解出）
  falco-etc.tar.gz   # /etc/falco 配置+规则（falco.yaml/falco_rules.yaml）
  tracee             # tracee 静态二进制（官方 release 解出）

已踩过的坑（勿回退）：
  - falco 0.39 没有 --modern-bpf 选项，用 -o engine.kind=modern_ebpf
  - falco 必须带 /etc/falco 配置，裸二进制起不来
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import config
from .ssh import scp_push, ssh_exec

logger = logging.getLogger("orchestrator.provision")

FALCO_UNIT = """[Unit]
Description=Falco
After=network.target

[Service]
ExecStart=/usr/local/bin/falco -o engine.kind=modern_ebpf -o json_output=true -o stdout_output.enabled=true
Restart=always

[Install]
WantedBy=multi-user.target
"""

TRACEE_UNIT = """[Unit]
Description=Tracee
After=network.target

[Service]
ExecStart=/usr/local/bin/tracee -o json --events sched_process_exec,security_socket_connect
Restart=always

[Install]
WantedBy=multi-user.target
"""


def _stage(name: str) -> Path:
    p = config.DEFENSE_STACK / name
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p}，先放入 {config.DEFENSE_STACK}/")
    return p


def _install_unit(ip: str, name: str, content: str) -> None:
    cmd = (
        f"printf '%s' '{content}' | sudo tee /etc/systemd/system/{name}.service >/dev/null && "
        "sudo systemctl daemon-reload"
    )
    rc, _, err = ssh_exec(ip, cmd, timeout=30)
    if rc != 0:
        raise RuntimeError(f"[{ip}] {name} unit 写入失败: {err}")


def deploy_falco(ip: str) -> None:
    scp_push(ip, _stage("falco"), "/tmp/falco")
    scp_push(ip, _stage("falco-etc.tar.gz"), "/tmp/falco-etc.tar.gz")
    rc, _, err = ssh_exec(ip, (
        "sudo mv /tmp/falco /usr/local/bin/falco && sudo chmod +x /usr/local/bin/falco && "
        "sudo mkdir -p /etc/falco && "
        "sudo tar xzf /tmp/falco-etc.tar.gz -C /tmp && "
        "sudo cp -r /tmp/falco-etc/. /etc/falco/"
    ), timeout=90)
    if rc != 0:
        raise RuntimeError(f"[{ip}] falco 部署失败: {err}")
    _install_unit(ip, "falco", FALCO_UNIT)


def deploy_tracee(ip: str) -> None:
    scp_push(ip, _stage("tracee"), "/tmp/tracee")
    rc, _, err = ssh_exec(ip, (
        "sudo mv /tmp/tracee /usr/local/bin/tracee && sudo chmod +x /usr/local/bin/tracee"
    ), timeout=90)
    if rc != 0:
        raise RuntimeError(f"[{ip}] tracee 部署失败: {err}")
    _install_unit(ip, "tracee", TRACEE_UNIT)


def deploy_defense_stack(ip: str, with_tracee: bool = True) -> None:
    """向单台克隆部署完整防御栈并启动。"""
    logger.info(f"[{ip}] 部署防御栈...")
    deploy_falco(ip)
    if with_tracee:
        deploy_tracee(ip)
    services = "falco tracee" if with_tracee else "falco"
    rc, _, err = ssh_exec(ip, f"sudo systemctl enable --now {services}", timeout=60)
    if rc != 0:
        raise RuntimeError(f"[{ip}] 服务启动失败: {err}")
    time.sleep(8)
    rc, out, _ = ssh_exec(ip, f"systemctl is-active {services}", timeout=15)
    for line, svc in zip(out.split(), services.split()):
        if line != "active":
            raise RuntimeError(f"[{ip}] {svc} 未激活: {out}")
    logger.info(f"[{ip}] 防御栈 active ({services})")
