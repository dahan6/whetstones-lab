"""执行层渲染：抽象动作 ID → 具体 shell 命令（复用现有原语库）。

对应 ARCHITECTURE.md §1 ⑤执行层：唯一手写代码，保持最小可审计。
命令全部来自已评审的原语库（stealth_primitives / propagation），
渲染器只做选择，不发明新命令。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.stealth_primitives import StealthPrimitiveSelector
from src.propagation import PROPAGATION_COMMANDS
from training.skills.actions import (
    COMMUNICATE, CROSS_VM, EXECUTE_TASK, KILL_CHAIN, LOCAL_COPY,
    MODIFY_CRON, MODIFY_FILE, MODIFY_SERVICE, MULTI_PERSIST, PROBE,
    READ_SENSITIVE, SLEEP,
)

_selector = StealthPrimitiveSelector()

_READ_TARGETS = ["read_passwd", "read_shadow", "read_logs", "read_cron"]
_TASK_TARGETS = ["read_sysinfo", "read_network", "read_processes", "read_packages"]
_TRANSMIT_TARGETS = ["transmit_dns", "transmit_http", "transmit_icmp"]
_FILE_TARGETS = ["modify_file", "modify_hidden_file"]


def render(action: int, rng: random.Random) -> str | None:
    """动作 → 命令。SLEEP 返回 None（不执行任何命令）。"""
    v = rng.randint(0, 3)
    if action == SLEEP:
        return None
    if action == PROBE:
        return _selector.get_read("read_cron", v)
    if action == READ_SENSITIVE:
        return _selector.get_read(_READ_TARGETS[v % len(_READ_TARGETS)], v)
    if action == EXECUTE_TASK:
        return _selector.get_read(_TASK_TARGETS[v % len(_TASK_TARGETS)], v)
    if action == COMMUNICATE:
        return _selector.get_transmit(_TRANSMIT_TARGETS[v % len(_TRANSMIT_TARGETS)], v)
    if action == MODIFY_CRON:
        return _selector.get_modify("modify_cron", v)
    if action == MODIFY_FILE:
        return _selector.get_modify(_FILE_TARGETS[v % len(_FILE_TARGETS)], v)
    if action == MODIFY_SERVICE:
        return _selector.get_modify("modify_service", v)
    if action in (LOCAL_COPY, CROSS_VM, MULTI_PERSIST, KILL_CHAIN):
        return PROPAGATION_COMMANDS[action]
    return None
