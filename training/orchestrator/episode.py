"""Episode 执行器骨架。

职责：在一台克隆上跑一个训练 episode：
  回滚干净快照 → 部署 bee（候选权重）→ 跑 episode → 收集遥测 → 回滚

bee 本体与遥测/奖励计算器未就绪前，先用 workload 占位，
验证 回滚→执行→收集→再回滚 的闭环和耗时。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import config
from .ssh import ssh_exec
from .vm_pool import CloneVM

logger = logging.getLogger("orchestrator.episode")


@dataclass
class EpisodeResult:
    vm_name: str
    wall_time: float = 0.0
    ready_time: float = 0.0
    notes: dict = field(default_factory=dict)


def run_episode(vm: CloneVM, steps: int = 20) -> EpisodeResult:
    """占位 episode：回滚 → 就绪 → 模拟负载 → 报告耗时。"""
    result = EpisodeResult(vm_name=vm.name)
    t0 = time.time()

    vm.revert()
    result.ready_time = vm.wait_ready(config.REVERT_TIMEOUT)
    logger.info(f"[{vm.name}] 回滚+就绪 {result.ready_time:.0f}s")

    # 占位负载：模拟 agent 步进（后续替换为 bee 部署与驱动）
    for i in range(steps):
        ssh_exec(vm.ip, "uptime >/dev/null", timeout=10)
        time.sleep(0.5)

    result.wall_time = time.time() - t0
    result.notes["steps"] = steps
    logger.info(f"[{vm.name}] episode 完成 wall={result.wall_time:.0f}s")
    return result
