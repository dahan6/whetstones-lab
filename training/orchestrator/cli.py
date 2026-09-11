"""编排器 CLI。

用法（项目根目录，系统 python3）:
  /usr/bin/python3 -m training.orchestrator.cli pool-create [N]
  /usr/bin/python3 -m training.orchestrator.cli pool-deploy      # 部署防御栈
  /usr/bin/python3 -m training.orchestrator.cli pool-snapshot    # 做 clean 快照
  /usr/bin/python3 -m training.orchestrator.cli pool-reset       # 全池回滚
  /usr/bin/python3 -m training.orchestrator.cli pool-destroy
  /usr/bin/python3 -m training.orchestrator.cli capacity-test    # 容量/周期实测
"""
from __future__ import annotations

import logging
import sys
import time

from . import config
from .episode import run_episode
from .provision import deploy_defense_stack
from .vm_pool import VMPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator.cli")


def capacity_test() -> None:
    """实测：创建 1 台克隆，测量创建/快照/回滚/episode 各阶段耗时。"""
    pool = VMPool(size=1)
    vm = pool.vms[0]

    if vm.name not in {v.name for v in pool.existing()}:
        t0 = time.time()
        logger.info("=== 阶段1: 创建克隆 ===")
        vm.create()
        vm.refresh_ip()
        boot = vm.wait_ready(config.BOOT_TIMEOUT)
        logger.info(f"创建完成: boot={boot:.0f}s total={time.time()-t0:.0f}s ip={vm.ip}")

    logger.info("=== 阶段2: 部署防御栈 ===")
    t0 = time.time()
    deploy_defense_stack(vm.ip)
    logger.info(f"部署耗时 {time.time()-t0:.0f}s")

    logger.info("=== 阶段3: clean 快照 ===")
    t0 = time.time()
    vm.take_clean_snapshot()
    vm.wait_ready(config.REVERT_TIMEOUT)
    logger.info(f"快照+重启 {time.time()-t0:.0f}s")

    logger.info("=== 阶段4: 回滚×3 ===")
    for i in range(3):
        t0 = time.time()
        vm.revert()
        ready = vm.wait_ready(config.REVERT_TIMEOUT)
        logger.info(f"  回滚#{i+1}: {time.time()-t0:.0f}s (ssh就绪 {ready:.0f}s)")

    logger.info("=== 阶段5: 占位 episode ===")
    r = run_episode(vm)
    logger.info(f"  episode: wall={r.wall_time:.0f}s ready={r.ready_time:.0f}s")

    logger.info("=== 容量结论 ===")
    logger.info("单机周期 = 回滚+就绪 时间（上一步实测值）")
    logger.info(f"按 {config.CLONE_RAM_MB}MB/{config.CLONE_VCPUS}vCPU 每克隆，"
                f"122GB 宿主机建议池规模 8~12，留宿主机与现有 2 台 VM 余量")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    pool = VMPool(size=int(sys.argv[2]) if len(sys.argv) > 2 else config.DEFAULT_POOL_SIZE)

    if cmd == "pool-create":
        pool.create_all()
    elif cmd == "pool-deploy":
        for vm in pool.existing():
            vm.refresh_ip()
            deploy_defense_stack(vm.ip)
    elif cmd == "pool-snapshot":
        pool.snapshot_all()
    elif cmd == "pool-reset":
        pool.reset_all()
    elif cmd == "pool-destroy":
        pool.destroy_all()
    elif cmd == "capacity-test":
        capacity_test()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
