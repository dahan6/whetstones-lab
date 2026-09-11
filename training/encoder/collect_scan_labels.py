"""扫描标签采集：在克隆上按随机间隔触发扫描波，制造带标签的数据。

原理：responder scan 会在系统上制造一轮 exec 事件爆发（find/ps/grep），
这些事件流进 rsyslog 汇聚日志。把"扫描进行中"的时间窗标记为
scan_active=1，其余为 0，即得探针任务的真值标签。

用法:
  python3 -m training.encoder.collect_scan_labels --vms range-l2-t1 range-l2-t2 --duration 1800
"""
from __future__ import annotations

import argparse
import logging
import random
import time

from training.orchestrator.responder import run_wave
from training.orchestrator.ssh import ssh_exec

logger = logging.getLogger("collect_scan_labels")


def collect(vm_names: list[str], ips: list[str], duration: int) -> None:
    rng = random.Random(42)
    t_end = time.time() + duration
    for vm, ip in zip(vm_names, ips):
        ssh_exec(ip, "date -Is")  # 连通性检查
    logger.info(f"开始采集: vms={vm_names} duration={duration}s")
    while time.time() < t_end:
        wait = rng.randint(120, 360)
        time.sleep(min(wait, max(0, t_end - time.time())))
        if time.time() >= t_end:
            break
        for vm, ip in zip(vm_names, ips):
            mode = rng.choice(["scan", "scan", "wave"])
            logger.info(f"[{vm}] 触发 {mode}")
            run_wave(ip, mode=mode)
    logger.info("采集结束")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vms", nargs="+", required=True)
    ap.add_argument("--duration", type=int, default=1800)
    args = ap.parse_args()

    from training.orchestrator.vm_pool import CloneVM
    ips = []
    for name in args.vms:
        vm = CloneVM(name)
        ips.append(vm.refresh_ip())
    collect(args.vms, ips, args.duration)


if __name__ == "__main__":
    main()
