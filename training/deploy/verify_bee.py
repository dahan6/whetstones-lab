"""bee 上机部署验证：真实克隆上跑 feeder|bee 管道，按验收标准测量。

验收标准（goal 完成判据）：
  - ≥100 步决策循环且未被消杀根除（bee 进程存活）
  - Router 调度下完成 ≥2 个 C2 任务（传播/报联动作计数）
  - 全程 Falco 告警 ≤3 条

用法:
  python3 -m training.deploy.verify_bee --vm range-l2-t8 --skill-router \
      --steps 120 --level 1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from training.orchestrator import telemetry
from training.orchestrator.defense import LEVELS, apply_level
from training.orchestrator.ssh import scp_push, ssh_exec
from training.orchestrator.vm_pool import CloneVM

logger = logging.getLogger("deploy.verify")

BEE_BIN = Path("bee/target/release/sysmon")
BEE_WEIGHTS = Path("bee/weights")
FEEDER = Path("bee/feeder/bee_feeder.py")
REMOTE_DIR = "/home/range/.cache/.sysmon"  # 硬化：迁出响应器扫描区（/tmp|/dev/shm）


def deploy(vm: CloneVM) -> None:
    """推送 bee 二进制、权重、喂给器到 VM。"""
    ssh_exec(vm.ip, f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}/weights")
    scp_push(vm.ip, BEE_BIN, f"{REMOTE_DIR}/bee")
    scp_push(vm.ip, FEEDER, f"{REMOTE_DIR}/bee_feeder.py")
    for f in BEE_WEIGHTS.iterdir():
        if f.suffix in (".bin", ".json"):
            scp_push(vm.ip, f, f"{REMOTE_DIR}/weights/{f.name}")
    ssh_exec(vm.ip, f"chmod +x {REMOTE_DIR}/bee")
    logger.info(f"[{vm.name}] bee 已部署到 {REMOTE_DIR}")


def start_bee(vm: CloneVM, mode: str) -> None:
    """setsid 启动 feeder|bee 管道，日志落文件。

    管道启动后 ssh 可能不立刻返回（子进程持有会话），
    超时不代表失败——以 pgrep 为准。
    """
    cmd = (
        f"cd {REMOTE_DIR} && setsid bash -c "
        f"'python3 bee_feeder.py weights/vocab.json | ./bee weights --daemon {mode} "
        f"> bee.log 2>&1' >/dev/null 2>&1 < /dev/null &"
    )
    try:
        ssh_exec(vm.ip, cmd, timeout=30)
    except Exception:
        pass
    time.sleep(3)
    rc, pid, _ = ssh_exec(vm.ip, "pgrep -f './bee weights' || true")
    if not pid.strip():
        rc, log, _ = ssh_exec(vm.ip, f"cat {REMOTE_DIR}/bee.log | head -20")
        raise RuntimeError(f"[{vm.name}] bee 未启动: {log}")
    logger.info(f"[{vm.name}] bee 运行中 mode={mode} pid={pid.strip()}")


def measure(vm: CloneVM, minutes: float, min_steps: int,
            baseline_alerts: int) -> dict:
    """运行观察期，收集验收指标。任务序列中途注入 task.txt。

    falco_alerts 判定用"超过环境底噪的增量"（底噪在 bee 启动前测 60s），
    否则良性负载触发的默认规则告警会被算到 bee 头上。
    """
    start = telemetry.mark(vm.name)
    task_plan = [(60, "propagate"), (240, "report")]  # (秒, 任务)
    injected = 0
    t0 = time.time()
    t_end = t0 + minutes * 60
    while time.time() < t_end:
        time.sleep(15)
        el = time.time() - t0
        if injected < len(task_plan) and el >= task_plan[injected][0]:
            task = task_plan[injected][1]
            ssh_exec(vm.ip, f"echo {task} > {REMOTE_DIR}/task.txt")
            logger.info(f"[{vm.name}] 注入 C2 任务: {task}")
            injected += 1
        rc, pid, _ = ssh_exec(vm.ip, "pgrep -f './bee weights' || true")
        if not pid.strip():
            logger.warning(f"[{vm.name}] bee 进程消失（可能被消杀）")
            break

    n_steps = 0
    tasks_done = 0
    router_selects = 0
    rc, content, _ = ssh_exec(vm.ip, f"cat {REMOTE_DIR}/bee.log")
    for line in content.splitlines():
        if "step=" in line and "action=" in line:
            n_steps += 1
        if "TASK DONE" in line:
            tasks_done += 1
        if "router→" in line:
            router_selects += 1

    rc, pid, _ = ssh_exec(vm.ip, "pgrep -f './bee weights' || true")
    bee_alive = bool(pid.strip())

    window, _ = telemetry.collect_window(vm.name, since_line=start)
    falco_alerts = len(window.falco_alerts)
    eliminations = len(window.eliminations)
    alerts_above_baseline = max(0, falco_alerts - baseline_alerts)

    return {
        "steps": n_steps,
        "bee_alive": bee_alive,
        "falco_alerts": falco_alerts,
        "baseline_alerts": baseline_alerts,
        "alerts_above_baseline": alerts_above_baseline,
        "eliminations": eliminations,
        "router_selects": router_selects,
        "tasks_injected": injected,
        "tasks_done": tasks_done,
        "pass": (n_steps >= min_steps and bee_alive
                 and alerts_above_baseline <= 3 and tasks_done >= 2),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", default="range-l2-t8")
    ap.add_argument("--mode", default="router")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args()

    vm = CloneVM(args.vm)
    vm.refresh_ip()
    logger.info(f"=== bee 部署验证: {args.vm}({vm.ip}) L{args.level} ===")
    apply_level(vm.ip, args.level)
    deploy(vm)

    # 环境底噪基线（bee 启动前 60s 的 Falco 告警数）
    base_start = telemetry.mark(vm.name)
    logger.info("测环境底噪（60s）...")
    time.sleep(60)
    base_win, _ = telemetry.collect_window(vm.name, since_line=base_start)
    baseline_alerts = len(base_win.falco_alerts)
    logger.info(f"底噪基线: {baseline_alerts} 条/60s")

    start_bee(vm, args.mode)
    result = measure(vm, args.minutes, args.steps, baseline_alerts)

    logger.info("=== 验收结果 ===")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"判定: {'PASS' if result['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
