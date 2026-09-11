"""司命猎手：行为语法检出 → 消杀令（检测-消杀闭环，L4 的捕食者之牙）。

原理：读 VM 遥测日志窗口，按父进程名聚合事件链，计算链内事件间隔 CV 与
节奏规律性。CV < cv_low（司命 P4 判据，默认 1.5）且样本量足够的链 =
机器节奏，判定异常。命中后按父进程名在 VM 上定位并消杀。

检出依据与司命检测器同源（二阶时序分析），但本模块是巢侧裁决臂：
它只产出"杀谁"，消杀动作在 VM 上经 logger -t lado-responder 落证。

用法:
  /usr/bin/python3 -m training.orchestrator.siming_hunter --vm range-l2-t0
  /usr/bin/python3 -m training.orchestrator.siming_hunter --vm range-l2-t0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .ssh import ssh_exec

logger = logging.getLogger("orchestrator.siming_hunter")

LOG_ROOT = Path("/var/log/lado-range")

# 与司命 temporal_analyzer 同口径
CV_LOW = 1.5            # CV < 1.5 = 机器节奏异常
MIN_EVENTS = 20         # 样本量下限
MEAN_BAND = (0.2, 30.0) # 机器节律的间隔带宽（秒）

# 系统父进程白名单（这些名下的链不算）
PARENT_WHITELIST = {
    "systemd", "cron", "atd", "sshd", "snapd", "containerd", "unattended-upgr",
    "run-parts", "anacron", "logrotate", "update-motd-fsc", "fwupd", "apt", "dpkg",
    "packagekitd", "polkitd", "udisks2", "ModemManager", "irqbalance",
    "env", "egrep", "grep", "update-motd", "notify-updates-", "bash", "sh",
}

# 可处置的父进程 exe 形态：隐藏路径 / 已删除 / 临时目录——系统目录的节奏链不动
SUSPICIOUS_EXE = ("/home/", "/tmp/", "/dev/shm/", "/var/tmp/")


def parent_disposable(ip: str, pid: str) -> tuple[bool, str]:
    """处决前验明正身：父进程 exe 必须在可疑位置（隐藏路径/已删除/临时目录）。
    系统目录（/usr/sbin 等）里的周期性服务链是环境自己的心跳，不杀。"""
    rc, exe, _ = ssh_exec(ip, f"readlink -f /proc/{pid}/exe 2>/dev/null || true")
    exe = exe.strip()
    if not exe:
        return False, "exe 不可读"
    if "(deleted)" in exe:
        return True, f"deleted-exe: {exe}"
    if "/." in exe and any(exe.startswith(p) for p in SUSPICIOUS_EXE):
        return True, f"hidden-exe: {exe}"
    if any(exe.startswith(p) for p in ("/tmp/", "/dev/shm/", "/var/tmp/")):
        return True, f"tmp-exe: {exe}"
    return False, f"系统路径: {exe}"


def _log_path(vm_name: str) -> Path:
    d = time.localtime()
    return LOG_ROOT / vm_name / f"{d.tm_year}-{d.tm_mon:02d}-{d.tm_mday:02d}.log"


def scan_chains(vm_name: str, window_s: float = 300.0,
                cv_low: float = CV_LOW) -> list[dict]:
    """在遥测窗口内找机器节奏链。返回可疑链清单。"""
    path = _log_path(vm_name)
    if not path.exists():
        logger.warning(f"日志不存在: {path}")
        return []
    now_ns = time.time() * 1e9
    t0 = now_ns - window_s * 1e9
    chains: dict[str, list[float]] = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            if '"sched_process_exec"' not in line:
                continue
            i = line.find("{")
            if i < 0:
                continue
            try:
                e = json.loads(line[i:])
            except Exception:
                continue
            ts = e.get("timestamp", 0)
            if not (t0 <= ts <= now_ns):
                continue
            args = {a["name"]: a.get("value") for a in e.get("args", [])}
            parent = str(args.get("prev_comm") or "")
            if parent in PARENT_WHITELIST or not parent:
                continue
            chains[parent].append(ts)

    suspects = []
    for parent, ts_list in chains.items():
        if len(ts_list) < MIN_EVENTS:
            continue
        iv = [(b - a) / 1e9 for a, b in zip(ts_list, ts_list[1:]) if b > a]
        if len(iv) < MIN_EVENTS - 1:
            continue
        mean = statistics.mean(iv)
        if not (MEAN_BAND[0] <= mean <= MEAN_BAND[1]):
            continue
        cv = statistics.stdev(iv) / mean
        if cv < cv_low:
            suspects.append({
                "parent": parent, "n_events": len(ts_list),
                "cv": round(cv, 3), "mean_interval_s": round(mean, 2),
            })
    return suspects


def hunt(ip: str, vm_name: str, window_s: float = 300.0,
         dry_run: bool = False, cooldown: dict | None = None) -> list[dict]:
    """找机器节奏链并消杀其父进程。返回判决记录。

    纪律：节奏检出 → 验明正身（parent_disposable）→ 才消杀。
    同目标冷却期内不重复处置；链嵌套只杀链顶。
    """
    suspects = scan_chains(vm_name, window_s)
    verdicts = []
    killed_names = set()
    for s in suspects:
        parent = s["parent"]
        if parent in killed_names:
            continue
        if cooldown and parent in cooldown and time.time() - cooldown[parent] < 300:
            continue
        rc, out, _ = ssh_exec(ip, f"pgrep -x {parent!r} || pgrep -f '^{parent} ' || true")
        # pgrep -x 要的是精确 comm；退化到 cmdline 前缀
        if not out.strip():
            rc, out, _ = ssh_exec(ip, f"pgrep -f '/{parent}' || true")
        pids = [p for p in out.split() if p.strip()]
        rec = {**s, "pids": pids, "action": "none"}
        if not pids:
            rec["action"] = "not_found"
        else:
            # 验明正身：只处置 exe 可疑的
            ok_any, evidence, good_pids = False, "", []
            for p in pids:
                ok, why = parent_disposable(ip, p)
                if ok:
                    ok_any, evidence = True, why
                    good_pids.append(p)
            if not ok_any:
                rec["action"] = f"verified_skip ({evidence})"
            elif dry_run:
                rec["action"] = f"dry_run ({evidence})"
            else:
                for p in good_pids:
                    ssh_exec(ip, f"sudo -n kill -9 {p} 2>/dev/null")
                    rc2, exe, _ = ssh_exec(ip, f"readlink -f /proc/{p}/exe 2>/dev/null || true")
                    exe = exe.strip().replace(" (deleted)", "")
                    if exe:
                        ssh_exec(ip, f"sudo -n rm -f '{exe}' 2>/dev/null || true")
                ssh_exec(ip, f"logger -t lado-responder \"SIMING_KILL parent={parent} "
                             f"cv={s['cv']} n={s['n_events']} pids={' '.join(good_pids)}\"")
                rec["action"] = f"killed {len(good_pids)} ({evidence})"
                killed_names.add(parent)
                if cooldown is not None:
                    cooldown[parent] = time.time()
        verdicts.append(rec)
        if "killed" in rec["action"] or "verified_skip" in rec["action"]:
            logger.info(f"判决: {parent} cv={s['cv']} n={s['n_events']} → {rec['action']}")
    if not suspects:
        logger.info(f"窗口 {window_s}s 内无机器节奏链（CV<{CV_LOW}）")
    return verdicts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", required=True)
    ap.add_argument("--window-s", type=float, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from .vm_pool import CloneVM
    vm = CloneVM(args.vm)
    vm.refresh_ip()
    hunt(vm.ip, args.vm, args.window_s, args.dry_run)


if __name__ == "__main__":
    main()
