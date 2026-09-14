"""司命猎手：行为语法检出 → 消杀令（检测-消杀闭环，L4 的捕食者之牙）。

原理：读 VM 遥测日志窗口，按父进程名聚合事件链，计算链内事件间隔 CV 与
节奏规律性。CV < cv_low（司命 P4 判据，默认 1.5）且样本量足够的链 =
机器节奏，判定异常。命中后按父进程名在 VM 上定位并消杀。

检出依据与司命检测器同源（二阶时序分析），但本模块是巢侧裁决臂：
它只产出"杀谁"，消杀动作在 VM 上经 logger -t lado-responder 落证。

用法:
  /usr/bin/python3 -m training.orchestrator.siming_hunter --vm range-vm-0
  /usr/bin/python3 -m training.orchestrator.siming_hunter --vm range-vm-0 --dry-run
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


def parent_disposable(ip: str, pid: str, chain_evidence: str = "") -> tuple[bool, str]:
    """处决前验明正身。v2（E-M 修订）：exe + cmdline 双查 + 链证据可作依据。

    v1 教训（E-M §35）：只查 /proc/pid/exe 时，bash/python 脚本木马的 exe
    永远是解释器 /usr/bin/bash → "系统路径" → 结构性放行。v2 补 cmdline
    （脚本本体路径）；且链证据（connect 信标/超低 CV）本身可作处决依据——
    行为即身份，衣服（exe 位置）不再一票否决。
    """
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
    # v2①：解释器身体 → 查 cmdline 里的脚本本体路径
    rc, cmdl, _ = ssh_exec(ip, f"tr '\\0' ' ' < /proc/{pid}/cmdline 2>/dev/null || true")
    cmdl = cmdl.strip()
    for tok in cmdl.split()[:4]:
        if any(tok.startswith(p) for p in ("/tmp/", "/dev/shm/", "/var/tmp/")):
            return True, f"tmp-script: {tok}"
        if "/." in tok and any(tok.startswith(p) for p in SUSPICIOUS_EXE):
            return True, f"hidden-script: {tok}"
    # v2②：链证据本身可处决——低速率机器信标是行为实锤，不因身体在系统目录而豁免
    if chain_evidence and exe in ("/usr/bin/bash", "/bin/bash", "/usr/bin/python3", "/usr/bin/dash", "/bin/sh"):
        return True, f"beacon-interp: {exe} [{chain_evidence}]"
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
                "intervals": [round(x, 3) for x in iv[:60]],
                "cv": round(cv, 3), "mean_interval_s": round(mean, 2),
            })
    return suspects


def scan_connects(vm_name: str, window_s: float = 900.0, cv_low: float = CV_LOW,
                  min_events: int = 5) -> list[dict]:
    """connect 信标扫描（E-M v2 新增感知面）。

    按 (dst_ip, dst_port) 聚合 AF_INET connect 事件，算间隔 CV——
    低速率信标（RAT 60s/矿机 30s）在 exec 链上凑不够 min_events=20，
    但 connect 序列 5 个事件即可判节奏。排除 DNS/syslog/ssh 环境噪声。
    """
    path = _log_path(vm_name)
    if not path.exists():
        return []
    now_ns = time.time() * 1e9
    t0 = now_ns - window_s * 1e9
    seqs: dict[tuple, list] = defaultdict(list)
    parent_of: dict[int, int] = {}   # 遥测家谱：pid → ppid（exec 事件供给）
    with open(path, errors="replace") as f:
        for line in f:
            is_exec = "sched_process_exec" in line
            if not is_exec and ("security_socket_connect" not in line or "AF_INET" not in line):
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
            if is_exec:
                pid, ppid = e.get("processId"), e.get("parentProcessId")
                if pid and ppid:
                    parent_of[pid] = ppid
                continue
            if '"sin_port":"53"' in line or '"sin_port":"514"' in line or '"sin_port":"22"' in line:
                continue
            if '"sin_addr":"0.0.0.0"' in line or '"sin_addr":"127.0.0.53"' in line:
                continue
            dst = ""
            for a in e.get("args", []):
                if a.get("name") == "remote_addr":
                    v = a.get("value", {})
                    dst = f"{v.get('sin_addr')}:{v.get('sin_port')}"
            if not dst:
                continue
            seqs[dst].append((ts, e.get("processId"), e.get("parentProcessId"),
                              e.get("processName", "?")))

    suspects = []
    for dst, lst in seqs.items():
        if len(lst) < min_events:
            continue
        lst.sort()
        iv = [(b[0] - a[0]) / 1e9 for a, b in zip(lst, lst[1:]) if b[0] > a[0]]
        if len(iv) < min_events - 1:
            continue
        mean = statistics.mean(iv)
        if not (0.5 <= mean <= 600):
            continue
        cv = statistics.stdev(iv) / mean
        if cv >= cv_low:
            continue
        # 处决目标解析：连接是 3s 短命子进程，循环本体在祖先。
        # 家族可能多实例交错（重部署残留）且遥测簿记有一跳抖动——
        # 取最近 5 次 connect 的祖先链并集，逐个候选处决（hunt 端验活+验身）
        proc_ids: dict = defaultdict(int)
        for _, pid, ppid, pn in lst:
            proc_ids[(pid, pn)] += 1
        (_, proc_hint), _ = max(proc_ids.items(), key=lambda kv: kv[1])
        candidates = []
        for _, pid, ppid, pn in lst[-5:]:
            ancestor = ppid or pid
            for _ in range(6):
                up = parent_of.get(ancestor)
                if not up or up == 1:
                    break
                ancestor = up
            if ancestor and ancestor > 1 and ancestor not in candidates:
                candidates.append(ancestor)
        suspects.append({
            "kind": "beacon", "parent": proc_hint,
            "intervals": [round(x, 2) for x in iv[:60]],
            "pid_hint": candidates[0] if candidates else pid,
            "pid_candidates": candidates, "child_hint": pid, "ppid_hint": ppid,
            "dst": dst, "cv": round(cv, 3), "n_events": len(lst),
            "mean_interval_s": round(mean, 2),
        })
    return suspects


def hunt(ip: str, vm_name: str, window_s: float = 300.0,
         dry_run: bool = False, cooldown: dict | None = None) -> list[dict]:
    """找机器节奏链与 connect 信标并消杀。返回判决记录。

    纪律（v2，E-M §35 修订）：行为证据即处决依据——链 CV 与 connect 信标
    本身可授权处决，不再仅凭 exe 位置一票否决；杀后株连复活链
    （cron 引用清理）。系统链仍由白名单+验身保护，误报纪律不变。
    """
    suspects = scan_chains(vm_name, window_s, cv_low=2.5)   # 宽门进，模型+旧门双轨定罪
    verdicts = []
    killed_names = set()
    killed_pids = set()
    from .model_judge import judge as model_judge_fn
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
        mj = model_judge_fn(s.get("intervals") or [], is_suspicious=1)
        s["model"] = mj
        model_convict = ((mj["danger"] is not None and mj["danger"] >= mj["meta_thr"])
                         or (mj["machine_prob"] is not None and mj["machine_prob"] >= 0.85))
        if s["cv"] >= CV_LOW and not model_convict:
            continue   # 旧门不中且模型不定罪 → 放行（记录在 s["model"]）
        rec = {**s, "pids": pids, "action": "none", "model_convict": model_convict}
        if not pids:
            rec["action"] = "not_found"
        else:
            # 验明正身 v2：位置 + 证据双轨授权
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
                _execute_kill(ip, parent, good_pids, evidence, s)
                killed_pids.update(good_pids)
                rec["action"] = f"killed {len(good_pids)} ({evidence})"
                killed_names.add(parent)
                if cooldown is not None:
                    cooldown[parent] = time.time()
        verdicts.append(rec)
        if "killed" in rec["action"] or "verified_skip" in rec["action"]:
            logger.info(f"判决: {parent} cv={s['cv']} n={s['n_events']} → {rec['action']}")

    # v2：connect 信标面——证据授权，直接定位信标进程
    for s in scan_connects(vm_name):
        key = f"beacon:{s['dst']}"
        if cooldown and key in cooldown and time.time() - cooldown[key] < 300:
            continue
        pid, cmdl = None, ""
        for cand in s.get("pid_candidates") or [s["pid_hint"]]:
            if cand in killed_pids:
                continue
            rc, out, _ = ssh_exec(ip, f"test -d /proc/{cand} && tr '\\0' ' ' < /proc/{cand}/cmdline 2>/dev/null || true")
            if out.strip():
                pid, cmdl = cand, out.strip()
                break
        if pid is None:
            rec0 = {**s, "action": "gone"}
            verdicts.append(rec0)
            continue
        if not cmdl:
            # 子进程短命 → 沿父链上溯（bash→timeout→循环本体，最多 4 代）
            cand = s.get("ppid_hint")
            for _ in range(4):
                if not cand or cand == 1:
                    break
                rc, out, _ = ssh_exec(ip, f"test -d /proc/{cand} && tr '\\0' ' ' < /proc/{cand}/cmdline 2>/dev/null || true")
                cl = out.strip()
                if cl:
                    pid, cmdl = cand, cl
                    break
                rc, out, _ = ssh_exec(ip, f"awk '/^PPid/{{print $2}}' /proc/{cand}/status 2>/dev/null", )
                cand = (out or "").strip() or None
        mj = model_judge_fn(s.get("intervals") or [], is_suspicious=1)
        s["model"] = mj
        rec = {**s, "cmdline": cmdl[:120], "action": "none"}
        if not cmdl:
            rec["action"] = "gone"
        else:
            ok, why = parent_disposable(ip, pid, chain_evidence=f"beacon {s['dst']} cv={s['cv']} n={s['n_events']}")
            rec["verify"] = why
            if not ok:
                rec["action"] = f"verified_skip ({why})"
            elif dry_run:
                rec["action"] = f"dry_run ({why})"
            else:
                _execute_kill(ip, s["parent"], [str(pid)], why, s)
                killed_pids.add(str(pid))
                rec["action"] = f"killed 1 ({why})"
                if cooldown is not None:
                    cooldown[key] = time.time()
        verdicts.append(rec)
        if "killed" in rec["action"] or "verified_skip" in rec["action"]:
            logger.info(f"信标判决: {s['dst']} cv={s['cv']} n={s['n_events']} → {rec['action']}")
    if not suspects:
        logger.info(f"窗口 {window_s}s 内无机器节奏链（CV<{CV_LOW}）")
    return verdicts


def _execute_kill(ip: str, name: str, pids: list[str], evidence: str, s: dict) -> None:
    """处决 + 株连：先取证（cmdline/载体）再杀——kill -9 后 /proc 即焚。"""
    bodies = []
    for p in pids:
        rc2, exe, _ = ssh_exec(ip, f"readlink -f /proc/{p}/exe 2>/dev/null || true")
        exe = exe.strip().replace(" (deleted)", "")
        rc3, cmdl, _ = ssh_exec(ip, f"tr '\\0' ' ' < /proc/{p}/cmdline 2>/dev/null || true")
        for tok in (cmdl or "").split()[:4]:
            if tok.startswith(("/tmp/", "/dev/shm/", "/var/tmp/", "/home/")) and len(tok) > 4:
                bodies.append(tok)
        ssh_exec(ip, f"sudo -n kill -9 {p} 2>/dev/null")
        if exe and not exe.startswith(("/usr/bin/bash", "/bin/bash", "/usr/bin/python3")):
            bodies.append(exe)
    # 株连：删载体 + 清 cron 引用
    for b in set(bodies):
        ssh_exec(ip, f"sudo -n rm -f '{b}' 2>/dev/null || true; "
                     f"crontab -l 2>/dev/null | grep -vF '{b}' | crontab - 2>/dev/null || true")
    ssh_exec(ip, f"logger -t lado-responder \"SIMING_KILL name={name} "
                 f"cv={s.get('cv')} n={s.get('n_events')} dst={s.get('dst','-')} "
                 f"evidence={evidence[:80]} bodies={','.join(set(bodies))[:80]} pids={' '.join(pids)}\"")


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
