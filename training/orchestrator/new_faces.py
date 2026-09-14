#!/usr/bin/env python3
"""E-H′ v2 新增三个检测面（§42）：falco 接线之外的两个遥测面。

- scan_fanout：网络散射（同源短窗触碰 ≥K 个不同 dst:port——LLM 侦察形状）
- scan_combo：一次性高危组合（同 uid 窗口内 ≥2 类敏感动作——Atomic TTP 形状）

返回判决记录（检出证据为主，处决仍走授权制）。
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

LOG_ROOT = Path("/var/log/lado-range")

FANOUT_MIN_DSTS = 6          # 同源 5 分钟内 ≥6 个不同目的地 = 扫描形状
FANOUT_WINDOW_S = 300
COMBO_WINDOW_S = 240
COMBO_MIN_CATS = 2

COMBO_PATTERNS = {
    "sensitive_read": re.compile(r"shadow|sudoers|/etc/passwd|id_rsa|\.pem|credentials", re.I),
    "account_create": re.compile(r"useradd|adduser|passwd \w+"),
    "cron_modify": re.compile(r"crontab"),
    "kmod_load": re.compile(r"insmod|modprobe"),
    "archive_collect": re.compile(r"^(zip|tar|gzip) .*(home|etc|root)"),
    "perm_escalate": re.compile(r"chmod .*[47]77|chmod .*\\+s|setuid"),
    "history_clear": re.compile(r"history -c|bash_history.*unlink|rm .*bash_history|shred .*history"),
    "cred_search": re.compile(r"find .*(-name|grep).*(\.aws|credential|secret|\.ssh|password)", re.I),
    "service_create": re.compile(r"systemctl (enable|start|daemon-reload)|systemd/system/.*\.service"),
}


def _iter_events(vm_name: str, t0_ns: float, t1_ns: float):
    path = LOG_ROOT / vm_name / f"{time.strftime('%F')}.log"
    if not path.exists():
        return
    with open(path, errors="replace") as f:
        for line in f:
            i = line.find("{")
            if i < 0:
                continue
            try:
                e = json.loads(line[i:])
            except Exception:
                continue
            ts = e.get("timestamp", 0)
            if t0_ns <= ts <= t1_ns:
                yield line, e


def scan_fanout(vm_name: str, window_s: float = FANOUT_WINDOW_S,
                min_dsts: int = FANOUT_MIN_DSTS) -> list[dict]:
    """网络散射：同源 uid 触碰大量不同目的 = 侦察/扫描形状（非周期）。"""
    now = time.time() * 1e9
    t0 = now - window_s * 1e9
    by_src: dict[int, set] = defaultdict(set)
    detail: dict[int, list] = defaultdict(list)
    for line, e in _iter_events(vm_name, t0, now):
        if "security_socket_connect" not in line or "AF_INET" not in line:
            continue
        if any(f'"sin_port":"{p}"' in line for p in ("53", "514", "22")):
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
        uid = e.get("userId", -1)
        by_src[uid].add(dst)
        detail[uid].append((e.get("processName", "?"), dst))
    out = []
    for uid, dsts in by_src.items():
        if len(dsts) >= min_dsts:
            out.append({
                "kind": "fanout", "uid": uid, "n_dsts": len(dsts),
                "dsts": sorted(dsts)[:12],
                "procs": sorted({p for p, _ in detail[uid]})[:6],
            })
    return out


def scan_combo(vm_name: str, window_s: float = COMBO_WINDOW_S,
               min_cats: int = COMBO_MIN_CATS) -> list[dict]:
    """一次性高危组合：窗口内同一 uid 出现 ≥min_cats 类敏感动作。

    Atomic TTP 单发无节奏，但一次攻击链的"动作类别组合"本身就是签名。
    """
    now = time.time() * 1e9
    t0 = now - window_s * 1e9
    by_uid: dict[int, dict] = defaultdict(lambda: defaultdict(list))
    for line, e in _iter_events(vm_name, t0, now):
        if "sched_process_exec" not in line:
            continue
        proc = e.get("processName", "")
        argv = ""
        for a in e.get("args", []):
            if a.get("name") == "argv":
                argv = " ".join(str(x) for x in (a.get("value") or []))[:300]
                break
        text = f"{proc} {argv}"
        for cat, pat in COMBO_PATTERNS.items():
            if pat.search(text):
                by_uid[e.get("userId", -1)][cat].append(f"{proc}: {argv[:80]}")
    out = []
    for uid, cats in by_uid.items():
        if len(cats) >= min_cats:
            out.append({
                "kind": "combo", "uid": uid,
                "cats": {c: v[:2] for c, v in cats.items()},
                "n_cats": len(cats),
            })
    return out


if __name__ == "__main__":
    import sys as _sys
    vm = _sys.argv[1] if len(_sys.argv) > 1 else "range-vm-2"
    print("fanout:", json.dumps(scan_fanout(vm), ensure_ascii=False)[:400])
    print("combo:", json.dumps(scan_combo(vm), ensure_ascii=False)[:400])
