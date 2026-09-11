"""遥测收集器：读 rsyslog 汇聚的 VM 日志，解析成结构化事件。

数据源：/var/log/lado-range/<vm>/<date>.log
  - lado-responder: 扫描/消杀动作（SCAN/KILL/CLEAN/DISABLE/ELIMINATE）
  - falco: JSON 告警行（falco 输出 json_output=true）
  - tracee: JSON 事件行（sched_process_exec 等）

纪律：这里只做"收集与解析"，绝不给行为打分——裁决归 Falco/响应器，
奖励计算归 reward.py。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger("orchestrator.telemetry")

LOG_ROOT = Path("/var/log/lado-range")

RESPONDER_RE = re.compile(r"lado-responder: (?P<action>\w+) (?P<rest>.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class EliminationEvent:
    kind: str          # KILL / CLEAN / DISABLE
    detail: dict


@dataclass
class WaveEvent:
    hits: int
    killed: int
    cleaned: int
    disabled: int


@dataclass
class TelemetryWindow:
    """一个 episode 时间窗内收集到的事件。"""
    falco_alerts: list[dict] = field(default_factory=list)
    waves: list[WaveEvent] = field(default_factory=list)
    eliminations: list[EliminationEvent] = field(default_factory=list)
    raw_lines: int = 0


def _log_path(vm_name: str, day: date | None = None) -> Path:
    d = day or date.today()
    return LOG_ROOT / vm_name / f"{d.year}-{d.month:02d}-{d.day:02d}.log"


def parse_line(line: str, win: TelemetryWindow) -> None:
    win.raw_lines += 1

    m = RESPONDER_RE.search(line)
    if m:
        action, rest = m.group("action"), m.group("rest")
        kv = dict(KV_RE.findall(rest))
        if action == "SCAN" and "hits" in kv:
            # SCAN 行先到，ELIMINATE 行后补全 wave
            win.waves.append(WaveEvent(hits=int(kv["hits"]), killed=0,
                                       cleaned=0, disabled=0))
        elif action == "ELIMINATE":
            if win.waves:
                w = win.waves[-1]
                w.killed = int(kv.get("killed", 0))
                w.cleaned = int(kv.get("cleaned", 0))
                w.disabled = int(kv.get("disabled", 0))
        elif action in ("KILL", "CLEAN", "DISABLE"):
            win.eliminations.append(EliminationEvent(kind=action, detail=kv))
        return

    # Falco / Tracee 输出 JSON（经 syslog 转发，行内嵌 JSON）
    if '"rule"' in line or '"eventName"' in line:
        jstart = line.find("{")
        if jstart >= 0:
            try:
                obj = json.loads(line[jstart:])
            except json.JSONDecodeError:
                return
            if "rule" in obj:          # falco 告警
                rule = str(obj.get("rule", ""))
                if rule.startswith("Falco internal"):  # 内部噪音，不算裁决
                    return
                win.falco_alerts.append(obj)
            # tracee 事件暂不全量存（量大），episode 分析需要时再加


def collect_window(vm_name: str, since_line: int = 0) -> tuple[TelemetryWindow, int]:
    """收集 VM 日志中 since_line 之后的事件。

    返回 (窗口, 当前总行数)。episode 开始时记行号，结束时取增量，
    避免跨 episode 串扰。

    流式读取：日志可达 400MB+/天，全量 read_text 的字符串列表会让
    pymalloc arena 只涨不还（v4/v5 训练驱动器 OOM 至 92-101GB 的根因）。
    """
    path = _log_path(vm_name)
    win = TelemetryWindow()
    if not path.exists():
        logger.warning(f"日志不存在: {path}（检查 rsyslog 权限/转发）")
        return win, since_line
    n = 0
    with open(path, errors="replace") as f:
        for line in f:
            if n >= since_line:
                parse_line(line, win)
            n += 1
    return win, n


def mark(vm_name: str) -> int:
    """记录当前日志行号（episode 起点）。流式计数，不全量读。"""
    path = _log_path(vm_name)
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n
