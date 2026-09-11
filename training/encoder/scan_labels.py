"""扫描标签数据集构建：克隆遥测日志 → scan_active 标签窗口 + 新事件流。

数据源: /var/log/lado-range/<vm>/<date>.log
  - tracee 事件行: `... <vm> tracee[pid]: {json}`（tracee[pid]: 格式，
    与旧 parse_events.parse_line 的 "tracee:" 假设不同，这里按首个 '{' 提取）
  - 扫描标记行: `lado-responder: SCAN done`（波次时间戳）

输出:
  checkpoints/scan_labels.npz   — X(N,112) y(N)（用编码器词表，含未知词的窗口丢弃）
  checkpoints/clone_events.jsonl — 新事件流（tokens_v2 格式，供编码器增量预训练）

用法:
  python3 -m training.encoder.scan_labels --vms range-l2-t1 range-l2-t2 \
      --encoder checkpoints/encoder.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
from parse_events import event_to_tokens  # noqa: E402

from training.encoder.data import N_TOKENS, STRIDE, WINDOW  # noqa: E402

logger = logging.getLogger("encoder.scan_labels")

LOG_ROOT = Path("/var/log/lado-range")
SCAN_BEFORE_S = 15   # 扫描标记向前回溯（扫描 exec 爆发在标记前后 ~15s 内）

# 背景噪音进程：systemd-resolved 的 DNS 连接、savelog/logrotate 刷屏，
# 会把窗口稀释到什么都学不到（实测 6176/16 每窗全是 resolved CONN）
NOISE_PROC_PREFIXES = ("systemd-resolve", "systemd-rc-loca", "sd-resolve",
                       "savelog", "logrotate")


def _is_noise(event: dict) -> bool:
    proc = event.get("processName", "")
    return any(proc.startswith(p) for p in NOISE_PROC_PREFIXES)


# 消杀签名：响应器扫描链的特征（readlink 读 /proc/*/exe、sudo 子进程、responder 自身）
_SWEEP_PROCS = {"PROC:readlink", "PROC:lado-respond"}
_SWEEP_PARENTS = {"PARENT:lado-respond"}


def _is_sweep_event(tokens: list[str]) -> bool:
    proc = tokens[1] if len(tokens) > 1 else ""
    parent = tokens[3] if len(tokens) > 3 else ""
    if proc in _SWEEP_PROCS or parent in _SWEEP_PARENTS:
        return True
    # sudo 拉起的扫描子进程（find/ps/grep/xargs 经 sudo）
    if parent == "PARENT:sudo" and proc.startswith("PROC:"):
        return True
    return False


def parse_clone_log(vm_name: str, day: str | None = None) -> tuple[list[dict], list[float]]:
    """解析单台克隆日志 → (tracee 事件流, 扫描时间戳列表)。

    事件: {"ts": iso, "epoch_s": float, "tokens": [...]}
    扫描时间戳: epoch 秒列表

    读该 VM 的全部日期文件：时钟偏移的 VM 会写入不同日期的文件
    （rsyslog 模板用的是消息携带的客户端时间戳）。
    """
    if day:
        paths = [LOG_ROOT / vm_name / f"{day}.log"]
    else:
        paths = sorted((LOG_ROOT / vm_name).glob("*.log"))
    events: list[dict] = []
    scan_ts: list[float] = []
    prev_epoch: float | None = None
    cut_idx = 0   # 最后一次时钟前跳时的事件下标（之前的事件全部丢弃）

    for path in paths:
        if not path.exists():
            continue
        prev_line_ts: datetime | None = None
        for line in path.read_text(errors="replace").splitlines():
            # 时钟偏移检测：行墙钟相比前一行向前跳 >2h → 记录切点
            try:
                line_ts = datetime.fromisoformat(line.split()[0])
            except (ValueError, IndexError):
                line_ts = None
            if line_ts and prev_line_ts:
                jump = (line_ts - prev_line_ts).total_seconds()
                if jump > 7200:
                    cut_idx = len(events)
            if line_ts:
                prev_line_ts = line_ts
            if '"eventName"' not in line:
                continue
            j = line.find("{")
            if j < 0:
                continue
            try:
                event = json.loads(line[j:])
            except json.JSONDecodeError:
                continue
            epoch_s = event.get("timestamp", 0) / 1e9

            # 扫描标记：从 tracee 事件流里识别响应器自身活动
            # （syslog 行时间戳是 VM 墙钟误贴宿主机时区，差 8h，不可用）
            if event.get("eventName") == "sched_process_exec":
                proc = event.get("processName", "")
                argv_str = str(event.get("args", []))
                if "lado-responder" in argv_str or proc == "lado-responder":
                    scan_ts.append(epoch_s)

            if _is_noise(event):
                continue

            delta_ms = 0.0 if prev_epoch is None else max(0.0, (epoch_s - prev_epoch) * 1000)
            prev_epoch = epoch_s
            tokens = event_to_tokens(event, delta_ms)
            events.append({
                # 标签用系统墙钟（syslog 行 ts 的小时）：
                # cron/anacron/系统行为跟墙钟走，tracee 内核对 date -s 免疫
                "ts": line.split()[0],
                "epoch_s": epoch_s,
                "host": vm_name,
                "tokens": tokens,
            })
    # 事件按时间重排（多文件拼接后保证 ΔT 与窗口顺序正确）
    events.sort(key=lambda e: e["epoch_s"])
    if cut_idx > 0:
        # 丢弃时钟偏移前的旧行为事件（它们是"白天"标签下的夜间行为，污染标签）
        logger.info(f"{vm_name}: 时钟偏移切点 {cut_idx}/{len(events)}，丢弃偏移前事件")
        events = events[cut_idx:]
    return events, scan_ts


def build(vms: list[str], vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X_rows, y_rows, all_events = [], [], []
    n_drop = 0
    for vm in vms:
        events, scan_ts = parse_clone_log(vm)
        all_events.extend(events)
        scan_arr = np.array(sorted(scan_ts))
        logger.info(f"{vm}: 事件={len(events)} 扫描标记={len(scan_ts)}")

        for start in range(0, len(events) - WINDOW - 1, STRIDE):
            win = events[start:start + WINDOW]
            # 内容定义标签：窗口内含 ≥2 个消杀签名事件 → scan_active=1
            # （时间对齐标签在繁忙背景下 ~50% 是哑的——窗口里根本没有扫描事件。
            #  探针的目的是验证编码器保留"消杀签名在场"的信息，
            #  这正是 bee 运行时需要的检测能力）
            n_sig = sum(1 for e in win if _is_sweep_event(e["tokens"]))
            in_scan = n_sig >= 2
            flat = [t for e in win for t in e["tokens"]]
            try:
                ids = [vocab[t] for t in flat]
            except KeyError:
                n_drop += 1
                continue
            X_rows.append(ids)
            y_rows.append(1 if in_scan else 0)

    X = np.array(X_rows, dtype=np.int64) if X_rows else np.zeros((0, WINDOW * N_TOKENS), dtype=np.int64)
    y = np.array(y_rows, dtype=np.int64) if y_rows else np.zeros((0,), dtype=np.int64)
    logger.info(f"窗口={len(y)} scan=1 占 {y.mean() if len(y) else 0:.2%} 丢弃(未知词)={n_drop}")
    return X, y, all_events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vms", nargs="+", required=True)
    ap.add_argument("--encoder", default="checkpoints/encoder.pt")
    ap.add_argument("--out", default="checkpoints/scan_labels.npz")
    ap.add_argument("--events-out", default="checkpoints/clone_events.jsonl")
    args = ap.parse_args()

    ckpt = torch.load(args.encoder, weights_only=False)
    vocab: dict[str, int] = ckpt["vocab"]

    X, y, events = build(args.vms, vocab)
    np.savez(args.out, X=X, y=y)

    with open(args.events_out, "w") as f:
        for e in events:
            f.write(json.dumps({"ts": e["ts"], "host": e["host"],
                                "tokens": e["tokens"]}) + "\n")
    logger.info(f"保存 → {args.out}  事件流 → {args.events_out} ({len(events)} 条)")


if __name__ == "__main__":
    main()
