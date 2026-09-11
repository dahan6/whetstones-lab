"""NDR 模拟器（流量裁判）：给 bee 的 C2 通信打异常分。

检测逻辑（对应真实 NDR 的核心能力）：
  1. 周期检测：连接间隔的低方差是机器行为的指纹（固定轮询的死穴）——
     用最近 N 个间隔的变异系数(CV=std/mean)判周期性，CV<0.15 判为机器
  2. 新目标/端口检测：目标:端口不在该主机历史基线里即加分
  3. 频率异常：单位时间连接数超基线泊松上限

数据源：宿主机对靶场网桥(virbr-lado)的包捕获（tcpdump 文本解析）。
输出：score_connection(record) → 0.0~1.0 异常分；窗口聚合给 episode 打分。

用法（报联时机训练时由训练器调用）:
  from training.ndr.simulator import NdrSimulator
  ndr = NdrSimulator()
  ndr.learn_baseline(seconds=120)          # 学基线（无 bee 时）
  score = ndr.score_window(records)         # 给一段 C2 连接窗打分
"""
from __future__ import annotations

import math
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ConnRecord:
    ts: float
    src: str
    dst: str
    dport: int
    proto: str      # "tcp" | "udp"


@dataclass
class NdrSimulator:
    # 基线
    dst_port_ok: set[tuple[str, int]] = field(default_factory=set)
    rate_mean: float = 2.0
    rate_std: float = 1.0
    # 周期检测参数
    cv_threshold: float = 0.15
    window: int = 6

    def learn_baseline_from_records(self, records: list[ConnRecord]) -> None:
        """从一段"无 bee"流量学基线：常见目标:端口 + 连接速率。"""
        cnt: dict[tuple[str, int], int] = defaultdict(int)
        for r in records:
            cnt[(r.dst, r.dport)] += 1
        # 出现 ≥2 次的目标视为正常
        self.dst_port_ok = {k for k, v in cnt.items() if v >= 2}
        if len(records) >= 2 and records[-1].ts > records[0].ts:
            span = records[-1].ts - records[0].ts
            rates = [len([r for r in records if t <= r.ts < t + 10])
                     for t in range(int(records[0].ts), int(records[-1].ts), 10)]
            self.rate_mean = statistics.mean(rates) if rates else 0.0
            self.rate_std = (statistics.pstdev(rates) or 1.0) if rates else 1.0

    def score_connection(self, r: ConnRecord,
                         history: list[ConnRecord]) -> float:
        """单连接异常分（0~1）。history 是同 src→dst:port 的历史连接。"""
        s = 0.0
        # ① 新目标/端口（降权：目标新是 bee 的宿命，主攻行为模式）
        if (r.dst, r.dport) not in self.dst_port_ok:
            s += 0.15
        # ② 周期性（同 src→dst:port 的间隔 CV）
        ivs = [b.ts - a.ts for a, b in zip(history[-self.window - 1:-1],
                                           history[-self.window:])]
        if len(ivs) >= 4:
            m = statistics.mean(ivs)
            if m > 0:
                cv = statistics.pstdev(ivs) / m
                if cv < self.cv_threshold:
                    # 周期惩罚升权：完美周期是 C2 的头号行为指纹
                    s += 0.85 * (1 - cv / self.cv_threshold)
        return min(s, 1.0)

    def total_sequence_cv(self, records: list[ConnRecord]) -> float:
        """总报联序列的周期分（不分 flow）——C2 换端口也躲不掉。"""
        if len(records) < 5:
            return 0.0
        ts = sorted(r.ts for r in records)
        ivs = [b - a for a, b in zip(ts[:-1], ts[1:])]
        m = statistics.mean(ivs)
        if m <= 0:
            return 0.0
        cv = statistics.pstdev(ivs) / m
        return 0.85 * max(0.0, 1 - cv / self.cv_threshold) if cv < self.cv_threshold else 0.0

    def score_window(self, records: list[ConnRecord]) -> dict:
        """给一段 C2 连接窗打聚合分。"""
        by_flow: dict[tuple, list[ConnRecord]] = defaultdict(list)
        for r in records:
            by_flow[(r.src, r.dst, r.dport)].append(r)
        total, per = 0.0, 0
        for (src, dst, dport), hist in by_flow.items():
            hist.sort(key=lambda x: x.ts)
            for i, r in enumerate(hist):
                total += self.score_connection(r, hist[:i + 1])
                per += 1
        # ③ 全局频率：src 总报联频率超基线（通道交替规避 flow 周期检测的兜底）
        by_src: dict[str, list[ConnRecord]] = defaultdict(list)
        for r in records:
            by_src[r.src].append(r)
        freq_pen = 0.0
        for src, recs in by_src.items():
            span = max(recs[-1].ts - recs[0].ts, 1.0)
            rate = len(recs) / (span / 60.0)   # 次/分钟
            if rate > self.rate_mean + 2 * max(self.rate_std, 1.0):
                freq_pen = min((rate - self.rate_mean) / 10.0, 0.6)
        seq_cv_pen = self.total_sequence_cv(records)
        return {"ndr_score": total / max(per, 1) + freq_pen + seq_cv_pen,
                "connections": per}


# ── 包捕获（tcpdump 文本解析）─────────────────────────────────

def capture_virbr(iface: str, seconds: float,
                  sport_filter: tuple[int, ...] = ()) -> list[ConnRecord]:
    """抓靶场网桥流量 seconds 秒，解析为 ConnRecord。"""
    cmd = ["timeout", str(int(seconds) + 1), "tcpdump", "-i", iface, "-nn", "-q",
           "tcp or udp"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    recs = []
    for line in out.splitlines():
        # 形如: 16:00:01.123456 IP 192.0.2.119.54321 > 192.0.2.1.8443: tcp 60
        try:
            ts_part, rest = line.split(" IP ", 1)
            proto = "udp" if "UDP" in line or " udp " in line else "tcp"
            src_part, dst_part = rest.split(" > ", 1)
            src = src_part.rsplit(".", 1)[0]
            dst_full = dst_part.split(":")[0]
            dst = dst_full.rsplit(".", 1)[0]
            dport = int(dst_full.rsplit(".", 1)[1])
            h, m, s = ts_part.split(":")
            ts = int(h) * 3600 + int(m) * 60 + float(s)
            if not sport_filter or dport in sport_filter:
                recs.append(ConnRecord(ts, src, dst, dport, proto))
        except Exception:
            continue
    return recs
