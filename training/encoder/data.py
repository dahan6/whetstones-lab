"""编码器数据：真实 Tracee token 流 → 窗口样本 + 探针标签。

输入格式（lado-range/detector/data/tokens_v2.jsonl）:
  {"ts": "...", "host": "...", "tokens": [ET, PROC, ARGV, PARENT, UID, DST, DT]}

窗口样本: W=16 个连续事件 × 7 token。
探针标签（自监督，无需人工标注）:
  - day_night: 窗口末事件的小时（6~22 为白天）
  - proc_class: 窗口内主导进程的粗类（sshd/cron/systemd/journald/other）
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

WINDOW = 16
STRIDE = 8
N_TOKENS = 7

PROC_CLASSES = ["sshd", "cron", "systemd", "journald", "other"]

# 背景噪音进程（与 scan_labels 同源）：resolved 的 DNS 连接、savelog 刷屏
# 会淹没窗口。旧数据 tokens_v2 采集时未过滤，统一在加载时过滤。
NOISE_PROC_PREFIXES = ("systemd-resolve", "systemd-rc-loca", "sd-resolve",
                       "savelog", "logrotate")


def _is_noise_tokens(tokens: list[str]) -> bool:
    proc = tokens[1] if len(tokens) > 1 else ""
    return any(proc.startswith(f"PROC:{p}") for p in NOISE_PROC_PREFIXES)


def proc_class_of(win_tokens: list[str]) -> int:
    """窗口内主导 PROC 的粗类。win_tokens 为窗口全部 token（扁平）。"""
    procs = [t.split(":", 1)[1] for t in win_tokens if t.startswith("PROC:")]
    if not procs:
        return PROC_CLASSES.index("other")
    top = Counter(procs).most_common(1)[0][0]
    for i, name in enumerate(PROC_CLASSES[:-1]):
        if name in top:
            return i
    return PROC_CLASSES.index("other")


def day_night_of(ts: str) -> int:
    """0=白天 1=夜间。"""
    hour = datetime.fromisoformat(ts).hour
    return 0 if 6 <= hour < 22 else 1


class TokenDataset:
    def __init__(self, jsonl_paths: str | Path | list,
                 vocab: dict[str, int] | None = None):
        if isinstance(jsonl_paths, (str, Path)):
            jsonl_paths = [jsonl_paths]
        self.paths = [Path(p) for p in jsonl_paths]
        self.vocab: dict[str, int] = vocab or {}
        self.id2tok: list[str] = []
        self.events: list[tuple[str, list[int]]] = []  # (ts, token_ids)
        self._load(build_vocab=vocab is None)

    def _load(self, build_vocab: bool) -> None:
        all_lines: list[str] = []
        for path in self.paths:
            all_lines.extend(Path(path).read_text().splitlines())
        if build_vocab:
            toks_all: set[str] = set()
            for line in all_lines:
                toks_all.update(json.loads(line)["tokens"])
            self.id2tok = sorted(toks_all)
            self.vocab = {t: i for i, t in enumerate(self.id2tok)}
        else:
            self.id2tok = [None] * len(self.vocab)
            for t, i in self.vocab.items():
                self.id2tok[i] = t
        for line in all_lines:
            obj = json.loads(line)
            if _is_noise_tokens(obj["tokens"]):
                continue
            try:
                ids = [self.vocab[t] for t in obj["tokens"]]
            except KeyError:
                continue   # 外部词表：含未知词的事件整条跳过
            self.events.append((obj["ts"], ids))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def windows(self) -> list[dict]:
        """滑动窗口样本。"""
        samples = []
        n = len(self.events)
        for start in range(0, n - WINDOW - 1, STRIDE):
            win = self.events[start:start + WINDOW]
            nxt = self.events[start + WINDOW]
            flat_ids = [tid for _, ids in win for tid in ids]
            flat_toks = [self.id2tok[i] for i in flat_ids]
            samples.append({
                "input": np.array(flat_ids, dtype=np.int64),
                "target": np.array(nxt[1], dtype=np.int64),   # 下一事件 7 token
                "day_night": day_night_of(win[-1][0]),
                "proc_class": proc_class_of(flat_toks),
                "ts": win[-1][0],
            })
        return samples


def build_samples(jsonl_paths, vocab: dict[str, int] | None = None) -> tuple[TokenDataset, list[dict]]:
    ds = TokenDataset(jsonl_paths, vocab=vocab)
    return ds, ds.windows()
