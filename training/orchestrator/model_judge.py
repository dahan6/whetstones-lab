#!/usr/bin/env python3
"""模型判决器：把眼的训练模型接进实战猎杀链（§36 遗留的"模型化 hunt"首段）。

- RhythmClassifier（eye_rhythm.pt）：25 维间隔分布 → {machine, human, system}
- MetaScorer（eye_meta_v2.pt）：12 维面分数 → danger 概率（deploy_threshold 随 ckpt）

判决角色：替代/并联旧 CV<1.5 统计门——检测交给学习模型，
授权（parent_disposable 证据制）与误报纪律不变。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

EYE_DIR = os.environ.get("WHETSTONES_EYE_DIR", "models/eye")


class RhythmClassifier(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


class MetaScorer(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return torch.sigmoid(self.out(self.body(x))).squeeze(-1)


_CACHE: dict | None = None


def load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    rd = torch.load(f"{EYE_DIR}/eye_rhythm.pt", map_location="cpu", weights_only=False)
    md = torch.load(f"{EYE_DIR}/eye_meta_v2.pt", map_location="cpu", weights_only=False)
    rhythm = RhythmClassifier(rd.get("input_dim", 25))
    rhythm.load_state_dict(rd["state"])
    rhythm.eval()
    meta = MetaScorer(md.get("input_dim", 12))
    meta.load_state_dict(md["state"])
    meta.eval()
    _CACHE = {"rhythm": rhythm, "meta": meta,
              "meta_thr": float(md.get("deploy_threshold", 0.030))}
    return _CACHE


def rhythm_features(intervals: list[float]) -> np.ndarray | None:
    if len(intervals) < 8:
        return None
    a = np.array(intervals, dtype=np.float64)
    m, s = a.mean(), a.std()
    if m == 0:
        return None
    cv = s / m
    skew = float(((a - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    kurt = float(((a - m) ** 4).mean() / s ** 4) if s > 0 else 3.0
    burst = float((s - m) / (s + m)) if (s + m) > 0 else 0.0
    log_iv = np.log10(np.clip(a, 0.01, 300))
    hist, _ = np.histogram(log_iv, bins=16, range=(-2, 2.5), density=True)
    return np.concatenate([[cv, skew, kurt, burst, m, np.median(a), a.max(), a.min(), len(a)], hist])


@torch.no_grad()
def judge(intervals: list[float], is_suspicious: int = 1) -> dict:
    """间隔序列 → 模型判决。返回 machine_prob / danger / 阈值。"""
    models = load()
    out = {"machine_prob": None, "danger": None, "meta_thr": models["meta_thr"]}
    rf = rhythm_features(intervals)
    if rf is not None:
        logits = models["rhythm"](torch.tensor(rf.reshape(1, -1), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1)[0]
        out["machine_prob"] = float(probs[0])
    if len(intervals) >= 4:
        a = np.array(intervals, dtype=np.float64)
        m, s = a.mean(), a.std()
        feats = np.array([
            s / m if m > 0 else 0.0,
            (s - m) / (s + m) if (s + m) > 0 else 0.0,
            float(((a - m) ** 3).mean() / s ** 3) if s > 0 else 0.0,
            m, len(intervals),
            0, 0, 0, 0, 0,                       # P2/P3/P7/P0/P1 代理（待接）
            (time.time() % 86400) / 3600,
            is_suspicious,
        ], dtype=np.float32)
        danger = models["meta"](torch.tensor(feats.reshape(1, -1)))
        out["danger"] = float(danger)
    return out
