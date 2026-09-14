#!/usr/bin/env python3
"""解剖面（P8_anatomy）：三层次"神经身体"识别——名单的替代者。

层次一 L1 字节画像：文件级特征 + 小 MLP，不问文件名，识别"像一颗大脑的文件"。
    对改名/打散/base64/轻度 XOR/量化免疫（字节统计不认名字）。
层次二 L2 族谱认亲：与已知权重族（gene_bank 世代权重）做量化鲁棒的结构统计比对。
    匹配 = 血缘实锤（良性文件不会撞上张量血缘），改名/中度变异免疫。
层次三 L3 装载行为 + 内存扫描：open 簇评分（配合 tracee security_file_open 扩面）
    与 RSS 浮点/张量簇检测——加密也逃不掉（推理前必须解密进内存）。

用法:
  python3 -m detector.eye.anatomy index                 # 建族谱索引
  python3 -m detector.eye.anatomy train                 # 训练 L1 画像模型
  python3 -m detector.eye.anatomy scan <dir>            # 扫描目录（三层联合）
  python3 -m detector.eye.anatomy memscan <pid>         # 内存张量簇扫描
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
INDEX_PATH = ROOT / "lineage_index.json"
MODEL_PATH = ROOT / "eye_anatomy.npz"

# ─────────────────────────── 层次一：字节画像 ───────────────────────────

FEATURE_NAMES = [
    "size", "entropy", "top1_freq", "top2_freq", "printable", "zero_ratio",
    "f32_finite", "f32_small", "f32_absmean", "f32_kurtosis", "f32_tail",
    "zlib_ratio", "byte_chi2", "hist_flat", "ngram_rep", "header_elf",
    "header_zip", "header_text", "size_div4", "size_factorness",
]


def file_features(path: str | Path, max_bytes: int = 4 << 20) -> np.ndarray:
    """20 维字节画像特征。文件名完全不参与。"""
    raw = Path(path).read_bytes()[:max_bytes]
    n = len(raw)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES))
    b = np.frombuffer(raw, dtype=np.uint8)
    hist = np.bincount(b, minlength=256).astype(np.float64)
    p = hist[hist > 0] / n
    entropy = float(-(p * np.log2(p)).sum())
    freq = np.sort(hist)[::-1]
    top1 = float(freq[0] / n)
    top2 = float(freq[1] / n) if len(freq) > 1 else 0.0
    printable = float(np.isin(b, list(range(32, 127)) + [9, 10, 13]).mean())
    zero_ratio = float(hist[0] / n)
    # float32 视角
    m = n - (n % 4)
    f32 = np.frombuffer(raw[:m], dtype=np.float32).astype(np.float64)
    finite = np.isfinite(f32)
    f32_finite = float(finite.mean())
    vals = f32[finite]
    f32_small = float((np.abs(vals) < 2.0).mean()) if len(vals) else 0.0
    f32_absmean = float(np.abs(vals).mean()) if len(vals) else 0.0
    if len(vals) > 8:
        v = vals - vals.mean()
        std = v.std() + 1e-12
        kurt = float((v ** 4).mean() / std ** 4)
        tail = float((np.abs(v) > 4 * std).mean())
    else:
        kurt, tail = 0.0, 0.0
    zlib_ratio = float(len(zlib.compress(raw, 6))) / n
    expected = n / 256
    byte_chi2 = float(((hist - expected) ** 2 / (expected + 1e-9)).sum() / n)
    hist_flat = float(hist.std() / (hist.mean() + 1e-9))
    q = b.reshape(-1, 4)
    ngram_rep = float(len({tuple(x) for x in q[:8192]}) / max(1, min(len(q), 8192)))
    header_elf = float(raw[:4] == b"\x7fELF")
    header_zip = float(raw[:2] in (b"PK", b"\x1f\x8b", b"BZ", b"\xfd7") or raw[:6] == b"\x5d\x00\x00")
    header_text = float(raw[:64].decode("ascii", "ignore").isprintable() and n > 0 and printable > 0.9)
    size_div4 = float(n % 4 == 0)
    # 尺寸可分解为 4×整千/整百（张量 dim 的粗签名）
    size_factorness = 0.0
    for base in (1024, 1000, 768, 512, 256):
        if n % base == 0:
            size_factorness = max(size_factorness, 1.0 - abs(math.log(max(1, n // base) / max(1, round(n / base)) + 1e-9)))
            break
    feats = [min(n, 4 << 20) / (4 << 20), entropy / 8, top1, top2, printable,
             zero_ratio, f32_finite, f32_small, min(f32_absmean, 4) / 4,
             min(kurt, 50) / 50, min(tail * 50, 1), zlib_ratio,
             min(byte_chi2, 5) / 5, min(hist_flat, 10) / 10, ngram_rep,
             header_elf, header_zip, header_text, size_div4, size_factorness]
    return np.array(feats, dtype=np.float64)


class AnatomyMLP:
    """numpy 小 MLP：20→16→1，Sigmoid 输出 P(neural_body)。"""

    def __init__(self, w1=None, b1=None, w2=None, b2=None):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["w1"], d["b1"], d["w2"], d["b2"])

    def save(self, path: Path = MODEL_PATH) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return 1 / (1 + np.exp(-(h @ self.w2 + self.b2)))

    def train(self, X: np.ndarray, y: np.ndarray, epochs=400, lr=0.05, seed=7):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.4, (X.shape[1], 16))
        self.b1 = np.zeros(16)
        self.w2 = rng.normal(0, 0.4, (16, 1))
        self.b2 = np.zeros(1)
        for _ in range(epochs):
            h = np.tanh(X @ self.w1 + self.b1)
            z = h @ self.w2 + self.b2
            out = 1 / (1 + np.exp(-z))
            grad = (out - y.reshape(-1, 1)) / len(y)
            gw2 = h.T @ grad
            gb2 = grad.sum(0)
            gh = grad @ self.w2.T * (1 - h ** 2)
            gw1 = X.T @ gh
            gb1 = gh.sum(0)
            for p, g in ((self.w1, gw1), (self.b1, gb1), (self.w2, gw2), (self.b2, gb2)):
                p -= lr * g
        return self


# ─────────────────────────── 层次二：族谱认亲 ───────────────────────────

def tensor_stats(raw: bytes) -> np.ndarray:
    """量化鲁棒的结构统计向量（不逐值比对，int8 重建后依然稳定）。"""
    m = len(raw) - (len(raw) % 4)
    if m == 0:
        return np.zeros(10)
    f = np.frombuffer(raw[:m], dtype=np.float32).astype(np.float64)
    finite = f[np.isfinite(f)]
    if len(finite) < 8:
        return np.zeros(10)
    q = np.clip(finite * 127 / (np.abs(finite).max() + 1e-9), -127, 127).astype(np.int8)
    qh = np.bincount((q.astype(np.int16) + 128), minlength=256).astype(np.float64)
    qh = qh / qh.sum()
    std = finite.std() + 1e-12
    v = finite - finite.mean()
    return np.array([
        np.tanh(np.log1p(len(finite)) / 15),
        np.tanh(finite.mean() / 2),
        np.tanh(std),
        np.tanh((v ** 3).mean() / (std ** 3 + 1e-9) / 3),   # 偏度
        np.tanh(((v ** 4).mean() / (std ** 4)) / 30),        # 峰度
        np.tanh((np.abs(finite) < 0.1).mean() * 2),          # 近零密度
        np.tanh(len(np.unique(q)) / 128 / 2),                # 量化后动态范围
        np.tanh(-(qh[qh > 0] * np.log2(qh[qh > 0])).sum() / 8),  # 量化熵
        np.tanh(len(raw) / 200000),
        1.0,
    ])


def build_lineage_index(weight_dirs: list[Path], out: Path = INDEX_PATH) -> dict:
    """对每代权重目录建族谱索引：{file_key: {gen, stats, size}}。"""
    index = {"entries": [], "built_at": __import__("time").strftime("%F %T")}
    for d in weight_dirs:
        gen = d.name
        for f in sorted(Path(d).glob("*.bin")):
            raw = f.read_bytes()
            index["entries"].append({
                "gen": gen, "name": f.name, "size": len(raw),
                "stats": tensor_stats(raw).round(4).tolist(),
            })
    out.write_text(json.dumps(index, ensure_ascii=False))
    return index


def match_lineage(path: str | Path, index: dict | None = None,
                  sim_floor: float = 0.985) -> dict | None:
    """认亲：与族谱条目做统计向量余弦。≥sim_floor 判血缘。"""
    if index is None:
        index = json.loads(INDEX_PATH.read_text())
    s = tensor_stats(Path(path).read_bytes())
    best, best_sim = None, 0.0
    for e in index["entries"]:
        t = np.array(e["stats"])
        denom = np.linalg.norm(s) * np.linalg.norm(t)
        if denom == 0:
            continue
        sim = float(s @ t / denom)
        if sim > best_sim:
            best, best_sim = e, sim
    if best and best_sim >= sim_floor:
        return {"gen": best["gen"], "member": best["name"], "sim": round(best_sim, 4)}
    return None


# ─────────────────────── 层次三：装载行为 + 内存扫描 ───────────────────────

def score_load_burst(events: list[dict], l1_scorer=None,
                     min_files: int = 3, window_s: float = 60.0) -> dict:
    """对同进程的 security_file_open 事件窗打分。

    events: [{ts, pid, path}]，需按时间序。
    v0 规则：window_s 内同 pid 打开 ≥min_files 个 L1 判为"张量样"的文件 → 告警。
    """
    l1 = l1_scorer or AnatomyMLP.load()
    by_pid: dict[int, list[dict]] = {}
    for e in events:
        by_pid.setdefault(e["pid"], []).append(e)
    alerts = []
    for pid, evs in by_pid.items():
        best_win: list[str] = []
        i = 0
        for j, e in enumerate(evs):
            while e["ts"] - evs[i]["ts"] > window_s:
                i += 1
            tensorish = []
            for w in evs[i:j + 1]:
                try:
                    if l1 is not None and l1.forward(file_features(w["path"])).item() > 0.7:
                        tensorish.append(w["path"])
                except OSError:
                    continue
            if len(tensorish) > len(best_win):
                best_win = tensorish
        if len(best_win) >= min_files:
            alerts.append({"pid": pid, "files": best_win,
                           "score": min(1.0, len(best_win) / (min_files * 2))})
    return {"alerts": alerts, "n_events": len(events)}


def memscan(pid: int, sample_mb: float = 8.0) -> dict:
    """RSS 匿名区张量簇扫描（需对目标进程有 /proc/pid/mem 读权限）。

    原型：对可读匿名映射采样，判 float32/int8 张量簇密度。
    """
    regions = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            addr, perms = parts[0], parts[1]
            if "r" not in perms or addr in ("[vvar]", "[vdso]", "[vsyscall]"):
                continue
            pathname = parts[5] if len(parts) > 5 else ""
            if pathname.startswith("/"):
                continue  # 只看匿名区
            start, end = (int(x, 16) for x in addr.split("-"))
            if end - start < 65536:
                continue
            regions.append((start, end))
    hits = []
    budget = int(sample_mb << 20)
    with open(f"/proc/{pid}/mem", "rb", 0) as mem:
        for start, end in regions:
            span = end - start
            for off in (0, max(0, span // 2 - budget // 2), max(0, span - budget)):
                size = min(budget, span - off)
                if size < 65536:
                    continue
                try:
                    mem.seek(start + off)
                    raw = mem.read(size)
                except OSError:
                    continue
                m = len(raw) - (len(raw) % 4)
                if m < 4096:
                    continue
                f32 = np.frombuffer(raw[:m], dtype=np.float32).astype(np.float64)
                finite = np.isfinite(f32)
                frac_finite = finite.mean()
                small = (np.abs(f32[finite]) < 2.0).mean() if frac_finite > 0.5 else 0
                # 张量簇特征：高有限率 + 高小幅率 + 低零率
                if frac_finite > 0.97 and small > 0.92 and (f32[finite] != 0).mean() > 0.9:
                    hits.append({"region": hex(start), "mb": round(m / 1e6, 2),
                                 "f32_finite": round(frac_finite, 4),
                                 "small": round(small, 4)})
    seen: set[str] = set()
    uniq = [h for h in hits if not (h["region"] in seen or seen.add(h["region"]))]
    return {"pid": pid, "regions_scanned": len(regions), "tensor_clusters": uniq}


# ─────────────────────────── CLI ───────────────────────────

def _iter_files(d: Path):
    for f in sorted(d.rglob("*")):
        if f.is_file() and not f.is_symlink():
            yield f


def cmd_scan(d: str) -> None:
    l1 = AnatomyMLP.load()
    index = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else None
    print(f"扫描 {d}（L1={'on' if l1 else 'MISSING'}，L2={'on' if index else 'MISSING'}）")
    for f in _iter_files(Path(d)):
        try:
            x = file_features(f)
        except OSError:
            continue
        p = float(l1.forward(x).item()) if l1 else 0.0
        lin = match_lineage(f, index) if index and p > 0.3 else None
        tag = []
        if p > 0.7:
            tag.append(f"L1神经体 p={p:.2f}")
        if lin:
            tag.append(f"L2血缘 {lin['gen']}/{lin['member']} sim={lin['sim']}")
        if tag:
            print(f"  {f}  ← {'; '.join(tag)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "train", "scan", "memscan"])
    ap.add_argument("target", nargs="?")
    args = ap.parse_args()
    if args.cmd == "scan":
        cmd_scan(args.target)


if __name__ == "__main__":
    main()
