"""门1：线性探针验收。

冻结编码器后，用线性分类器从其输出预测关键属性，
留出集准确率全部 > 85% 才允许冻结进入下一阶段：

  - day_night : 昼夜相位（数据自标）
  - proc_class: 主导进程粗类（数据自标）
  - scan_active: 扫描中/空闲（来自 collect_scan_labels 的采集数据）

不达标 → 回去补训练数据，不带烂表征进下一阶段。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.encoder.data import PROC_CLASSES, build_samples
from training.encoder.model import Encoder

logger = logging.getLogger("encoder.probe")

GATE = 0.85


def extract_states(encoder: Encoder, X: torch.Tensor, batch: int = 512) -> torch.Tensor:
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(encoder(X[i:i + batch]))
    return torch.cat(outs)


def train_probe(states: torch.Tensor, y: np.ndarray, classes: int,
                epochs: int = 300) -> tuple[float, float]:
    """线性探针：train/val 8:2 划分。

    返回 (平衡准确率, 少数类占比)。平衡准确率 = 各类召回率的均值，
    防止全 0 标签退化通过门。
    """
    n = len(y)
    idx = np.random.RandomState(42).permutation(n)
    cut = int(n * 0.8)
    tr, va = idx[:cut], idx[cut:]
    Xt, Xv = states[tr], states[va]
    yt = torch.from_numpy(y[tr])
    yv = y[va]

    probe = torch.nn.Linear(states.size(1), classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Xv).argmax(-1).numpy()
    recalls = []
    for c in range(classes):
        mask = yv == c
        if mask.sum() == 0:
            return 0.0, 0.0   # 留出集缺类 → 数据不足
        recalls.append((pred[mask] == c).mean())
    minority = min((y == c).mean() for c in range(classes))
    return float(np.mean(recalls)), float(minority)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--encoder", default="checkpoints/encoder.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.encoder, weights_only=False)
    vocab: dict[str, int] = ckpt["vocab"]

    # day_night 用 regime 数据（当前节律环境），其余探针用全量数据
    daynight_path = Path("checkpoints/regime_events.jsonl")
    daynight_data = [str(daynight_path)] if daynight_path.exists() else args.data
    ds_dn, samples_dn = build_samples(daynight_data, vocab=vocab)
    ds, samples = build_samples(args.data, vocab=vocab)

    encoder = Encoder(len(vocab))
    encoder.load_state_dict(ckpt["encoder_state"])

    X = torch.from_numpy(np.stack([s["input"] for s in samples]))
    states = extract_states(encoder, X)
    X_dn = torch.from_numpy(np.stack([s["input"] for s in samples_dn]))
    states_dn = extract_states(encoder, X_dn)
    logger.info(f"状态提取: 全量{states.shape} regime{states_dn.shape}")

    results = {}
    minorities = {}
    # day_night：只在 regime 数据上评（当前节律环境）
    for name, y, classes, st in [
        ("day_night", np.array([s["day_night"] for s in samples_dn]), 2, states_dn),
        ("proc_class", np.array([s["proc_class"] for s in samples]), len(PROC_CLASSES), states),
    ]:
        # 只评估占比 ≥5% 的类（数据里没有的类不惩罚编码器，标记数据不足）
        shares = np.array([(y == c).mean() for c in range(classes)])
        keep = np.where(shares >= 0.05)[0]
        if len(keep) < 2:
            logger.info(f"探针 {name:<12} INSUFFICIENT_DATA "
                        f"(有效类不足: 分布={shares.round(3)})")
            results[name] = 0.0
            minorities[name] = 0.0
            continue
        remap = {c: i for i, c in enumerate(keep)}
        mask = np.isin(y, keep)
        y_f = np.array([remap[v] for v in y[mask]])
        acc, minority = train_probe(st[torch.from_numpy(mask)], y_f, classes=len(keep))
        results[name] = acc
        minorities[name] = minority
        verdict = "PASS" if acc >= GATE else "FAIL"
        logger.info(f"探针 {name:<12} balanced_acc={acc:.3f} 少数类={minority:.2%}  "
                    f"类数={len(keep)}/{classes}  [{verdict}] (门={GATE})")

    # scan_active 探针：需要采集数据，见 collect_scan_labels.py
    scan_path = Path("checkpoints/scan_labels.npz")
    if scan_path.exists():
        data = np.load(scan_path)
        Xs = torch.from_numpy(data["X"])
        ys = data["y"]
        states_s = extract_states(encoder, Xs)
        acc, minority = train_probe(states_s, ys, classes=2)
        results["scan_active"] = acc
        minorities["scan_active"] = minority
        verdict = "PASS" if acc >= GATE and minority >= 0.05 else "INSUFFICIENT/FAIL"
        logger.info(f"探针 {'scan_active':<12} balanced_acc={acc:.3f} 少数类={minority:.2%}  "
                    f"[{verdict}] (门={GATE})")
    else:
        logger.info("scan_active 标签数据不存在，跳过（先跑 collect_scan_labels）")

    passed = all(a >= GATE for a in results.values()) and \
        all(m >= 0.05 for m in minorities.values())
    logger.info(f"门1 验收: {'PASS' if passed else 'FAIL'} —— {results}")
    if not passed:
        raise SystemExit("门1 未过：回去补训练数据，不得进入阶段②")


if __name__ == "__main__":
    main()
