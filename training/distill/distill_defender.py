"""defender 蒸馏：TinyGPT (0.85M) → 端上小 MLP（三档压力预警）。

流程（ARCHITECTURE.md §3 阶段⑤）：
  1. 加载 TinyGPT prior.pt，对真实 token 流逐事件算 surprise（事件内 token NLL 取 max）
  2. 特征：[历史 token 直方图 ‖ 候选事件 token 直方图]（2×V 维）
  3. MLP(2V→128→64→1) 回归 surprise
  4. 门4：留出集上 Kendall's τ ≥ 0.8（蒸馏版 vs TinyGPT 的危险度排序）
  5. 导出 bee 权重格式（fp32 同字节）

用法:
  python3 -m training.distill.distill_defender \
      --data ~/defense-lab/detector/data/tokens_v2.jsonl checkpoints/clone_events.jsonl \
      --prior ~/defense-lab/detector/model/prior.pt \
      --out checkpoints/defender_distilled.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("distill")

sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
from train_prior import TinyGPT, CTX  # noqa: E402


class DefenderMLP(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_events(paths: list[str], stoi: dict) -> list[list[int]]:
    """加载事件流 → token id 序列（用 prior 的 stoi，不是编码器词表）。"""
    unk = stoi.get("<UNK>", 0)
    seq = []
    for p in paths:
        for line in open(p):
            obj = json.loads(line)
            seq.extend(stoi.get(t, unk) for t in obj["tokens"])
    return seq


def tinygpt_surprise(model: TinyGPT, seq: list[int], ev_start: int,
                     event_len: int = 7) -> float:
    """候选事件 seq[ev_start:ev_start+event_len] 的 surprise（事件内 NLL 取 max）。"""
    s = max(0, ev_start + event_len - CTX)
    end = min(len(seq), ev_start + event_len)
    x = torch.tensor([seq[s:end]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0]  # (L, V)
    nlls = []
    for m in range(ev_start, end):
        j = m - 1 - s
        if j < 0:
            continue
        nlls.append(F.cross_entropy(logits[j:j + 1],
                                    torch.tensor([seq[m]])).item())
    return max(nlls) if nlls else 0.0


def build_dataset(model: TinyGPT, seq: list[int], vocab: int,
                  n_events: int, event_len: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """(历史直方图‖事件直方图) → surprise 数据集。"""
    X, y = [], []
    n_tokens = n_events * event_len
    stride = event_len * 3
    for ev_start in range(CTX, min(len(seq) - event_len, n_tokens), stride):
        hist = np.zeros(vocab, dtype=np.float32)
        for t in seq[ev_start - CTX:ev_start]:
            hist[t] += 1
        hist /= CTX
        ev = np.zeros(vocab, dtype=np.float32)
        for t in seq[ev_start:ev_start + event_len]:
            ev[t] += 1
        ev /= event_len
        s = tinygpt_surprise(model, seq, ev_start)
        X.append(np.concatenate([hist, ev]))
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default="checkpoints/defender_distilled.pt")
    ap.add_argument("--n-events", type=int, default=3000)
    args = ap.parse_args()

    ckpt = torch.load(args.prior, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi))
    model.load_state_dict(ckpt["model"])
    model.eval()
    V = len(stoi)
    logger.info(f"TinyGPT 加载: vocab={V} p995={ckpt['baseline_nll']['p995']:.3f}")

    seq = load_events(args.data, stoi)
    logger.info(f"token 总量={len(seq)}")
    X, y = build_dataset(model, seq, V, args.n_events)
    logger.info(f"数据集: {X.shape} surprise 分布 p50={np.median(y):.2f} p95={np.percentile(y, 95):.2f}")

    idx = np.random.RandomState(42).permutation(len(y))
    cut = int(len(y) * 0.8)
    Xt = torch.from_numpy(X[idx[:cut]]); yt = torch.from_numpy(y[idx[:cut]])
    Xv = torch.from_numpy(X[idx[cut:]]); yv = y[idx[cut:]]

    mlp = DefenderMLP(V)
    logger.info(f"蒸馏 MLP 参数量={sum(p.numel() for p in mlp.parameters())}")
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for epoch in range(30):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = F.mse_loss(mlp(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"epoch {epoch+1}/30 mse={tot:.4f}")

    # 门4：Kendall's τ（留出集排序一致性）
    from scipy.stats import kendalltau
    with torch.no_grad():
        pred = mlp(Xv).numpy()
    tau, p = kendalltau(pred, yv)
    logger.info(f"门4: Kendall's τ={tau:.3f} (p={p:.1e}) 门=0.8")
    passed = tau >= 0.8
    logger.info(f"门4: {'PASS' if passed else 'FAIL → 高危动作附近加密样本重蒸'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mlp_state": mlp.state_dict(), "stoi": stoi,
                "tau": float(tau), "p995": ckpt["baseline_nll"]["p995"],
                "gate4_passed": passed}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
