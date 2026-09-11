"""编码器预训练：真实 Tracee 流上的下一事件预测。

用法:
  python3 -m training.encoder.pretrain \
      --data ~/lado-range/detector/data/tokens_v2.jsonl \
      --out checkpoints/encoder.pt
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.encoder.data import build_samples
from training.encoder.model import PretrainModel

logger = logging.getLogger("encoder.pretrain")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="checkpoints/encoder.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    ds, samples = build_samples(args.data)
    logger.info(f"事件={len(ds.events)} 窗口={len(samples)} 词表={ds.vocab_size}")

    X = torch.from_numpy(np.stack([s["input"] for s in samples]))
    Y = torch.from_numpy(np.stack([s["target"] for s in samples]))

    model = PretrainModel(ds.vocab_size)
    n_params = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"编码器参数量={n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(samples)
    idx = np.arange(n)

    for epoch in range(args.epochs):
        t0 = time.time()
        np.random.shuffle(idx)
        tot_loss, tot_acc, nb = 0.0, 0.0, 0
        for i in range(0, n, args.batch):
            b = idx[i:i + args.batch]
            xb, yb = X[b], Y[b]
            logits = model(xb)
            loss = sum(F.cross_entropy(lg, yb[:, k])
                       for k, lg in enumerate(logits)) / len(logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc = sum((lg.argmax(-1) == yb[:, k]).float().mean().item()
                          for k, lg in enumerate(logits)) / len(logits)
            tot_loss += loss.item()
            tot_acc += acc
            nb += 1
        logger.info(f"epoch {epoch+1}/{args.epochs}  loss={tot_loss/nb:.4f}  "
                    f"next-token-acc={tot_acc/nb:.3f}  ({time.time()-t0:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "vocab": ds.vocab,
        "config": {"window": 16, "tok_dim": 16, "event_dim": 48, "state_dim": 48},
    }, args.out)
    logger.info(f"保存 → {args.out}（仅编码器权重 + 词表，预训练头已弃）")


if __name__ == "__main__":
    main()
