"""defender 容量扫描：τ-容量曲线，定位 τ≥0.8 的最小容量点。

用法:
  python3 -m training.distill.defender_scan --layers 3 --d-model 128 --ctx 128 \
      --data /tmp/merged_tokens.jsonl --prior /tmp/prior_v2/prior.pt \
      --out /tmp/scan_L.pt --epochs 4
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

sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
from train_prior import TinyGPT, CTX  # noqa: E402

logger = logging.getLogger("distill.scan")
EVENT_LEN = 7


class FlexTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int, ctx: int, layers: int,
                 heads: int = 4):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = self.tok(x) + self.pos.weight[:T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device),
                          diagonal=1)
        h = self.block(h, mask=mask)
        return self.head(self.norm(h))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.prior, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    V = len(stoi)
    teacher = TinyGPT(V)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval().to(dev)

    seq = []
    for p in args.data:
        for line in open(p):
            seq.extend(stoi.get(t, 0) for t in json.loads(line)["tokens"])
    cut = int(len(seq) * 0.8)

    student = FlexTransformer(V, args.d_model, args.ctx, args.layers).to(dev)
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(f"student L{args.layers} d{args.d_model} ctx{args.ctx} "
                f"参数量={n_params} ({n_params*4/1e6:.2f}MB fp32 / "
                f"{n_params/1e6:.2f}MB int8) dev={dev}")
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        starts = torch.randperm(cut - args.ctx - 1)[:200000].tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(starts) - 255, 256):
            bs = starts[i:i + 256]
            x = torch.tensor([seq[s:s + args.ctx] for s in bs]).to(dev)
            y = torch.tensor([seq[s + 1:s + args.ctx + 1] for s in bs]).to(dev)
            loss = lossf(student(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info(f"epoch {ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
                    f"({time.time()-t0:.0f}s)")

    student.eval()
    y_gpt, y_stu = [], []
    with torch.no_grad():
        n = 0
        for ev in range(cut + CTX, len(seq) - EVENT_LEN, EVENT_LEN * 3):
            s1 = ev + EVENT_LEN - CTX
            lg = teacher(torch.tensor([seq[s1:ev + EVENT_LEN]]).to(dev))[0]
            y_gpt.append(max(F.cross_entropy(
                lg[m - 1 - s1:m - s1], torch.tensor([seq[m]]).to(dev)).item()
                for m in range(ev, ev + EVENT_LEN)))
            s2 = ev + EVENT_LEN - args.ctx
            ls = student(torch.tensor([seq[s2:ev + EVENT_LEN]]).to(dev))[0]
            y_stu.append(max(F.cross_entropy(
                ls[m - 1 - s2:m - s2], torch.tensor([seq[m]]).to(dev)).item()
                for m in range(ev, ev + EVENT_LEN)))
            n += 1
            if n >= 2000:
                break

    from scipy.stats import kendalltau
    tau, p = kendalltau(y_stu, y_gpt)
    logger.info(f"RESULT L{args.layers}d{args.d_model}c{args.ctx} "
                f"params={n_params} tau={tau:.3f}")
    torch.save({"student": student.state_dict(), "stoi": stoi,
                "config": {"layers": args.layers, "d_model": args.d_model,
                           "ctx": args.ctx},
                "tau": tau, "n_params": n_params}, args.out)


if __name__ == "__main__":
    main()
