"""小型 defender：1 层 transformer 直接训练 next-token，兼作端上打分器。

证据链（2026-07-30）：
  bigram τ=0.549（无上下文）→ GRU τ≤0.691（压缩记忆学不出注意力的
  "具体 token 回溯"）→ 教师自一致 ctx64 τ=0.897（0.8 可达，函数类是关键）
故学生必须是 transformer；ctx64/d64/1层 ≈ 70K 参数（280KB，预算内），
部署端单层注意力前向可手写（~300 行 Rust）。

用法:
  python3 -m training.distill.train_defender_small \
      --data /tmp/merged_tokens.jsonl --prior /tmp/prior_v2/prior.pt \
      --out checkpoints/defender_small.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
from train_prior import TinyGPT, CTX  # noqa: E402

logger = logging.getLogger("distill.small")

EVENT_LEN = 7
SCTX = 64      # 学生上下文
D_MODEL = 64


class SmallTransformer(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(SCTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=256,
            batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = self.tok(x) + self.pos.weight[:T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device),
                          diagonal=1)
        h = self.block(h, mask=mask)
        return self.head(self.norm(h))


def load_seq(paths: list[str], stoi: dict) -> list[int]:
    seq = []
    for p in paths:
        for line in open(p):
            seq.extend(stoi.get(t, 0) for t in json.loads(line)["tokens"])
    return seq


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default="checkpoints/defender_small.pt")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.prior, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    V = len(stoi)
    teacher = TinyGPT(V)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval().to(dev)

    seq = load_seq(args.data, stoi)
    cut = int(len(seq) * 0.8)
    logger.info(f"token={len(seq)} vocab={V} dev={dev}")

    student = SmallTransformer(V).to(dev)
    logger.info(f"student 参数量={sum(p.numel() for p in student.parameters())}")
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        starts = torch.randperm(cut - SCTX - 1)[:200000].tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(starts) - 255, 256):
            bs = starts[i:i + 256]
            x = torch.tensor([seq[s:s + SCTX] for s in bs]).to(dev)
            y = torch.tensor([seq[s + 1:s + SCTX + 1] for s in bs]).to(dev)
            loss = lossf(student(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info(f"epoch {ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
                    f"({time.time()-t0:.0f}s)")

    # 门4：事件级 max-NLL，学生 ctx64 vs 教师 ctx128
    student.eval()
    y_gpt, y_stu = [], []
    stride = EVENT_LEN * 3
    with torch.no_grad():
        n = 0
        for ev in range(cut + CTX, len(seq) - EVENT_LEN, stride):
            s1 = ev + EVENT_LEN - CTX
            x1 = torch.tensor([seq[s1:ev + EVENT_LEN]]).to(dev)
            lg = teacher(x1)[0]
            nlls = [F.cross_entropy(lg[m - 1 - s1:m - s1],
                                    torch.tensor([seq[m]]).to(dev)).item()
                    for m in range(ev, ev + EVENT_LEN)]
            y_gpt.append(max(nlls))
            s2 = ev + EVENT_LEN - SCTX
            x2 = torch.tensor([seq[s2:ev + EVENT_LEN]]).to(dev)
            ls = student(x2)[0]
            nlls2 = [F.cross_entropy(ls[m - 1 - s2:m - s2],
                                     torch.tensor([seq[m]]).to(dev)).item()
                     for m in range(ev, ev + EVENT_LEN)]
            y_stu.append(max(nlls2))
            n += 1
            if n >= 2000:
                break

    from scipy.stats import kendalltau
    tau, p = kendalltau(y_stu, y_gpt)
    logger.info(f"留出事件数={len(y_gpt)}")
    logger.info(f"门4: Kendall's τ={tau:.3f} (p={p:.1e}) 门=0.8")
    logger.info(f"门4: {'PASS' if tau >= 0.8 else 'FAIL'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), "stoi": stoi,
                "config": {"d_model": D_MODEL, "ctx": SCTX}, "tau": tau}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
