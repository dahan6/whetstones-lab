"""defender 蒸馏（GRU 版）：带上下文的序列模型逼近 TinyGPT。

bigram 的失败证据（2026-07-30）：逐位置 τ 最好仅 0.678（ARGV），
PARENT/UID/DT 位置 τ<0.15——TinyGPT 的打分优势在 128-token 上下文，
无上下文的 n-gram 路线封顶 ~0.55。GRU 以 ~30K 参数换取上下文窗口，
部署端仍是简单前向（每 token 6 次矩阵乘，Rust 手写 ~200 行）。

蒸馏目标：TinyGPT 的 token 级 NLL（MSE 回归）。
门4 判据：留出事件级 max-NLL 打分与 TinyGPT 的 Kendall's τ ≥ 0.8。

用法:
  python3 -m training.distill.distill_gru \
      --data /tmp/merged_tokens.jsonl --prior /tmp/prior_v2/prior.pt \
      --out checkpoints/defender_gru.pt
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

logger = logging.getLogger("distill.gru")

EVENT_LEN = 7


import sys as _sys
_sys.path.insert(0, "/home/lab/adaptive-agent-sim")
from training.distill.defender_scan import FlexTransformer


class TransformerStudent(nn.Module):
    def __init__(self, vocab, d_model=128, ctx=128, layers=4):
        super().__init__()
        self.base = FlexTransformer(vocab, d_model, ctx, layers)
        self.reg = nn.Linear(d_model, 1)
    def forward(self, x):
        T = x.size(1)
        h = self.base.tok(x) + self.base.pos.weight[:T]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.base.block(h, mask=mask)
        return self.reg(self.base.norm(h)).squeeze(-1)   # (B,T) 逐位置NLL回归


class GRUStudent(nn.Module):
    """token → embed → GRU → 每位置 NLL 回归。"""

    def __init__(self, vocab: int, d_emb: int = 64, d_hid: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_emb)
        self.gru = nn.GRU(d_emb, d_hid, batch_first=True)
        self.head = nn.Linear(d_hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.embed(x))
        return self.head(h).squeeze(-1)   # (B, T)


@torch.no_grad()
def tinygpt_nll(model: TinyGPT, seq: list[int], batch: int = 256) -> np.ndarray:
    """全序列逐 token NLL：无重叠 CTX 窗，logits[j] 预测 seq[base+j+1]，
    窗间无缝衔接（每窗最后一个输出预测下一窗首 token）。"""
    V = model.head.out_features
    n = (len(seq) - 1) // CTX
    out = np.zeros(len(seq), dtype=np.float32)
    for b in range(0, n, batch):
        ids = list(range(b, min(b + batch, n)))
        x = torch.tensor([seq[i * CTX: i * CTX + CTX] for i in ids])
        y = torch.tensor([seq[i * CTX + 1: i * CTX + CTX + 1] for i in ids])
        logits = model(x)
        lp = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                             reduction="none").view(len(ids), CTX)
        for k, i in enumerate(ids):
            out[i * CTX + 1: i * CTX + CTX + 1] = lp[k].numpy()
        if b % (batch * 10) == 0:
            logger.info(f"  tinygpt 打分 {b * CTX}/{len(seq)}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default="checkpoints/defender_gru.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n-events", type=int, default=2000)
    args = ap.parse_args()

    ckpt = torch.load(args.prior, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    V = len(stoi)
    teacher = TinyGPT(V)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()

    seq = []
    for p in args.data:
        for line in open(p):
            seq.extend(stoi.get(t, 0) for t in json.loads(line)["tokens"])
    logger.info(f"token 总量={len(seq)} vocab={V}")

    # 教学信号：TinyGPT 逐 token NLL
    t0 = time.time()
    nll = tinygpt_nll(teacher, seq)
    logger.info(f"TinyGPT 打分完成 ({time.time()-t0:.0f}s) "
                f"mean={nll[1:].mean():.3f} p99={np.quantile(nll[1:], 0.99):.3f}")

    cut = int(len(seq) * 0.8)
    student = TransformerStudent(V)
    logger.info(f"student 参数量={sum(p.numel() for p in student.parameters())}")
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)

    # token 级蒸馏：MSE 回归教师 NLL + 高分区 pairwise 保排序
    T = 128
    idx = np.arange(0, cut - T - 1, T // 2)
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        tot, nb = 0.0, 0
        for b in range(0, len(idx) - 63, 64):
            batch = idx[b:b + 64]
            x = torch.tensor([seq[i:i + T] for i in batch])
            y = torch.tensor([nll[i + 1:i + T + 1] for i in batch])
            pred = student(x)
            flat_p, flat_y = pred.reshape(-1), y.reshape(-1)
            hi = (flat_y > 0.5).nonzero().squeeze(-1)
            i1 = hi[torch.randint(0, len(hi), (256,))] if len(hi) else \
                torch.randint(0, flat_p.numel(), (256,))
            i2 = torch.randint(0, flat_p.numel(), (256,))
            sign = torch.sign(flat_y[i1] - flat_y[i2])
            rank_loss = F.softplus(-sign * (flat_p[i1] - flat_p[i2])).mean()
            loss = F.mse_loss(pred, y) + 0.5 * rank_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info(f"epoch {ep+1}/{args.epochs} loss={tot/max(1,nb):.4f} "
                    f"({time.time()-t0:.0f}s)")

    # 门4：留出集事件级 max-NLL 排序一致性
    student.eval()
    y_gpt, y_gru = [], []
    stride = EVENT_LEN * 3
    n = 0
    with torch.no_grad():
        for ev in range(cut + CTX, len(seq) - EVENT_LEN, stride):
            s = max(0, ev + EVENT_LEN - CTX)
            e = ev + EVENT_LEN
            xg = torch.tensor([seq[s:e]])
            lg = teacher(xg)[0]
            nlls_g = [F.cross_entropy(lg[m - 1 - s:m - s],
                                      torch.tensor([seq[m]])).item()
                      for m in range(ev, e) if m - 1 - s >= 0]
            y_gpt.append(max(nlls_g) if nlls_g else 0.0)
            pred = student(torch.tensor([seq[s:e]]))[0]
            vals = [pred[m - s].item() for m in range(ev, e)]
            y_gru.append(max(vals))
            n += 1
            if n >= args.n_events:
                break

    from scipy.stats import kendalltau
    tau, p = kendalltau(y_gru, y_gpt)
    logger.info(f"留出事件数={len(y_gpt)}")
    logger.info(f"门4: Kendall's τ={tau:.3f} (p={p:.1e}) 门=0.8")
    logger.info(f"门4: {'PASS' if tau >= 0.8 else 'FAIL'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), "stoi": stoi,
                "config": {"d_emb": 32, "d_hid": 64},
                "tau": tau}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
