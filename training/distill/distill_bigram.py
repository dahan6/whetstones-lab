"""defender 蒸馏（bigram 版）：转移概率表逼近 TinyGPT。

动机：直方图特征丢了 token 顺序，而 surprise 本质是"异常转移"，
顺序正是关键。bigram 用 P(tok_t|tok_{t-1}) 转移表直接命中本质，
且部署端是**纯查表**——没有网络、没有推理、确定性、与 Python 天然同算术。

表大小: V×V fp32 ≈ 169² × 4B ≈ 114KB（预算内）
门4: Kendall's τ ≥ 0.8（留出集上 vs TinyGPT 的事件危险度排序）

用法:
  python3 -m training.distill.distill_bigram \
      --data ~/defense-lab/detector/data/tokens_v2.jsonl checkpoints/clone_events.jsonl \
      --prior ~/defense-lab/detector/model/prior.pt \
      --out checkpoints/defender_bigram.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.home() / "defense-lab" / "detector"))
from train_prior import TinyGPT, CTX  # noqa: E402

logger = logging.getLogger("distill.bigram")

EVENT_LEN = 7
ALPHA = 0.1   # 加性平滑


def load_seq(paths: list[str], stoi: dict) -> list[int]:
    unk = stoi.get("<UNK>", 0)
    seq = []
    for p in paths:
        for line in open(p):
            obj = json.loads(line)
            seq.extend(stoi.get(t, unk) for t in obj["tokens"])
    return seq


def tinygpt_event_surprise(model: TinyGPT, seq: list[int], ev_start: int) -> float:
    s = max(0, ev_start + EVENT_LEN - CTX)
    end = min(len(seq), ev_start + EVENT_LEN)
    x = torch.tensor([seq[s:end]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0]
    nlls = []
    for m in range(ev_start, end):
        j = m - 1 - s
        if j < 0:
            continue
        nlls.append(F.cross_entropy(logits[j:j + 1],
                                    torch.tensor([seq[m]])).item())
    return max(nlls) if nlls else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default="checkpoints/defender_bigram.npz")
    ap.add_argument("--n-events", type=int, default=2000)
    args = ap.parse_args()

    ckpt = torch.load(args.prior, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi))
    model.load_state_dict(ckpt["model"])
    model.eval()
    V = len(stoi)
    seq = load_seq(args.data, stoi)
    logger.info(f"token 总量={len(seq)} vocab={V}")

    # bigram 计数（前 80% 做训练，后 20% 做留出评估）
    cut = int(len(seq) * 0.8)
    counts = np.full((V, V), ALPHA, dtype=np.float64)
    for a, b in zip(seq[:cut - 1], seq[1:cut]):
        counts[a, b] += 1
    logp = np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)

    # 留出集事件：bigram 分 vs TinyGPT 分
    stride = EVENT_LEN * 3
    y_gpt, y_bi = [], []
    n = 0
    for ev_start in range(cut + CTX, len(seq) - EVENT_LEN, stride):
        y_gpt.append(tinygpt_event_surprise(model, seq, ev_start))
        end = min(len(seq), ev_start + EVENT_LEN)
        scores = [-logp[seq[m - 1], seq[m]] for m in range(ev_start, end)]
        y_bi.append(max(scores) if scores else 0.0)
        n += 1
        if n >= args.n_events:
            break

    from scipy.stats import kendalltau
    tau, p = kendalltau(y_bi, y_gpt)
    logger.info(f"留出事件数={len(y_gpt)}")
    logger.info(f"门4: Kendall's τ={tau:.3f} (p={p:.1e}) 门=0.8")
    passed = tau >= 0.8
    logger.info(f"门4: {'PASS' if passed else 'FAIL'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, logp=logp.astype(np.float32),
             stoi=json.dumps(stoi), tau=float(tau))
    logger.info(f"保存 → {args.out}（表 {logp.nbytes/1024:.0f}KB）")


if __name__ == "__main__":
    main()
