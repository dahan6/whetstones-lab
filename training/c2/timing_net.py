"""报联时机网络（修饰器）+ ES 训练器。

时机网络：小 MLP，输入[通道可检性,距上次报联,任务pending,紧急度,NDR近期分,
当前CV,随机种子] → 输出下次报联间隔[5,300]s。
训练：ES（演化策略）在模拟 C2 任务流上，NDR 模拟器打分。
reward = 及时送达奖励 − NDR 异常惩罚 − 报联成本。

用法:
  python3 -m training.c2.timing_net --gens 20 --pop 16 --out checkpoints/timing_net.pt
  （评估对比：学习的时机 vs 固定30s轮询，NDR 分越低越好）
"""
from __future__ import annotations

import argparse
import logging
import math
import statistics

import numpy as np
import torch
import torch.nn as nn

from training.ndr.simulator import ConnRecord, NdrSimulator

logger = logging.getLogger("c2.timing")

CHANNELS = [("http", 8443, 0.7), ("dns", 8053, 0.3)]   # (名,端口,可检性)
FEAT = 8


class TimingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))
        # 输出 bias=-2 → 初始 sigmoid(-2)=0.12 → interval~40s 中频出发
        # （0 时 152s 低频出发，ES 爬不出低及时率局部最优）
        with torch.no_grad():
            self.net[-1].bias.fill_(-2.0)
        # sigmoid 映射到 [5,300]，初始 sigmoid(0)=0.5 → ~152s 中段，
        # 避免 Softplus(0)→clamp 下限把所有候选钉死在 5s（ES 无信号）

    def interval(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            v = torch.sigmoid(self.net(feat)).item()
        return 5.0 + 295.0 * v


def simulate_episode(net: TimingNet, ndr: NdrSimulator,
                     seed: int, steps: int = 40) -> dict:
    """模拟一个 C2 episode：任务随机注入，bee 按时机网络报联，NDR 打分。
    返回 reward 与指标。"""
    rng = np.random.default_rng(seed)
    t = 0.0
    next_beacon = 0.0
    pending: list[tuple[float, str]] = []     # (注入时刻, 任务)
    delivered = 0
    total_wait = 0.0
    conns: list[ConnRecord] = []
    ndr_recent = 0.0
    n_beacons = 0

    # 任务计划：steps 内随机注 3-6 个
    plan = sorted((float(rng.uniform(0, steps * 8)), ch)
                  for ch in rng.choice(["http", "dns"],
                                       size=int(rng.integers(3, 7))))

    for _ in range(steps):
        t += 8.0   # 每 step 8s
        # 任务注入
        while plan and plan[0][0] <= t:
            _, ch = plan.pop(0)
            pending.append((t, ch))
        # 到报联时刻
        if t >= next_beacon:
            if pending:
                inj, ch = pending.pop(0)
                delivered += 1
                total_wait += t - inj
            else:
                ch = CHANNELS[int(rng.integers(0, 2))][0]
            dport = 8443 if ch == "http" else 8053
            r = ConnRecord(t, "192.0.2.119", "192.0.2.1", dport, ch)
            conns.append(r)
            n_beacons += 1
            # NDR 对这条打分
            hist = [c for c in conns if c.dport == dport]
            ndr_recent = ndr.score_connection(r, hist)
            # 时机网络决定下次间隔
            ivs = [b.ts - a.ts for a, b in zip(conns[-7:-1], conns[-6:])]
            cv = (statistics.pstdev(ivs) / statistics.mean(ivs)
                  if len(ivs) >= 4 and statistics.mean(ivs) > 0 else 1.0)
            feat = torch.tensor([
                0.7 if ch == "http" else 0.3,
                min((t - conns[-2].ts) / 300.0, 1.0) if len(conns) > 1 else 0.0,
                1.0 if pending else 0.0,
                min(len(pending) / 3.0, 1.0),
                ndr_recent,
                min(cv, 1.0),
                float(rng.random()),
                t / (steps * 8.0),
            ], dtype=torch.float32)
            gap = net.interval(feat)
            next_beacon = t + gap

    # reward：送达率、平均等待、NDR 异常、报联成本
    n_tasks = delivered + len(pending)
    timely = delivered / max(n_tasks, 1)
    avg_wait = total_wait / max(delivered, 1)
    win = ndr.score_window(conns)
    # reward 重平衡（旧版 25/0.2 时学习时机选择高频报联拿及时率满分，
    # 周期惩罚和报联成本压不住）：NDR 权重 60 让周期差异主导，报联 0.5 逼低频
    reward = (25.0 * timely
              - 0.05 * avg_wait
              - 60.0 * win["ndr_score"]
              - 0.5 * n_beacons)
    return {"reward": reward, "timely": timely, "avg_wait": avg_wait,
            "ndr": win["ndr_score"], "beacons": n_beacons, "delivered": delivered}


def fixed_baseline(ndr: NdrSimulator, seed: int, gap: float = 30.0,
                   steps: int = 40) -> dict:
    """固定轮询基线（gap 秒）。"""
    class Fixed(TimingNet):
        def interval(self, feat): return gap
    return simulate_episode(Fixed(), ndr, seed, steps)


def get_vec(m: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().numpy().ravel() for p in m.parameters()])


def set_vec(m: nn.Module, v: np.ndarray) -> None:
    i = 0
    for p in m.parameters():
        n = p.numel()
        p.data = torch.from_numpy(v[i:i + n].reshape(p.shape)).float()
        i += n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default="checkpoints/timing_net.pt")
    args = ap.parse_args()

    # NDR 基线：正常流量（系统已有连接）
    ndr = NdrSimulator()
    ndr.learn_baseline_from_records([
        ConnRecord(float(i), "192.0.2.119", "192.0.2.1", 53, "udp")
        for i in range(0, 400, 17)] + [
        ConnRecord(float(i), "192.0.2.119", "192.0.2.103", 22, "tcp")
        for i in range(0, 400, 61)])

    net = TimingNet()
    base = get_vec(net)
    dim = len(base)
    logger.info(f"时机网络参数量={dim}")
    rng = np.random.default_rng(42)

    for gen in range(args.gens):
        noise = rng.standard_normal((args.pop, dim)).astype(np.float32)
        fits = np.zeros(args.pop)
        for i in range(args.pop):
            set_vec(net, base + args.sigma * noise[i])
            fits[i] = np.mean([simulate_episode(net, ndr, 1000 + gen * 100 + j)["reward"]
                               for j in range(4)])
        ranks = np.argsort(np.argsort(fits))
        top = max(2, args.pop // 3)
        w = np.maximum(ranks - (args.pop - top), 0).astype(np.float32)
        w = w / w.sum() if w.sum() > 0 else np.ones(args.pop) / args.pop
        base = base + args.lr * args.sigma * (w[:, None] * noise).sum(axis=0)
        set_vec(net, base)
        ev = np.mean([simulate_episode(net, ndr, 5000 + j)["reward"] for j in range(6)])
        logger.info(f"gen {gen+1}/{args.gens} 适应度 avg={fits.mean():.2f} "
                    f"max={fits.max():.2f} eval={ev:.2f}")

    # 对比：学习的时机 vs 固定 30s
    learned = [simulate_episode(net, ndr, 9000 + j) for j in range(10)]
    fixed = [fixed_baseline(ndr, 9000 + j) for j in range(10)]
    for name, res in (("学习时机", learned), ("固定30s", fixed)):
        logger.info(
            f"{name}: reward={np.mean([r['reward'] for r in res]):.2f} "
            f"及时率={np.mean([r['timely'] for r in res]):.2f} "
            f"等待={np.mean([r['avg_wait'] for r in res]):.0f}s "
            f"NDR={np.mean([r['ndr'] for r in res]):.3f} "
            f"报联数={np.mean([r['beacons'] for r in res]):.0f}")

    torch.save({"net": net.state_dict(), "feat": FEAT}, args.out)
    logger.info(f"保存 → {args.out}")


if __name__ == "__main__":
    main()
