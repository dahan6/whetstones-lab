"""缺口3.2 重训脚本：用 feat_dim=12（含任务类型 one-hot）重训四技能头。

流程：
  1. 加载冻结编码器
  2. 生成 BC 语料（含任务类型标签——每个教学样本带对应 pending_type）
  3. BC warmstart → 保存
  4. 导出为 bee 部署格式

用法:
  python3 -m training.skills.retrain_feat12 --out bee/weights/

前置：靶场 VM（range-l2-t*）可用，编码器 checkpoint 存在。
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[2]))

from training.encoder.model import Encoder, STATE_DIM
from training.skills.heads import SkillHead, FEAT_DIM, get_vector, set_vector
from training.skills.state import make_feat
from training.skills.actions import (
    COMMUNICATE, CROSS_VM, KILL_CHAIN, LOCAL_COPY, MODIFY_CRON,
    MODIFY_FILE, MODIFY_SERVICE, PROBE, READ_SENSITIVE, SLEEP,
)
from training.skills.bc_data import gen_b_samples

logger = logging.getLogger("skills.retrain_feat12")

# 教学数据：每个动作 + 对应任务类型（缺口3.2：技能头任务感知）
# (action_id, pending_type, n_samples)
TEACHING_DATA = [
    # A 技能：维护动作（无任务 pending）
    (SLEEP,          None,         50),
    (PROBE,          None,         15),
    (READ_SENSITIVE, "read_files", 20),  # read_files 任务对齐
    (MODIFY_FILE,    None,         10),
    # C 技能：任务执行
    (PROBE,          None,         10),
    (READ_SENSITIVE, "read_files", 15),
    (COMMUNICATE,    "report",     20),  # report 任务对齐
    (MODIFY_CRON,    "persist",    15),  # persist 任务对齐
    (MODIFY_SERVICE, "persist",    10),
    # D 技能：传播
    (LOCAL_COPY,     "propagate",  10),
    (CROSS_VM,       "propagate",  10),
    (MODIFY_CRON,    "persist",    5),   # persist 也覆盖 D
    (KILL_CHAIN,     "propagate",  5),
]


def load_encoder(ckpt_path: str) -> Encoder:
    ckpt = torch.load(ckpt_path, weights_only=False)
    enc = Encoder(vocab_size=len(ckpt["vocab"]))
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()
    enc._vocab = ckpt["vocab"]
    return enc


def gen_synthetic_bc(encoder: Encoder, n_per_sample: int = 1) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成带任务类型标签的 BC 语料。

    用编码器处理随机 token 窗口产生 state，配合 make_feat(pending_type=...)
    产生 feat(12)。
    """
    rng = random.Random(42)
    vocab_size = len(encoder._vocab)
    skill_map = {
        # action → skill
        SLEEP: "A", PROBE: "A", READ_SENSITIVE: "A", MODIFY_FILE: "A",
        COMMUNICATE: "C", MODIFY_CRON: "C", MODIFY_SERVICE: "C",
        LOCAL_COPY: "D", CROSS_VM: "D", KILL_CHAIN: "D",
    }
    # C 技能也用 PROBE/READ_SENSITIVE
    skill_map_c_override = {PROBE: "C", READ_SENSITIVE: "C"}

    corp: dict[str, list[tuple[np.ndarray, int]]] = {"A": [], "C": [], "D": []}

    for action, pending_type, n in TEACHING_DATA:
        skill = skill_map.get(action, "C")
        # 如果有 pending_type，按任务重定向到 C（因为 C 是任务执行技能）
        if pending_type == "read_files":
            skill = skill_map_c_override.get(action, skill)
        if skill not in corp:
            continue

        for _ in range(n * n_per_sample):
            # 随机 token 窗口
            tokens = torch.randint(0, vocab_size, (1, 16, 7))
            with torch.no_grad():
                state = encoder(tokens).squeeze(0).numpy()
            feat = make_feat(
                alive_steps=rng.randint(0, 500),
                task_progress=rng.random() * 0.5,
                steps_since_action=rng.randint(0, 50),
                pending_type=pending_type,
            )
            x = np.concatenate([state, feat]).astype(np.float32)
            corp[skill].append((x, action))

    # 转换为 (X, y)
    result = {}
    for skill, pairs in corp.items():
        if not pairs:
            continue
        X = np.array([p[0] for p in pairs], dtype=np.float32)
        y = np.array([p[1] for p in pairs], dtype=np.int64)
        result[skill] = (X, y)
        logger.info(f"  {skill}: {len(y)} 样本, 动作分布={np.bincount(y, minlength=12).tolist()}")

    return result


def bc_train(head: SkillHead, X: np.ndarray, y: np.ndarray, epochs: int = 80) -> float:
    Xt = torch.from_numpy(X[:, :STATE_DIM])
    Xf = torch.from_numpy(X[:, STATE_DIM:])
    yt = torch.from_numpy(y)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for ep in range(epochs):
        logits = head(Xt, Xf)
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (logits.argmax(-1) == yt).float().mean().item()
    return acc


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="checkpoints/encoder.pt")
    ap.add_argument("--out", default="bee/weights/")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    logger.info(f"缺口3.2 重训：FEAT_DIM={FEAT_DIM}, STATE_DIM={STATE_DIM}")
    logger.info("加载冻结编码器...")
    encoder = load_encoder(args.encoder)

    logger.info("生成 BC 语料（含任务类型 one-hot）...")
    corpus = gen_synthetic_bc(encoder)

    # B 技能程序生成
    Xb, yb = gen_b_samples(200)
    corpus["B"] = (Xb, yb)

    # 训练四技能头
    heads = {}
    for skill in "ABCD":
        if skill not in corpus:
            logger.warning(f"技能 {skill} 无语料，跳过")
            continue
        X, y = corpus[skill]
        head = SkillHead(skill)
        acc = bc_train(head, X, y, args.epochs)
        logger.info(f"技能 {skill} BC 完成: acc={acc:.3f} ({len(y)} 样本)")
        heads[skill] = head
        # 保存 checkpoint
        save_path = f"checkpoints/skill_{skill.lower()}_feat12.pt"
        torch.save({
            "head_state": head.state_dict(),
            "feat_dim": FEAT_DIM,
            "skill": skill,
        }, save_path)
        logger.info(f"  → {save_path}")

    logger.info("四技能头重训完成。")
    logger.info("下一步：用 scripts/chain_router_deploy.sh 导出+部署验证")
    logger.info(f"  或手动: python3 -m training.skills.export --encoder {args.encoder}"
                f" --all-heads --router checkpoints/router_v7_best.pt --out {args.out}")


if __name__ == "__main__":
    main()
