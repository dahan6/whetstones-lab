"""权重加载：编码器/技能头的统一入口。"""
from __future__ import annotations

import torch

from training.encoder.model import Encoder
from training.skills.heads import SkillHead


def load_encoder(path: str = "checkpoints/encoder.pt") -> Encoder:
    ckpt = torch.load(path, weights_only=False)
    vocab: dict[str, int] = ckpt["vocab"]
    enc = Encoder(len(vocab))
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()
    enc._vocab = vocab          # 驱动器需要词表做 tokenize
    return enc


def load_head(skill: str, path: str) -> SkillHead:
    head = SkillHead(skill)
    ckpt = torch.load(path, weights_only=False)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return head


def save_head(head: SkillHead, path: str, meta: dict | None = None) -> None:
    torch.save({"head_state": head.state_dict(),
                "skill": head.skill, "meta": meta or {}}, path)
