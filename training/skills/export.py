"""权重导出：PyTorch checkpoint → bee 部署格式（manifest.json + 裸 fp32 .bin）。

纪律（红线3）：fp32 原始字节（little-endian，行主序），不定量；
导出后由纯 numpy 定序参考前向验证（同输入逐层误差 <1e-6，门5）。

用法:
  python3 -m training.skills.export --encoder checkpoints/encoder.pt \
      --head checkpoints/skill_c.pt --skill C --out bee/weights/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("skills.export")


def _dump_tensor(out: Path, name: str, arr: np.ndarray, manifest: list) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    fname = f"{name}.bin"
    arr.tofile(out / fname)
    manifest.append({"name": name, "shape": list(arr.shape), "file": fname})


def _dump_tensor_int8(out: Path, name: str, arr: np.ndarray, manifest: list) -> None:
    """int8 量化存储（红线3 修订：决策栈 int8，省 75% 体积）。
    per-tensor scale=max/127，.bin 存 int8 字节，scale 存 manifest fp32。
    bee 端加载时 q×scale 反量化到 fp32 前向（存储 int8，算术 fp32 一致）。"""
    arr = np.ascontiguousarray(arr, dtype="<f4")
    s = float(np.abs(arr).max() / 127.0) or 1.0
    q = np.clip(np.round(arr / s), -127, 127).astype(np.int8)
    fname = f"{name}.bin"
    q.tofile(out / fname)
    manifest.append({"name": name, "shape": list(arr.shape), "file": fname,
                     "dtype": "int8", "scale": s})


def export_encoder(ckpt_path: str, out: Path, manifest: list) -> None:
    _dump = _dump_tensor_int8
    ckpt = torch.load(ckpt_path, weights_only=False)
    state = ckpt["encoder_state"]
    for key in ("embed.weight", "event_fc.weight", "event_fc.bias",
                "pool_fc.weight", "pool_fc.bias"):
        _dump(out, key, state[key].numpy(), manifest)
    # 词表一并导出（bee 感知层需要 token→id 映射）
    (out / "vocab.json").write_text(json.dumps(ckpt["vocab"], ensure_ascii=False))
    logger.info(f"编码器导出: {ckpt_path} → {out} (+vocab.json)")


def export_head(ckpt_path: str, skill: str, out: Path, manifest: list,
                flat: bool = False) -> None:
    ckpt = torch.load(ckpt_path, weights_only=False)
    state = ckpt["head_state"]
    # PyTorch Sequential: net.0/net.2/net.4
    # flat=True  → head_a.0.weight（四头+router 扁平布局，bee --daemon router 用）
    # flat=False → head.0.weight（单头布局，--daemon <skill> 用）
    prefix = f"head_{skill.lower()}" if flat else "head"
    for i, src in enumerate(("net.0", "net.2", "net.4")):
        _dump_tensor_int8(out, f"{prefix}.{i}.weight", state[f"{src}.weight"].numpy(), manifest)
        _dump_tensor_int8(out, f"{prefix}.{i}.bias", state[f"{src}.bias"].numpy(), manifest)
    logger.info(f"技能头 {skill} 导出: {ckpt_path} (prefix={prefix})")


def export_router(ckpt_path: str, out: Path, manifest: list) -> None:
    ckpt = torch.load(ckpt_path, weights_only=False)
    state = ckpt["router_state"]
    for i, src in enumerate(("net.0", "net.2", "net.4")):
        _dump_tensor_int8(out, f"router.{i}.weight", state[f"{src}.weight"].numpy(), manifest)
        _dump_tensor_int8(out, f"router.{i}.bias", state[f"{src}.bias"].numpy(), manifest)
    logger.info(f"Router 导出: {ckpt_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--skill", default=None)
    ap.add_argument("--all-heads", action="store_true",
                    help="导出全部四技能头（head_a~head_d 扁平布局）")
    ap.add_argument("--router", default=None, help="router.pt 路径")
    ap.add_argument("--out", default="bee/weights")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    export_encoder(args.encoder, out, manifest)
    if args.all_heads:
        for s in "abcd":
            export_head(f"checkpoints/skill_{s}.pt", s.upper(), out, manifest, flat=True)
    elif args.head:
        export_head(args.head, args.skill or "?", out, manifest)
    if args.router:
        export_router(args.router, out, manifest)
    # 紧凑 JSON（bee 端极简解析器按此格式切分）
    (out / "manifest.json").write_text(
        json.dumps({"tensors": manifest}, separators=(",", ":")))
    logger.info(f"manifest.json 写出（{len(manifest)} 个张量）→ {out}")


if __name__ == "__main__":
    main()
