#!/usr/bin/env bash
# D L1 完成 → 门2-D → Router 全池训练 的自动接力链
# 轮询 skill_d.pt mtime 变化判定 D L1 完成
set -uo pipefail
cd /home/lab/adaptive-agent-sim

D_CKPT=checkpoints/skill_d.pt
BASE=$(stat -c %Y "$D_CKPT")
echo "[chain] 等待 D L1 完成（当前 mtime=$BASE）..."
while true; do
    NOW=$(stat -c %Y "$D_CKPT")
    if [ "$NOW" != "$BASE" ]; then
        echo "[chain] D L1 已保存（mtime=$NOW）"
        break
    fi
    sleep 60
done

echo "[chain] === 门2 验收 D（t0-t3）==="
python3 -m training.skills.gate2 --skills D --vm-ids 0 1 2 3 2>&1 | grep -E "技能|门2"

echo "[chain] === Router 全池训练 ==="
python3 -m training.router.train_router --level 1 --gens 15 --pop 12 2>&1 | grep -E "gen|门3|保存"

echo "[chain] 链路完成"
