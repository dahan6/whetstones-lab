#!/usr/bin/env bash
# Router 训练完成 → 导出全套权重 → bee 上机部署验证（goal 验收线）
set -uo pipefail
cd /home/lab/adaptive-agent-sim

R=checkpoints/router.pt
echo "[chain2] 等待 Router 训练产物..."
while [ ! -f "$R" ]; do sleep 60; done
echo "[chain2] router.pt 出现，等待训练进程退出（60s 缓冲）..."
sleep 60

echo "[chain2] === 导出全套权重 ==="
python3 -m training.skills.export --encoder checkpoints/encoder.pt \
    --all-heads --router "$R" --out bee/weights 2>&1 | tail -2

echo "[chain2] === bee 上机验证（t0，L1）==="
python3 -m training.deploy.verify_bee --vm range-l2-t0 --mode router \
    --steps 100 --minutes 6 --level 1 2>&1 | tail -15

echo "[chain2] 链路完成"
