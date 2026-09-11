#!/usr/bin/env bash
# LADO 靶场：一键还原 VM 到干净基线（销毁差量盘，从 backing file 重建）
# 用法: ./reset-vm.sh <vm-name> <tier> [ram_mb] [vcpus]
set -euo pipefail

NAME="${1:?用法: reset-vm.sh <vm-name> <tier> [ram_mb] [vcpus]}"
TIER="${2:?缺少 tier（重建 cloud-init 需要）}"
RAM="${3:-4096}"
VCPUS="${4:-2}"
DISK="$HOME/lado-range/images/${NAME}.qcow2"

virsh destroy "$NAME" 2>/dev/null || true
virsh undefine "$NAME" --nvram 2>/dev/null || virsh undefine "$NAME" 2>/dev/null || true
rm -f "$DISK"
rm -rf "$HOME/lado-range/images/${NAME}-seed"

echo "[reset-vm] 旧实例已销毁，重建中..."
exec "$HOME/lado-range/scripts/create-vm.sh" "$NAME" "$TIER" "$RAM" "$VCPUS"
