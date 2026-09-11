#!/usr/bin/env bash
# LADO 靶场：从基础镜像克隆一台梯度防御 VM
# 用法: ./create-vm.sh <vm-name> <tier> [ram_mb] [vcpus]
#   tier: L0(裸基线) | L1(auditd) | L2(falco) | ...
# 示例: ./create-vm.sh range-l0-a L0 4096 2
set -euo pipefail

NAME="${1:?用法: create-vm.sh <vm-name> <tier> [ram_mb] [vcpus]}"
TIER="${2:?缺少防御层级 tier}"
RAM="${3:-4096}"
VCPUS="${4:-2}"

BASE_IMG="$HOME/lado-range/images/noble-server-cloudimg-amd64.img"
DISK="$HOME/lado-range/images/${NAME}.qcow2"
SEED_DIR="$HOME/lado-range/images/${NAME}-seed"
USER_DATA="$HOME/lado-range/xml/user-data.template"
NETWORK="lado-isolated"
PUBKEY="$(cat "$HOME/.ssh/lado_range.pub")"

[ -f "$DISK" ] && { echo "错误: $DISK 已存在，先 reset-vm.sh 或删除"; exit 1; }

# 1. 差量磁盘：backing file 保证同镜像同构，重置即删了重建
qemu-img create -f qcow2 -b "$BASE_IMG" -F qcow2 "$DISK" 20G

# 2. cloud-init 种子（hostname / tier / ssh key 注入）
mkdir -p "$SEED_DIR"
sed -e "s/VM_HOSTNAME_PLACEHOLDER/$NAME/" \
    -e "s/DEFENSE_TIER_PLACEHOLDER/$TIER/g" \
    -e "s|SSH_PUBKEY_PLACEHOLDER|$PUBKEY|" \
    "$USER_DATA" > "$SEED_DIR/user-data"
cat > "$SEED_DIR/meta-data" <<EOF
instance-id: ${NAME}-$(date +%s)
local-hostname: ${NAME}
EOF
cloud-localds "$SEED_DIR/seed.iso" "$SEED_DIR/user-data" "$SEED_DIR/meta-data"

# 3. 定义并启动（挂隔离网，无图形，串口日志落盘便于调试）
# 注意：virt-install/cloud-localds 的 shebang 是 env python3，会被 conda 抢走，
# 必须显式用系统 python 执行
/usr/bin/python3 /usr/bin/virt-install \
  --name "$NAME" \
  --memory "$RAM" --vcpus "$VCPUS" \
  --disk "path=$DISK,format=qcow2,bus=virtio" \
  --disk "path=$SEED_DIR/seed.iso,device=cdrom" \
  --network "network=$NETWORK,model=virtio" \
  --os-variant ubuntu24.04 \
  --graphics none \
  --serial file,path="$HOME/lado-range/telemetry/${NAME}-console.log" \
  --noautoconsole \
  --import

echo "[create-vm] $NAME (tier=$TIER) 已启动，等待 cloud-init..."
echo "[create-vm] 查看 IP: virsh domifaddr $NAME"
echo "[create-vm] 登录: ssh -i ~/.ssh/lado_range range@<ip>"
