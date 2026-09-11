#!/usr/bin/env bash
# LADO 靶场隔离兜底：即使 libvirt 网络配置被改动，
# 也强制禁止隔离网段的任何流量被转发到物理接口。
# 需要 root 执行。幂等：重复运行不会叠加规则。
set -euo pipefail

BRIDGE="virbr-lado"
SUBNET="192.0.2.0/24"

# 1. 禁止从隔离网桥转发出去的任何流量
iptables -C FORWARD -i "$BRIDGE" -j DROP 2>/dev/null || \
  iptables -I FORWARD 1 -i "$BRIDGE" -j DROP

# 2. 禁止宿主机把任何流向隔离网段的流量路由出去（双向上锁）
iptables -C FORWARD -o "$BRIDGE" -m conntrack --ctstate NEW -j DROP 2>/dev/null || \
  iptables -I FORWARD 1 -o "$BRIDGE" -m conntrack --ctstate NEW -j DROP

# 3. OUTPUT 兜底：隔离网段地址不应出现在默认路由里，防御性检查
if ! ip route get 192.0.2.100 | head -1 | grep -q "dev $BRIDGE"; then
  echo "WARNING: 隔离网段流量可能不经 $BRIDGE，请检查路由表" >&2
fi

echo "[lockdown] 隔离规则已生效："
iptables -L FORWARD -n -v --line-numbers | head -8
echo "[lockdown] 验证方法：VM 内 ping 192.0.2.1 应通，ping 任何公网地址应失败"
