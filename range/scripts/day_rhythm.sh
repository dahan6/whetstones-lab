#!/usr/bin/env bash
# 白天节律：持续交互式运维活动（短间隔）
ACTIONS=(
  "ls -la /var/log | head -20"
  "ps aux --sort=-%mem | head -10"
  "systemctl status cron --no-pager"
  "df -h"
  "free -m"
  "ss -tlnp"
  "journalctl -n 15 --no-pager"
  "cat /proc/loadavg"
  "find /etc -maxdepth 1 -name '*.conf' | head"
  "cat /etc/os-release"
  "uptime"
  "systemctl list-units --type=service --state=running --no-pager | head -15"
  "head -5 /etc/passwd"
  "ip addr show"
  "git --version 2>/dev/null; vim --version 2>/dev/null | head -1"
)
while true; do
  eval "${ACTIONS[$((RANDOM % ${#ACTIONS[@]}))]}" > /dev/null 2>&1
  sleep 0.$((RANDOM % 4 + 3))   # 白天交互式操作是稠密的（0.3-0.7s）
done
