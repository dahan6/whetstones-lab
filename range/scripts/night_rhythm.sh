#!/usr/bin/env bash
# 夜间节律：安静基线 + 每 10 分钟一波批处理（模拟 nightly batch）
ACTIONS=(
  "tar czf /tmp/bak-\$RANDOM.tgz /var/log 2>/dev/null; rm -f /tmp/bak-*.tgz"
  "find /var/log -name '*.log' -exec wc -l {} + 2>/dev/null | tail -1"
  "updatedb 2>/dev/null || true"
  "dpkg -l | wc -l"
  "journalctl --since '1 hour ago' --no-pager 2>/dev/null | wc -l"
  "du -sh /var/* 2>/dev/null | sort -rh | head -5"
)
while true; do
  sleep 600   # 安静 10 分钟
  for i in 1 2 3 4 5 6; do
    eval "${ACTIONS[$((RANDOM % ${#ACTIONS[@]}))]}" >/dev/null 2>&1
    sleep 2
  done
done
