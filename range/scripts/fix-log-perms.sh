#!/usr/bin/env bash
# 修复遥测日志读取权限（需要 sudo 执行一次）
# 让 adm 组（含训练用户）能读 rsyslog 收到的靶场日志
set -euo pipefail

cat >> /etc/rsyslog.d/10-lado-range.conf <<'EOF'

# 训练编排器需要读日志：adm 组可读
$DirGroup adm
$DirCreateMode 0750
$FileGroup adm
$FileCreateMode 0640
EOF

systemctl restart rsyslog
echo "[fix-log-perms] 已生效。历史目录权限手动修一次："
echo "  sudo chgrp -R adm /var/log/lado-range && sudo chmod -R g+rX /var/log/lado-range"
