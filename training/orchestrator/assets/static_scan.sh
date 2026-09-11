#!/usr/bin/env bash
# static_scan.sh: 静态扫描环节（检测矩阵，响应器波次的一部分）
# YARA 扫新增可执行体：$HOME 近 24h 新增可执行文件 + /tmp /dev/shm 可执行体。
# 命中输出 "YHIT <rule> <path>"（接响应器消杀语义）。
set -uo pipefail
RULES="/usr/local/share/yara/bee.yar"

scan_dir() {
    local dir="$1" days="${2:-1}"
    while read -r f; do
        [ -f "$f" ] || continue
        head -c4 "$f" 2>/dev/null | grep -q $'\x7fELF' || continue
        hits=$(yara "$RULES" "$f" 2>/dev/null | awk '{print $1}')
        for h in $hits; do
            echo "YHIT $h $f"
        done
    done < <(find "$dir" -maxdepth 4 -type f -perm -111 -mtime -"$days" 2>/dev/null)
}

scan_dir /tmp 1
scan_dir /dev/shm 1
scan_dir /home 6
scan_dir /var/tmp 1
exit 0
