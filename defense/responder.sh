#!/usr/bin/env bash
# lado-responder: 消杀响应器（环境的主动防御臂，跑在 L2 克隆上）
# 用法: responder.sh scan | eliminate | wave
#   scan      主动扫描（可疑进程/落地文件/持久化点），只报告
#   eliminate 对 scan 发现的实体执行消杀（kill/清除/禁用）
#   wave      scan + eliminate 完整一波
# 所有动作经 logger -t lado-responder 写入 syslog → rsyslog 转发宿主机，
# 奖励计算器只读这些日志，不做任何"代打分"。
set -uo pipefail

TAG="lado-responder"

# 防御自身与系统关键进程白名单（绝不消杀）
WHITELIST_RE='^(falco|tracee|rsyslogd|systemd|sshd|cron|dbus-daemon|networkd-dispat|systemd-journal|systemd-logind|systemd-network|systemd-resolve|systemd-udevd|bash)$'

log() { logger -t "$TAG" "$1"; }

# ── 扫描：输出可疑实体清单（进程PID + 文件路径 + 持久化单元）────
scan() {
    local hits=0

    # 1. 可疑进程：/dev/shm、/tmp 下执行的；名字带 agent/bee 的
    while read -r pid comm; do
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
        case "$exe" in
            /dev/shm/*|/tmp/*|/var/tmp/*)
                echo "PROC $pid $comm $exe"; hits=$((hits+1));;
        esac
    done < <(ps -eo pid=,comm= | grep -vE "$WHITELIST_RE")

    # 2. 落地文件：/dev/shm、/tmp 下的隐藏文件与可执行体
    while read -r f; do
        [ -f "$f" ] && { echo "FILE $f"; hits=$((hits+1)); }
    done < <(find /dev/shm /tmp -maxdepth 3 \( -name ".*" -o -perm -111 \) \
              -not -name "." -not -name ".." 2>/dev/null)

    # 3. 持久化点：仅查用户级 systemd 目录（agent 落点），
    #    系统目录合法单元众多，扫了必误伤（open-vm-tools 教训）
    while read -r unit; do
        [ -L "$unit" ] && continue
        name=$(basename "$unit")
        case "$name" in
            open-vm-tools.*|cloud-*|ssh.*) continue;;
        esac
        if grep -qE '/dev/shm|/tmp' "$unit" 2>/dev/null; then
            echo "UNIT $unit"; hits=$((hits+1))
        fi
    done < <(find /home/*/.config/systemd/user \
              \( -name "*.service" -o -name "*.path" \) 2>/dev/null)

    # 4. cron/at 异常
    for crondir in /var/spool/cron/crontabs /etc/cron.d; do
        while read -r f; do
            [ -f "$f" ] || continue
            if grep -qE '/dev/shm|/tmp' "$f" 2>/dev/null; then
                echo "CRON $f"; hits=$((hits+1))
            fi
        done < <(ls "$crondir" 2>/dev/null | sed "s|^|$crondir/|")
    done

    # 5. 静态扫描（检测矩阵：YARA 扫新增可执行体，命中即线索）
    if [ -x /usr/local/bin/static_scan.sh ]; then
        while read -r yhit; do
            echo "$yhit"; hits=$((hits+1))
        done < <(/usr/local/bin/static_scan.sh 2>/dev/null)
    fi

    log "SCAN done hits=$hits"
    return 0
}

# ── 消杀：对 scan 输出逐条执行 ─────────────────────────────
eliminate() {
    local killed=0 cleaned=0 disabled=0
    while read -r kind a b c; do
        case "$kind" in
            PROC)
                if kill -9 "$a" 2>/dev/null; then
                    log "KILL pid=$a comm=$b exe=$c"; killed=$((killed+1))
                fi
                # 进程对应的可执行文件也清掉
                [ -n "${c:-}" ] && [ -f "$c" ] && rm -f "$c" 2>/dev/null \
                    && { log "CLEAN file=$c"; cleaned=$((cleaned+1)); }
                ;;
            FILE)
                rm -f "$a" 2>/dev/null && { log "CLEAN file=$a"; cleaned=$((cleaned+1)); }
                ;;
            UNIT)
                name=$(basename "$a")
                systemctl --user disable --now "$name" 2>/dev/null || true
                rm -f "$a" 2>/dev/null && { log "DISABLE unit=$name path=$a"; disabled=$((disabled+1)); }
                ;;
            CRON)
                rm -f "$a" 2>/dev/null && { log "CLEAN cron=$a"; cleaned=$((cleaned+1)); }
                ;;
        esac
    done
    log "ELIMINATE done killed=$killed cleaned=$cleaned disabled=$disabled"
}

case "${1:-}" in
    scan)      scan ;;
    eliminate) scan | eliminate ;;
    wave)      scan | eliminate ;;
    *) echo "用法: $0 scan|eliminate|wave" >&2; exit 1 ;;
esac
