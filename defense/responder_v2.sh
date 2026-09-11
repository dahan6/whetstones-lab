#!/usr/bin/env bash
# lado-responder v2（L4 猎杀级）：在 v1 基础上新增三个捕食维度
#   看门狗猎杀（先拆免疫系统，再杀本体）/ 深度文件扫描（$HOME+/proc 删除链接）/ 解剖签名（身体构型）
# 用法: responder_v2.sh scan | eliminate | wave
# 输出类型: WDOG(看门狗) PROC FILE UNIT CRON ANAT(解剖签名)
# 纪律同 v1：所有动作经 logger -t lado-responder 写 syslog，奖励计算器只读日志。
set -uo pipefail

TAG="lado-responder"

# 防御自身与系统关键进程白名单（绝不消杀）
WHITELIST_RE='^(falco|tracee|rsyslogd|systemd|sshd|cron|dbus-daemon|networkd-dispat|systemd-journal|systemd-logind|systemd-network|systemd-resolve|systemd-udevd|bash|sh|sudo)$'

log() { logger -t "$TAG" "$1"; }

# ── 看门狗猎杀：长寿循环 shell、孤儿进程、隐藏目录执行体 ──
# 目标：复活链器官（kill_chain 看门狗的典型形态：bash while 循环 + 变量分片混淆）
scan_watchdogs() {
    # 1. 命令行含 while/sleep 循环的长寿 bash/sh（看门狗形态签名）
    while read -r pid etimes args; do
        [ "$etimes" -lt 120 ] && continue   # 只查活过 2 分钟的
        case "$args" in
            *while*sleep*|*sleep*while*|*"until"*)
                echo "WDOG $pid shell_loop ${args:0:120}";;
        esac
    done < <(ps -eo pid=,etimes=,args= | grep -E "^\s*[0-9]+ \s*[0-9]+ (bash|sh) " )

    # 2. 孤儿化用户进程（ppid=1 且非系统服务）——看门狗常用 setsid 脱管
    while read -r pid ppid comm args; do
        [ "$ppid" = "1" ] || continue
        case "$comm" in bash|sh|sleep)
            case "$args" in
                *$'while'*|*".backup"*|*"kill_chain"*) echo "WDOG $pid orphan_$comm ${args:0:120}";;
            esac;;
        esac
    done < <(ps -eo pid=,ppid=,comm=,args=)

    # 3. 引用 .backup/复活脚本的任何进程命令行
    while read -r pid comm args; do
        case "$args" in
            *".backup"*|*"backup"*) echo "WDOG $pid backup_ref $comm";;
        esac
    done < <(ps -eo pid=,comm=,args= | grep -vE "$WHITELIST_RE")
}

# ── 深度文件扫描：$HOME 隐藏目录 + /proc 删除链接 + 近期可执行体 ──
scan_deep_files() {
    # 1. /home 下隐藏路径中的可执行文件（落点硬化的正面打击）
    while read -r f; do
        [ -f "$f" ] && echo "FILE $f"
    done < <(find /home -type f -perm -111 -path "*/.*" 2>/dev/null \
              -not -path "*/.config/*" -not -path "*/.local/*" -not -path "*/.cargo/*" \
              -not -path "*/.rustup/*" -not -path "*/.vscode*" -not -path "*/.mozilla/*")

    # 2. 运行中但磁盘已删的进程（rebirth 自删除的取证指纹）
    while read -r pid; do
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        case "$exe" in *" (deleted)") echo "PROC $pid deleted_exe $exe";; esac
    done < <(ls /proc | grep -E '^[0-9]+$')

    # 2b. 运行中且 exe 在隐藏路径的进程（活体版深扫——文件扫文件，这条扫活物）
    while read -r pid comm; do
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
        case "$exe" in
            /home/*/.*) echo "PROC $pid hidden_exe $exe";;
        esac
    done < <(ps -eo pid=,comm= | grep -vE "$WHITELIST_RE")

    # 3. /home 下 48h 内新增/改动的可执行文件（安装记录外的自生二进制）
    while read -r f; do
        [ -f "$f" ] && echo "FILE $f"
    done < <(find /home -type f -perm -111 -mtime -2 2>/dev/null \
              -not -path "*/.cargo/*" -not -path "*/.rustup/*" -not -path "*/.local/*" \
              -not -path "*/.config/*")
}

# ── 解剖签名：身体构型识别（张量清单 + 权重簇）──
scan_anatomy() {
    # 1. vocab.json + 权重 .bin 簇同目录 = 行为语法身体构型
    while read -r v; do
        d=$(dirname "$v")
        nbin=$(find "$d" -maxdepth 2 -name "*.bin" 2>/dev/null | wc -l)
        if [ "$nbin" -ge 3 ]; then
            echo "ANAT $d vocab+${nbin}bin"
        fi
        # 词表内容指纹（8-token 行为语法标记）
        if grep -qE '"ET:|"PROC:|"ARGV:' "$v" 2>/dev/null; then
            echo "ANAT $v grammar_vocab"
        fi
    done < <(find /home /tmp /dev/shm -name "vocab.json" -type f 2>/dev/null)

    # 2. 散落的张量权重文件（router/head_* 命名）
    while read -r f; do
        echo "ANAT $f tensor_blob"
    done < <(find /home /tmp /dev/shm -type f \( -name "router*.bin" -o -name "head_*.bin" -o -name "*_fc.*.bin" \) 2>/dev/null)
}

# ── v1 原样保留的四个扫描维度（/tmp 域）──
scan_legacy() {
    # 1. 可疑进程：/dev/shm、/tmp 下执行的
    while read -r pid comm; do
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
        case "$exe" in
            /dev/shm/*|/tmp/*|/var/tmp/*)
                echo "PROC $pid $comm $exe";;
        esac
    done < <(ps -eo pid=,comm= | grep -vE "$WHITELIST_RE")

    # 2. 落地文件：/dev/shm、/tmp 下的隐藏文件与可执行体
    while read -r f; do
        [ -f "$f" ] && echo "FILE $f"
    done < <(find /dev/shm /tmp -maxdepth 3 \( -name ".*" -o -perm -111 \) \
              -not -name "." -not -name ".." 2>/dev/null)

    # 3. 持久化点：用户级 systemd 目录（扩大到全部可疑引用，不只 /tmp）
    while read -r unit; do
        [ -L "$unit" ] && continue
        name=$(basename "$unit")
        case "$name" in open-vm-tools.*|cloud-*|ssh.*) continue;; esac
        if grep -qE '/dev/shm|/tmp|\.cache|\.local/state' "$unit" 2>/dev/null; then
            echo "UNIT $unit"
        fi
    done < <(find /home/*/.config/systemd/user \
              \( -name "*.service" -o -name "*.path" \) 2>/dev/null)

    # 4. cron/at 异常（扩大到 .cache/.local/state 引用）
    for crondir in /var/spool/cron/crontabs /etc/cron.d; do
        while read -r f; do
            [ -f "$f" ] || continue
            if grep -qE '/dev/shm|/tmp|\.cache|\.local/state' "$f" 2>/dev/null; then
                echo "CRON $f"
            fi
        done < <(ls "$crondir" 2>/dev/null | sed "s|^|$crondir/|")
    done

    # 5. 静态扫描（YARA 矩阵，v1 原样）
    if [ -x /usr/local/bin/static_scan.sh ]; then
        while read -r yhit; do
            echo "$yhit"
        done < <(/usr/local/bin/static_scan.sh 2>/dev/null)
    fi
}

scan() {
    # 注意：绝不能 for line in $(...)——无引号替换会按空白拆碎每行
    local out hits
    out=$( { scan_watchdogs; scan_legacy; scan_deep_files; scan_anatomy; } )
    hits=$(printf '%s\n' "$out" | grep -c . || true)
    printf '%s\n' "$out"
    log "SCAN v2 done hits=$hits"
    return 0
}

# ── 消杀：先杀看门狗（拆免疫），再杀本体，再清文件 ──
eliminate() {
    local killed=0 cleaned=0 disabled=0
    local lines
    lines=$(cat)
    # 第一遍：WDOG 优先（先杀复活链，本体死了才起不来）
    for kind in WDOG PROC ANAT FILE UNIT CRON; do
        while read -r k a b c; do
            [ "$k" = "$kind" ] || continue
            case "$k" in
                WDOG)
                    if kill -9 "$a" 2>/dev/null; then
                        log "KILL_WDOG pid=$a why=$b"; killed=$((killed+1))
                    fi;;
                PROC)
                    if kill -9 "$a" 2>/dev/null; then
                        log "KILL pid=$a comm=$b exe=$c"; killed=$((killed+1))
                    fi
                    [ -n "${c:-}" ] && [ -f "$c" ] && rm -f "$c" 2>/dev/null \
                        && { log "CLEAN file=$c"; cleaned=$((cleaned+1)); };;
                ANAT)
                    # 解剖签名命中：目录整个清除（权重簇+词表）
                    if [ -d "$a" ]; then
                        rm -rf "$a" 2>/dev/null && { log "CLEAN_ANAT dir=$a"; cleaned=$((cleaned+1)); }
                    else
                        rm -f "$a" 2>/dev/null && { log "CLEAN_ANAT file=$a"; cleaned=$((cleaned+1)); }
                    fi;;
                FILE)
                    rm -f "$a" 2>/dev/null && { log "CLEAN file=$a"; cleaned=$((cleaned+1)); };;
                UNIT)
                    name=$(basename "$a")
                    systemctl --user disable --now "$name" 2>/dev/null || true
                    rm -f "$a" 2>/dev/null && { log "DISABLE unit=$name path=$a"; disabled=$((disabled+1)); };;
                CRON)
                    rm -f "$a" 2>/dev/null && { log "CLEAN cron=$a"; cleaned=$((cleaned+1)); };;
            esac
        done <<< "$lines"
    done
    log "ELIMINATE v2 done killed=$killed cleaned=$cleaned disabled=$disabled"
}

case "${1:-}" in
    scan)      scan ;;
    eliminate) scan | eliminate ;;
    wave)      scan | eliminate ;;
    *) echo "用法: $0 scan|eliminate|wave" >&2; exit 1 ;;
esac
