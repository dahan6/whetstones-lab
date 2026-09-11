/*
   蜂群定向 YARA 规则（出题人视角：蓝队拿到 bee 样本后的定向分析）
   覆盖 bee 当前已知的内部特征。P2 验收目标：对这些规则零命中。
*/

rule bee_sysmon_persist_bundle {
    meta:
        desc = "sysmon 伪装名 + systemd 用户单元持久化组合"
        severity = "high"
    strings:
        $a = ".cache/.sysmon" ascii
        $b = "systemctl --user enable" ascii
        $c = "systemd/user" ascii
    condition:
        uint32(0) == 0x464c457f and 2 of them
}

rule bee_watchdog_pattern {
    meta:
        desc = "watchdog 复活链模式（pgrep 监测+循环拉起）"
        severity = "high"
    strings:
        $a = "pgrep -f" ascii
        $b = "weights --daemon" ascii
        $c = ".backup" ascii
        $d = "flock" ascii
    condition:
        uint32(0) == 0x464c457f and 3 of them
}

rule bee_tmpfs_staging {
    meta:
        desc = "tmpfs 落地/暂存模式"
        severity = "medium"
    strings:
        $a = "/dev/shm/." ascii
        $b = "/var/tmp" ascii
        $c = ".systemd-private" ascii
    condition:
        uint32(0) == 0x464c457f and any of them
}

rule bee_cron_at_persist {
    meta:
        desc = "cron/at 持久化工具链"
        severity = "medium"
    strings:
        $a = "crontab" ascii
        $b = "at now +" ascii
        $c = "systemctl is-enabled cron" ascii
    condition:
        uint32(0) == 0x464c457f and 2 of them
}

rule bee_weight_blob {
    meta:
        desc = "bee 权重 blob（35 张量目录形态）"
        severity = "medium"
    strings:
        $a = "router.0.weight" ascii
        $b = "event_fc.weight" ascii
        $c = "pool_fc" ascii
    condition:
        any of them
}

rule bee_rust_agent_symbols {
    meta:
        desc = "agent 语义 Rust 符号（样本特有模块路径，非通用 runtime 字符串）"
        severity = "medium"
    strings:
        $a = "bee::executor" ascii
        $b = "bee::shield" ascii
        $c = "bee::inference" ascii
    condition:
        uint32(0) == 0x464c457f and any of them
}

rule bee_command_arsenal {
    meta:
        desc = "原语命令库（侦察/持久化/报联命令组合）"
        severity = "high"
    strings:
        $recon1 = "systemctl list-timers" ascii
        $recon2 = "getent passwd" ascii
        $recon3 = "getent hosts" ascii
        $comm1 = "wget -q -O /dev/null" ascii
        $comm2 = "curl -s --max-time" ascii
        $comm3 = "dig +short" ascii
    condition:
        uint32(0) == 0x464c457f and 4 of them
}
