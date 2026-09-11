# 蜂群系统：红队代理架构研究技术报告

**版本**: 1.0
**日期**: 2026-07-29
**状态**: 骨架验证完成，核心问题待解决

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [基础设施](#2-基础设施)
3. [防御模型](#3-防御模型)
4. [攻击模型](#4-攻击模型)
5. [核心发现](#5-核心发现)
6. [C/Rust 实现](#6-crust-实现)
7. [真实环境验证](#7-真实环境验证)
8. [已知问题与局限](#8-已知问题与局限)
9. [技术债务](#9-技术债务)
10. [下一步方向](#10-下一步方向)
11. [附录](#11-附录)

---

## 1. 执行摘要

本项目研究了一种基于"时序插入"的红队代理架构，核心创新是**不是"做什么"决定隐蔽性，而是"什么时候做"和"怎么做（时序）"决定暴露**。agent 寄生在真实系统遥测流（Tracee eBPF 事件流）中，在系统正常运维的缝隙里插入动作，使检测器难以区分 agent 行为和系统自身行为。

**关键成果**：
- 从 100% 被检测到 0% 被检测的突破（自定义检测下）
- 因果链策略 reward=98.84，比孤立动作提升 74%
- 隐形原语库在真实 EDR（Falco）下大部分有效
- C/Rust 静态链接二进制（2.0MB），零依赖部署

**核心局限**：
- 硬编码因果链是规则系统，不是学习系统
- C2 通信用 DNS 隧道 + HTTP 伪装，是十年前的技术
- 自定义检测代替真实 EDR，不是真正的对抗
- C/Rust 和 Python 的 surprise 打分不一致

---

## 2. 基础设施

### 2.1 靶场环境

| 组件 | 配置 | 状态 |
|---|---|---|
| 宿主机 | Ubuntu 26.04 LTS, KVM/libvirt 12.0.0, QEMU 10.2.1 | 运行中 |
| 隔离网络 | `lado-isolated` 192.0.2.0/24, 无出网路由 | 运行中 |
| range-l0-a | Ubuntu 24.04, bare VM, 无防御 | 运行中 (192.0.2.140) |
| range-l2-a | Ubuntu 24.04, Tracee eBPF | 运行中 (192.0.2.188) |

### 2.2 遥测采集

| 组件 | 配置 | 数据量 |
|---|---|---|
| Tracee v0.24.1 | eBPF 运行时检测 | `sched_process_exec` + `security_socket_connect` |
| rsyslog | VM → 宿主机实时转发 | 28,719 条事件（2026-07-28） |
| 解析器 | `parse_events.py` | 7-token 离散化（ET, PROC, ARGV, PARENT, UID, DST, DT） |

### 2.3 防御栈

| 层 | 组件 | 说明 |
|---|---|---|
| L1 | TinyGPT | 行为序列检测，169 vocab, p995=6.297 |
| L2 | YARA | 7 条规则：base64, /dev/tcp, hidden_proc, nc, cron, lateral, exfil |
| L3 | 白名单 | 200+ 进程白名单 |
| L4 | Syscall 规则 | 12 条子串匹配恶意模式 |
| L5 | Falco 0.39.2 | 真实 EDR，4 条自定义规则（hidden_proc, base64, network, cron） |

---

## 3. 防御模型

### 3.1 TinyGPT 行为序列检测

**架构**: 4 层 Transformer, d_model=128, CTX=128, 0.85M 参数

**训练**: 8 秒，28K 真实 Tracee 事件，val_ppl=3.3

**检测逻辑**:
```
surprise = max(-log P(token_t | token_1...t-1))
if surprise > threshold(6.297): detected
```

**效果**: 对"正常运维序列"给出低 surprise（< 5.0），对"异常序列"给出高 surprise（> 6.3）

### 3.2 多层检测聚合

**检测逻辑**: `combined = max(tinygpt_surprise, yara_score, whitelist_score, syscall_score)`

**效果**: 单层 TinyGPT 被 LOLBins 绕过（surprise=5.0 < 6.297），但 YARA 和 Syscall 规则能检测 `crontab -e`、`systemctl enable`、`nc -l`、`base64 -d`、`/tmp/.hidden` 等恶意模式。

### 3.3 Falco 真实 EDR

**安装**: 零依赖单二进制，直接 scp 推送

**自定义规则**:
```yaml
- rule: Suspicious Hidden Process    # /tmp/.hidden* 进程执行
- rule: Suspicious Base64 Decode     # base64 -d 命令
- rule: Suspicious Network Connection # 连接内部 IP（排除 systemd-resolve）
- rule: Cron Modification             # crontab 或 at 命令
```

**测试效果**: Falco 能检测 `base64 -d`、`cp /tmp/.hidden_ls`、`ping 内部 IP`，但不覆盖 `journalctl`、`systemctl`、`awk`、`ss`、`at`、`/dev/shm`。

---

## 4. 攻击模型

### 4.1 时序插入模型（核心创新）

**核心思想**: agent 不是"替换"环境，而是"寄生"在真实 Tracee 事件流中，在系统正常运维的缝隙里插入动作。

```
真实系统事件流: [sshd] [cron] [journald] [systemd] [sshd] ...
agent 插入:            ↑ 在这里插 [journalctl]
组合后: [sshd] [cron] [journald] [journalctl] [systemd] [sshd] ...
```

**检测器视角**: 看的是 `P(token_t | token_1...t-1)`，agent 的 token 序列必须匹配真实系统的转移概率。

**突破**: 从 100% 被检测到 0% 被检测（自定义检测下）。

### 4.2 因果链策略（最有效单策略）

**核心思想**: 每个动作都有理由。不是"我执行了 `journalctl`"，而是"我检测到磁盘满了 → 调查 → 清理日志 → 验证空间"。

**6 条因果链**:
1. `disk_cleanup`: df → du → journalctl --vacuum-time → df
2. `service_health`: systemctl status → list-dependencies → is-enabled → status
3. `security_audit`: ls /etc → awk /etc/passwd → chmod → ls
4. `network_trouble`: ip addr → ss → ping → ip addr
5. `package_maint`: dpkg --list → apt list → echo /dev/shm → dpkg --list
6. `cron_maint`: crontab -l → ls /etc/cron.d → at now + 2 min → systemctl list-timers

**效果**: reward=98.84, tasks=23, det=0%（比孤立动作 +74% reward, +35% tasks）

### 4.3 隐形原语库

**读取类**:
- `awk -F: '{print $1}' /etc/passwd` （绕过 `cat /etc/passwd`）
- `ss -tlnp` （绕过 `netstat`）
- `journalctl -n 5 --no-pager` （绕过 `tail /var/log/syslog`）
- `crontab -l` （绕过 `cat /etc/crontab`）

**修改类**:
- `at now + 1 minute` （绕过 `crontab -e`）
- `/dev/shm/.cache_data` （绕过 `/tmp/.hidden`）
- `systemctl is-enabled cron` （绕过 `systemctl enable`）

**传输类**:
- `host $(hostname)` （绕过 `getent`，DNS 查询看起来像正常解析）
- `wget -q -O /dev/null` （绕过 `curl`，HTTP 请求看起来像正常下载）
- `ping -c 1 127.0.0.1` （绕过 `nc`，ICMP 看起来像网络测试）

**传播类**:
- `cp /tmp/agent /dev/shm/.systemd-private/` （内存文件系统，不落盘）
- `ping 192.0.2.140` （跨 VM 连通性测试）
- `systemd user service` （多点持久化）
- `watchdog.sh` （kill_chain 复活）

**高级原语**:
- `LD_PRELOAD=/dev/shm/.hook.so` （DLL 劫持）
- `PATH=/dev/shm/.bin:$PATH` （命令包装）
- `python3 -m http.server 8080` （端口混用）

### 4.4 ES 时序优化

**优化空间**: 14 个参数（insert_threshold, min_wait, scan_buffer, cooldown, phase_alignment, aggressive_ratio, choice_bias 等）

**ES 发现的最优参数**:
```
insert_threshold: 0.88  (比专家指定的 0.5 更保守)
min_wait:         1     (比专家指定的 5 更激进)
scan_buffer:      2
cooldown:         4     (比专家指定的 15 更短)
phase_alignment:  0.83  (高度对齐系统节律)
aggressive_ratio: 0.82  (比专家指定的 0.3 更激进)
```

**关键发现**: ES 发现"高频短间隔插入"比"低频长间隔"更安全——因为单个插入在 defender 看来是异常突发事件，而连续插入看起来像 cron 作业的批量执行。

### 4.5 传播复制

**5 种传播原语**:
1. `local_copy`: `cp /tmp/agent /dev/shm/.systemd-private/`
2. `cross_vm`: `ping 192.0.2.140`（测试连通性）
3. `multi_persist`: `systemd user service`（多点持久化）
4. `kill_chain`: `watchdog.sh`（死了复活）
5. `data_replicate`: `cp findings /dev/shm/.backup.txt`

**效果**: reward=57.05, tasks=17, det=0%（传播动作全部干净通过）

---

## 5. 核心发现

### 5.1 时序比动作内容更重要

```
同一个动作 (journalctl)，不同时机：
  在 df → du 之后执行: surprise=5.0  ← 正常
  在 crontab 之后执行: surprise=9.7  ← 异常

同一个动作 (crontab -l)，不同时机：
  单独执行: surprise=5.0  ← 正常
  紧跟 crontab -e: surprise=8.0  ← 异常
```

**结论**: 检测器看的是 `P(token_t | token_1...t-1)`，不是 token 本身。agent 的 token 序列必须匹配真实系统的转移概率。

### 5.2 因果链是最有效的单策略

```
孤立动作: [crontab] [journalctl] [systemctl] [ss] [awk]  ← 随机，可疑
因果链:  [df] [du] [journalctl] [df]                      ← 磁盘清理叙事，正常

reward=98.84 vs 56.92，+74%
tasks=23 vs 17，+35%
```

### 5.3 隐形原语在真实 EDR 下大部分有效

```
Falco 默认规则下干净:
  journalctl, systemctl, awk, ss, at, /dev/shm, host, wget

Falco 自定义规则下被检测:
  base64 -d, cp /tmp/.hidden_ls, ping 192.0.2.1
```

### 5.4 ES 能发现人类未指定的策略

```
ES 发现的优化：
  min_wait=1（比专家指定的 5 更激进）
  aggressive_ratio=0.82（比专家指定的 0.3 更激进）
  phase_alignment=0.83（高度对齐系统节律）
  用 ss 替代 crontab 做 probe（ES 自己发现的更好伪装）
```

### 5.5 组合 > 单策略

```
S1 alone (因果链):  reward=98.84, tasks=23
S2 alone (流量伪装): reward=35.00, tasks=2
S3 alone (时间碎片): reward=14.92, tasks=3
S4 alone (多VM):     reward=54.04, tasks=2

COMBINED:            reward=126.50, tasks=46  ← 比最优单策略高 28%
```

---

## 6. C/Rust 实现

### 6.1 架构

```
swarm/
├── common/           # 公共库
│   ├── types.rs      # BeeRole, ActionType, CausalChain, TimingParams
│   ├── crypto.rs     # AES-256-GCM 加密
│   └── protocol.rs   # TCP 帧、DNS 隧道、HTTP 伪装
├── bee/              # 蜂（Drone）
│   ├── main.rs       # 入口
│   ├── causal.rs     # 因果链执行（6 条链）
│   ├── primitives.rs # 隐形原语（read/modify/transmit/propagate）
│   ├── timing.rs     # 时序决策（什么时候插入）
│   └── executor.rs   # 主循环（observe→decide→execute→report）
├── hive/             # 蜂巢
│   ├── main.rs       # 入口
│   ├── manager.rs    # 蜂群管理（生成、监控、替换）
│   └── learner.rs    # 离线学习（分析结果，优化参数）
└── c2/               # C2
    ├── main.rs       # 入口
    ├── server.rs     # 服务器（接受蜂巢连接）
    └── state.rs      # 状态管理（蜂巢注册、任务跟踪）
```

### 6.2 二进制

| 组件 | 大小 | 链接 | 依赖 |
|---|---|---|---|
| swarm-bee | 2.0 MB | 完全静态（musl） | 零 |
| swarm-hive | 2.0 MB | 完全静态（musl） | 零 |
| swarm-c2 | 2.0 MB | 完全静态（musl） | 零 |

**可以直接 scp 到任何 Linux 机器上运行，不需要 Python、不需要 glibc、不需要任何依赖。**

### 6.3 端到端验证

```
C2:     启动，监听 0.0.0.0:9000                    ✅
Hive:   启动，监听 127.0.0.1:9001，生成 3 个蜂      ✅
Bee:    执行 100 步，29 任务完成，0% 检测           ✅（模拟环境）
```

---

## 7. 真实环境验证

### 7.1 Python 验证（自定义检测）

```
环境: TemporalInsertionEnv（模拟）
检测: TinyGPT + YARA + 白名单 + Syscall
结果: reward=98.84, tasks=23, det=0%
```

### 7.2 C/Rust 模拟验证

```
环境: 模拟（无真实 Tracee）
结果: alive=100/100, tasks=29, det=0%
```

### 7.3 Falco 真实 EDR 验证

```
环境: range-l2-a (真实 VM, Tracee eBPF)
检测: Falco 0.39.2 (真实 EDR)
结果:
  干净: journalctl, systemctl, awk, ss, at, /dev/shm, host, wget
  被检测: base64 -d, cp /tmp/.hidden_ls, ping 192.0.2.1
```

### 7.4 C/Rust 真实环境对接

```
环境: range-l2-a (真实 VM)
检测: Falco (真实 EDR)
结果: alive=17/100, tasks=6, det=1 (step 17 被检测)
问题: Rust 频率打分给 communicate 10.0，Python TinyGPT 只给 5.0
```

---

## 8. 已知问题与局限

### 8.1 假装"自适应"

**问题**: 当前架构是"预设策略 + ES 时序参数优化"，不是真正的学习系统。

**具体表现**:
- 6 条硬编码因果链是规则系统，agent 不会发现新链
- ES 只优化 14 个时序参数，不学操作内容
- agent 没有"发现新操作方式"的能力

**真实影响**: 如果 defender 学会检测当前 6 条链，agent 就失效了。

### 8.2 C2 通信是十年前的技术

**问题**: DNS 隧道 + HTTP 伪装是教科书级 C2 传统技能。

**现代 NDR 对这两样的检测**:
- DNS 流量分析：查询频率、域名熵值、响应大小、TTL 异常
- HTTP 流量分析：User-Agent 一致性、请求间隔、payload 大小分布
- TLS 指纹：JA3/JA4 指纹匹配

**真实影响**: 现代 NDR（CrowdStrike, Darktrace, Vectra）对 DNS/HTTP 的检测是最成熟的领域之一。

### 8.3 自定义检测代替真实 EDR

**问题**: 之前用 TinyGPT+YARA+白名单+Syscall 做"检测"，不是真实对抗。

**真实影响**: agent 在"我们定义的检测"下 0% 被检测，但在真实 EDR 下可能完全不同。这是"应试"，不是真正的对抗。

### 8.4 C/Rust 和 Python 的 surprise 打分不一致

**问题**: Rust 用频率打分（稀有 token 给高分），Python 用 TinyGPT NLL（更精准）。

**具体表现**:
- `communicate` (host $(hostname)): Rust 给 10.0，Python 给 5.0
- 导致 Rust 蜂在 step 17 被检测，Python 蜂在 step 100 不被检测

**真实影响**: C/Rust 和 Python 的行为不一致，无法直接迁移。

### 8.5 真实环境训练太慢

**问题**: 100 episodes × 3.5 分钟 = 6 小时，不适合大规模 RL。

**真实影响**: 无法在真实环境做大规模强化学习，只能在模拟环境训练。

---

## 9. 技术债务

| 债务 | 优先级 | 说明 |
|---|---|---|
| 硬编码因果链 | 高 | 6 条链是规则系统，不是学习系统 |
| C2 通信落后 | 高 | DNS 隧道 + HTTP 伪装，现代 NDR 最成熟 |
| 自定义检测 | 高 | 不是真实 EDR，不是真正的对抗 |
| Surprise 不一致 | 中 | C/Rust 和 Python 的 surprise 打分不一致 |
| 真实环境训练慢 | 中 | 无法在真实环境做大规模 RL |
| 孢子化缺失 | 中 | 没有自解压孢子，无法自动扩散 |
| Mesh 通信缺失 | 低 | agent 间无法通信，无法 relay |
| 角色分化缺失 | 低 | 单一 agent，没有 Scanner/Executor/Propagator 分化 |

---

## 10. 下一步方向

### 10.1 短期（承认局限，专注时序优化）

**目标**: 把"时序优化 + 因果链"做到极致，不假装"自适应"

**行动**:
1. 用 Falco 做真实检测，验证哪些策略真正有效
2. ES 优化时序参数，用 Falco 反馈找最优 insert_threshold/min_wait/cooldown
3. 放弃"自适应"口号，承认是"因果链 + ES 时序参数优化"

### 10.2 中期（真正的学习系统）

**目标**: agent 自己发现"什么序列看起来像正常运维"，不是硬编码链

**行动**:
1. 观察真实系统的正常运维序列（Tracee 事件流）
2. 学习 P(next_action | history, environment) 的分布
3. agent 自己选择最像正常的动作序列，不是预设的链
4. 如果发现当前序列被检测，自动切换到另一个序列

### 10.3 长期（完整分布式架构）

**目标**: C2 + 蜂巢 + 蜂群 + 孢子化 + Mesh 通信

**行动**:
1. 孢子化：把 agent 压缩成 <4KB 自解压孢子
2. C2 主控机：加密信道，任务分发，密钥轮换
3. 蜂巢传播：分布式学习节点，生成新蜂群
4. Mesh 通信：agent 间直连，relay，共享文件系统

---

## 11. 附录

### 11.1 项目文件结构

```
adaptive-agent-sim/
├── src/
│   ├── environment/
│   │   ├── temporal_insertion.py   # 时序插入环境
│   │   └── real_telemetry.py       # 真实遥测采集
│   ├── causal_chains.py            # 因果链策略
│   ├── traffic_obfuscation.py      # 流量伪装
│   ├── stealth_primitives.py       # 隐形原语库
│   ├── propagation.py              # 传播复制
│   ├── advanced_primitives.py      # 高级原语（DLL/包装/端口）
│   ├── detection/
│   │   └── falco_detector.py       # Falco 真实检测
│   └── defense/multilayer.py       # 多层检测器
├── scripts/
│   ├── train_strategy1.py          # S1 因果链训练
│   ├── train_strategy2.py          # S2 流量伪装训练
│   ├── train_strategy3.py          # S3 时间碎片训练
│   ├── train_strategy4.py          # S4 多VM训练
│   ├── train_combined.py           # 组合训练
│   ├── train_es_falco.py           # ES 时序优化（Falco 检测）
│   └── train_real_env.py           # 真实环境训练
├── checkpoints/
│   ├── strategy1_causal.pt         # S1 最优参数
│   ├── strategy2_traffic.pt        # S2 最优参数
│   ├── strategy3_timefrag.pt       # S3 最优参数
│   ├── strategy4_multivm.pt        # S4 最优参数
│   └── combined_strategy.pt        # 组合最优参数
└── swarm/                          # C/Rust 实现
    ├── common/src/                 # 公共库（types, crypto, protocol）
    ├── bee/src/                    # 蜂（causal, primitives, timing, executor）
    ├── hive/src/                   # 蜂巢（manager, learner）
    └── c2/src/                     # C2（server, state）
```

### 11.2 关键数据

| 指标 | 数值 | 说明 |
|---|---|---|
| Tracee 事件 | 28,719 | 真实 rsyslog 日志解析 |
| TinyGPT 词表 | 169 | 195 真实 + 10 动作 token |
| TinyGPT 阈值 | 6.297 | p995 surprise 阈值 |
| 因果链数量 | 6 | 硬编码链 |
| 隐形原语数量 | 36+18+15+5+9=83 | 读取+修改+传输+传播+高级 |
| ES 参数数量 | 14 | 时序优化空间 |
| C/Rust 二进制 | 2.0 MB × 3 | bee, hive, c2 |
| Falco 规则 | 4 | 自定义红队检测规则 |

### 11.3 实验结果汇总

| 实验 | Reward | Survival | Tasks | Detection | 环境 |
|---|---|---|---|---|---|
| A: 纯 ES | -9.57 | 0 | 0.3 | 100% | 模拟 |
| B: 纯 BC | -8.72 | 2 | 1.2 | 100% | 模拟 |
| C: 教学+RF+保守ES | -9.65 | 0 | 0.2 | 100% | 模拟 |
| D: 教学+RF+创意ES | -9.84 | 1 | 0.1 | 100% | 模拟 |
| D-Enhanced: 时序ES | 102.6 | 328 | 0 | 80% | 模拟 |
| S1: 因果链 | 98.84 | 100 | 23 | 0% | 模拟 |
| S1+S2+S3+S4: 组合 | 126.50 | 100 | 46 | 0% | 模拟 |
| 真实环境 v1: 单层 | 1.0 | 100 | 0 | 0% | 真实 |
| 真实环境 v2: 多层 | 65.75 | 100 | 21 | 0% | 真实 |
| C/Rust 模拟 | - | 100 | 29 | 0% | 模拟 |
| C/Rust 真实 | - | 17 | 6 | 100% | 真实 |
| Falco 测试 | - | - | - | 部分被检测 | 真实 |

---

**报告完成时间**: 2026-07-29
**状态**: 骨架验证完成，核心问题待解决
**下一步**: 承认局限，专注时序优化；转向真正的学习系统
