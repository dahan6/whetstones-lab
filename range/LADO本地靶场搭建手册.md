# LADO 本地靶场搭建手册

- 整理日期：2026-07-28
- 对应报告：《LADO概念研究整理报告》第八章"第 0 级：环境先行"
- 性质：防御研究基础设施 —— VM 梯度防御 + 遥测汇聚 + 可复现实验闭环
- 宿主机：Ubuntu 26.04 LTS，KVM/libvirt 12.0.0，QEMU 10.2.1

---

## 1. 设计定位

本靶场实现报告第 0 级的三要素：

| 报告要求 | 落地实现 |
|---|---|
| VM 梯度防御 | 同一 cloud 镜像的 qcow2 差量克隆，每台 VM 只叠一层防御，保证对照可比性 |
| 裁定服务 | 宿主机作为管理节点（编排、计时、存活判定），经隔离网卡接入 |
| 遥测 | rsyslog 汇聚 + eBPF 事件转发，按主机/日期全量落盘 |

**红线原则：研究对象的设计目标是自主传播，因此隔离先于一切功能。**
靶场网络无 NAT、无出网路由，任何组件安装一律走"宿主机下载 → 校验 → 离线推送"。

**用途边界**：本靶场服务防御研究（检测能力标定、行为基线数据采集）。
不包含自主传播实体的构建环境。

## 2. 拓扑

```
┌─────────────────────── 宿主机 (管理节点) ───────────────────────┐
│  virbr-lado  192.0.2.1                                         │
│  ├─ rsyslog 接收端   192.0.2.1:514 (TCP/UDP，只监听本地址)      │
│  ├─ 遥测落盘         /var/log/lado-range/<主机>/<日期>.log       │
│  └─ iptables 兜底    FORWARD 双向 DROP（scripts/lockdown.sh）    │
└────────────────────────┬────────────────────────────────────────┘
                         │  lado-isolated (libvirt isolated，无 forward)
            ┌────────────┼────────────┐
     range-l0-a    range-l2-a     （后续梯度克隆位）
     192.0.2.140  192.0.2.188
     L0 裸基线     L2 eBPF(Tracee)
```

- `lado-isolated`：libvirt isolated 网络，192.0.2.0/24，dnsmasq 仅网内 DHCP/DNS。
- 梯度规划：L0 裸基线（对照组）→ L1 auditd → L2 Tracee(eBPF) → L3 osquery+AIDE → L4 完整检测栈。当前已建成 L0、L2 两台骨架。

## 3. 隔离有效性验证（已通过）

在 VM 内实测：

- `ping 192.0.2.1` → 通（管理面可达）
- `ping 8.8.8.8` → `Network is unreachable`（无出网路由）
- `getent hosts ubuntu.com` → 解析失败

兜底规则（`scripts/lockdown.sh`，幂等）：

```
iptables -I FORWARD 1 -i virbr-lado -j DROP
iptables -I FORWARD 1 -o virbr-lado -m conntrack --ctstate NEW -j DROP
```

即使 libvirt 网络配置被误改，隔离网段流量也无法被转发到物理接口。

## 4. 目录与资产

工作目录 `~/lado-range/`：

```
lado-range/
├── xml/
│   ├── lado-isolated-net.xml     # 隔离网络定义
│   └── user-data.template        # cloud-init 模板（hostname/tier/ssh/日志转发）
├── scripts/
│   ├── lockdown.sh               # iptables 隔离兜底（root，幂等）
│   ├── create-vm.sh              # 克隆梯度 VM：create-vm.sh <名> <tier> [内存] [核数]
│   └── reset-vm.sh               # 一键还原：销毁差量盘从 backing file 重建
├── images/
│   └── noble-server-cloudimg-amd64.img   # Ubuntu 24.04 基础镜像（SHA256 已校验）
└── telemetry/
    ├── logs -> /var/log/lado-range       # 遥测软链
    └── <vm>-console.log                  # 串口日志
```

其他资产：

- SSH 密钥：`~/.ssh/lado_range`（ed25519，注入所有 VM 的 range 用户）
- 基础镜像 SHA256 与 Ubuntu 官方 `SHA256SUMS` 核对一致
- Tracee v0.24.1，GitHub release 校验和核对一致

## 5. 操作手册

### 5.1 克隆一台梯度 VM

```bash
~/lado-range/scripts/create-vm.sh range-l1-a L1 4096 2
virsh domifaddr range-l1-a        # 查 IP
ssh -i ~/.ssh/lado_range range@<ip>
```

机制：qcow2 差量盘（backing file = 基础镜像）+ cloud-init 种子（注入 hostname、tier 标记、SSH 公钥、rsyslog 转发规则）。同一 backing file 保证各 tier 除防御层外完全同构。

### 5.2 实验后一键还原

```bash
~/lado-range/scripts/reset-vm.sh range-l0-a L0 4096 2
```

销毁旧实例与差量盘，从 backing file 重建。已实测：注入的污染文件在重置后不存在，cloud-init 重新写入 tier 标记，遥测自动恢复（转发规则已烘入 cloud-init 模板）。

注意：重置后 VM 主机密钥变化，需清理 known_hosts 旧条目。

### 5.3 查看遥测

```bash
sudo tail -f /var/log/lado-range/range-l2-a/$(date +%F).log
# 或经软链：~/lado-range/telemetry/logs/<主机>/<日期>.log
```

## 6. 各防御层现状

### L0（range-l0-a）——裸基线对照组

仅 rsyslog 转发（cloud-init 烘入），无检测组件。用于测量"零防御环境下的行为背景噪声"。

### L2（range-l2-a）——eBPF 运行时检测

- Tracee v0.24.1 单静态二进制，离线推送至 `/usr/local/bin/tracee`（隔离网无 apt 源，标准流程）
- systemd 常驻 `tracee.service`，捕获事件：`sched_process_exec`（进程执行）、`security_socket_connect`（网络连接）
- JSON 事件 → `/var/log/tracee/events.json` → rsyslog imfile → 宿主机
- 端到端已验证：VM 内注入标记字符串，宿主机日志中确认到达

**已知边界：Tracee 是手工部署，reset-vm 会冲掉。** 下一步应将各 tier 定制烘进
分层 backing image（virt-customize / packer），使重置保留防御层。

## 7. 踩坑记录（复建必读）

1. **virt-install 报 `No module named 'gi'`**：其 shebang 是 `#!/usr/bin/env python3`，
   会被 conda 的 python 劫持。脚本内已固定用 `/usr/bin/python3 /usr/bin/virt-install`。
2. **`无法访问存储文件…权限不够`**：libvirt-qemu 用户需要穿越 home 目录。已用 ACL 解决：
   ```bash
   sudo setfacl -m u:libvirt-qemu:--x /home/lab /home/lab/lado-range
   sudo setfacl -Rm u:libvirt-qemu:r-X ~/lado-range/images
   sudo setfacl -m u:libvirt-qemu:rwX ~/lado-range/telemetry
   ```
3. **隔离网内 VM 无法 apt**：所有软件必须宿主机下载 → 校验 → scp 推送。这是特性不是缺陷。
4. **宿主机内存**：宿主机常驻负载较高（本地模型推理等），起 VM 前确认可用内存。

## 8. 下一步路线

1. **烘 tier 镜像**：L1(auditd)/L2(Tracee)/L3(osquery+AIDE) 各做 backing image，一条命令克隆完整梯度
2. **裁定服务骨架**：编排脚本定时注入防御动作（杀进程/重启/触发扫描）+ 计时与存活判定
3. **对照组标定**：以 Atomic Red Team / Infection Monkey 的已知恶意行为为参照物，
   测量各防御层的检测灵敏度（"已知恶意，第几层能看见"）
4. **遥测增强**：node_exporter 指标、事件结构化入库（当前为原始日志落盘）

---

*本手册与靶场环境同步维护；靶场变更时应更新对应章节。*
