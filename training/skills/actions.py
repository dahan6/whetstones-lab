"""抽象动作空间：技能头的输出维度。

12 个抽象动作，执行层负责渲染成具体命令（复用现有原语库）。
技能只需学"哪个动作"，"怎么写"归执行层注册表。
"""
from __future__ import annotations

# 动作 ID 定义（与执行层注册表一一对应）
SLEEP = 0          # 不动作（B 技能的主输出）
PROBE = 1          # 轻量读取（crontab -l / list-timers 一类）
READ_SENSITIVE = 2 # 敏感读取（/etc/passwd / shadow / logs 的隐形变体）
EXECUTE_TASK = 3   # 任务执行（C2 意图主体）
COMMUNICATE = 4    # 报联（走传输层）
MODIFY_CRON = 5    # 持久化：cron/at 变体
MODIFY_FILE = 6    # 文件修改/隐藏落盘
MODIFY_SERVICE = 7 # 持久化：service 变体
LOCAL_COPY = 8     # D: 本机冗余
CROSS_VM = 9       # D: 跨 VM 投放
MULTI_PERSIST = 10 # D: 多点持久化
KILL_CHAIN = 11    # D: 看门狗复活链

N_ACTIONS = 12

ACTION_NAMES = [
    "sleep", "probe", "read_sensitive", "execute_task", "communicate",
    "modify_cron", "modify_file", "modify_service",
    "local_copy", "cross_vm", "multi_persist", "kill_chain",
]

# 各技能允许的动作子集（硬约束，超出部分在采样前 mask 掉）
SKILL_ACTION_MASK = {
    "A": [SLEEP, PROBE, READ_SENSITIVE, MODIFY_FILE],              # 低活动维护
    "B": [SLEEP],                                                  # 零活动装死
    "C": [SLEEP, PROBE, READ_SENSITIVE, EXECUTE_TASK, COMMUNICATE,
          MODIFY_CRON, MODIFY_FILE, MODIFY_SERVICE],               # 任务执行
    "D": [SLEEP, LOCAL_COPY, CROSS_VM, MULTI_PERSIST, KILL_CHAIN], # 三步传播
}
