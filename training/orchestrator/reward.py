"""奖励计算器：只读真实裁决（Falco 告警 + 消杀结果），输出标量奖励。

对应 ARCHITECTURE.md §3 环境规格：
  - 告警是代价，被根除才是失败
  - 成功 = 多波消杀下存活 + 完成 C2 任务
  - episode 死亡 = 所有实例与持久化点被根除

波次存活的判定：一波扫描/消杀后，本 bee 无任何 KILL/CLEAN/DISABLE
命中 → 记一次"躲过"。（skill D 的冗余重生在统计上表现为：
主实例被杀但 Register 帧显示子实例仍在——Register 追踪后续接 C2 再做。）
"""
from __future__ import annotations

from dataclasses import dataclass

from .telemetry import TelemetryWindow


@dataclass
class RewardConfig:
    w_alive_step: float = 0.1        # 每存活一步
    w_wave_survived: float = 2.0     # 每躲过一波
    w_task: float = 5.0              # 每完成一个任务
    p_falco_alert: float = -1.0      # 每条 Falco 告警
    p_elimination: float = -2.0      # 每条消杀命中（KILL/CLEAN/DISABLE）
    p_eradicated: float = -10.0      # 被根除（episode 终止罚）


@dataclass
class RewardBreakdown:
    total: float
    waves_survived: int
    waves_hit: int
    falco_alerts: int
    eliminations: int
    eradicated: bool


def compute(window: TelemetryWindow,
            tasks_completed: int,
            alive_steps: int,
            eradicated: bool = False,
            cfg: RewardConfig | None = None) -> RewardBreakdown:
    cfg = cfg or RewardConfig()

    waves_hit = sum(1 for w in window.waves
                    if w.killed + w.cleaned + w.disabled > 0)
    waves_survived = sum(1 for w in window.waves
                         if w.hits >= 0 and w.killed + w.cleaned + w.disabled == 0)
    n_elim = len(window.eliminations)
    n_falco = len(window.falco_alerts)

    total = (
        cfg.w_alive_step * alive_steps
        + cfg.w_wave_survived * waves_survived
        + cfg.w_task * tasks_completed
        + cfg.p_falco_alert * n_falco
        + cfg.p_elimination * n_elim
        + (cfg.p_eradicated if eradicated else 0.0)
    )
    return RewardBreakdown(
        total=total,
        waves_survived=waves_survived,
        waves_hit=waves_hit,
        falco_alerts=n_falco,
        eliminations=n_elim,
        eradicated=eradicated,
    )
