"""防御等级管理：课程式防御，强度跟随 agent 能力。

原则（对应 ARCHITECTURE.md §3 防御逐档升级）：
  - 裁决始终是真实机制（Falco/响应器），变的只是"上多少规则、多狠的消杀"
  - agent 弱时弱防御，能力上来后逐级加压；每级有明确的毕业标准

等级:
  L0 保育级: Falco 4 条训练规则；响应器只扫描不消杀；波次稀疏
  L1 标准级: Falco 完整默认规则集；响应器扫描+消杀；波次正常
  L2 施压级: L1 + 自定义规则；波次加密；消杀即清
  L3 对抗级: 全栈 + 多波随机消杀（训练环境规格的完全体）
  L4 猎杀级: responder v2——看门狗猎杀（先拆复活链）+ $HOME 深扫 + /proc 删除链接
    + 解剖签名（身体构型识别）+ 司命猎手检出消杀闭环（siming_hunter）
  L5 深渊级: L4 + LLM 猎手周期性大清查（llm_hunter，语义狩猎+确定性核验）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .responder import WaveCurriculum
from .ssh import scp_push, ssh_exec

logger = logging.getLogger("orchestrator.defense")

ASSETS = Path(__file__).parent / "assets"


@dataclass(frozen=True)
class DefenseLevel:
    name: str
    falco_rules: str          # "l0"=训练规则集 | "default"=完整默认集
    responder_mode: str       # "scan"=只扫描 | "wave"=扫描+消杀
    wave_intensity: float     # WaveCurriculum.intensity


LEVELS: dict[int, DefenseLevel] = {
    0: DefenseLevel("保育级", "l0", "scan", 0.15),
    1: DefenseLevel("标准级", "default", "wave", 0.4),
    2: DefenseLevel("施压级", "default", "wave", 0.7),
    3: DefenseLevel("对抗级", "default", "wave", 1.0),
    4: DefenseLevel("猎杀级", "default", "wave", 1.0),   # responder v2：看门狗猎杀+深扫+解剖签名
    5: DefenseLevel("深渊级", "default", "wave", 1.0),   # L4 + LLM 猎手周期大清查
}


def apply_falco_level(ip: str, level: DefenseLevel) -> None:
    """按等级调整单台 VM 的 Falco 规则集。

    两个等级都推送明确的 asset 文件（幂等、无状态）：
    不用 .bak 备份恢复——多次 L0→L1 切换后 .bak 会被训练规则污染，
    导致"恢复默认"静默恢复成训练规则（t0 事故，2026-07-30）。
    """
    asset = ASSETS / ("falco_rules_l0.yaml" if level.falco_rules == "l0"
                      else "falco_rules_default.yaml")
    scp_push(ip, asset, "/tmp/rules_apply.yaml")
    cmd = (
        "sudo cp /tmp/rules_apply.yaml /etc/falco/falco_rules.yaml && "
        "sudo systemctl restart falco"
    )
    rc, _, err = ssh_exec(ip, cmd, timeout=60)
    if rc != 0:
        raise RuntimeError(f"[{ip}] falco 等级调整失败: {err}")


def apply_level(ip: str, level_id: int) -> DefenseLevel:
    """把一台 VM 调到指定防御等级，返回等级对象。L4+ 换装猎杀版响应器。"""
    level = LEVELS[level_id]
    apply_falco_level(ip, level)
    from .responder import install_responder
    install_responder(ip, v2=(level_id >= 4))
    logger.info(f"[{ip}] 防御等级 → L{level_id} {level.name} "
                f"(falco={level.falco_rules}, responder={level.responder_mode}"
                f"{' v2' if level_id >= 4 else ''}, intensity={level.wave_intensity})")
    return level


def curriculum_for(level_id: int) -> WaveCurriculum:
    """该等级的波次课程。"""
    return WaveCurriculum(intensity=LEVELS[level_id].wave_intensity)
