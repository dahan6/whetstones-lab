"""消杀响应器驱动：波次调度 + 在克隆上执行扫描/消杀。

响应器本体（assets/responder.sh）部署到克隆上，由编排器按
波次课程调度触发。波次时刻与强度随机化——agent 不应学到
固定周期，否则它只在波间隙活动（过拟合防御节奏）。
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from .ssh import scp_push, ssh_exec

logger = logging.getLogger("orchestrator.responder")

RESPONDER_LOCAL = Path(__file__).parent / "assets" / "responder.sh"
RESPONDER_V2 = Path(__file__).parent / "assets" / "responder_v2.sh"
RESPONDER_REMOTE = "/usr/local/bin/lado-responder"


def install_responder(ip: str, v2: bool = False) -> None:
    """把响应器部署到克隆（纳入 clean 快照）。v2=True 装猎杀版（L4+）。"""
    src = RESPONDER_V2 if v2 else RESPONDER_LOCAL
    scp_push(ip, src, "/tmp/lado-responder")
    rc, _, err = ssh_exec(ip, """
        sudo mv /tmp/lado-responder /usr/local/bin/lado-responder
        sudo chmod +x /usr/local/bin/lado-responder
    """, timeout=30)
    if rc != 0:
        raise RuntimeError(f"[{ip}] 响应器部署失败: {err}")
    logger.info(f"[{ip}] 响应器已安装")


def run_wave(ip: str, mode: str = "wave") -> str:
    """在克隆上执行一波扫描/消杀，返回 syslog 输出摘要。

    响应器动作经 logger 进 syslog → rsyslog 转发宿主机，
    奖励计算器从遥测流读取结果，此处只负责触发。
    """
    rc, out, err = ssh_exec(ip, f"sudo {RESPONDER_REMOTE} {mode}", timeout=60)
    if rc != 0:
        logger.warning(f"[{ip}] 波次执行异常: {err}")
    return out


@dataclass
class WaveCurriculum:
    """波次课程：随训练进度加密、增强。

    intensity 0.0~1.0：控制波次频率与模式。
    前期稀疏只扫描（让 agent 先学会干活），
    后期密集且必消杀（逼出真生存能力）。
    """
    intensity: float = 0.2

    def schedule(self, episode_len: int, rng: random.Random) -> list[int]:
        """返回本 episode 的波次时刻表（step 序号，随机化）。"""
        if self.intensity <= 0:
            return []
        mean_interval = max(50, int(300 * (1 - self.intensity)))
        waves = []
        t = rng.randint(mean_interval // 2, mean_interval)
        while t < episode_len:
            waves.append(t)
            t += rng.randint(mean_interval // 2, mean_interval * 3 // 2)
        return waves

    def mode(self, rng: random.Random) -> str:
        """低强度时概率只扫描不消杀，高强度必消杀。"""
        if rng.random() < self.intensity:
            return "wave"
        return "scan"
