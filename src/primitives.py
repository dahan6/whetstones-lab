"""Simulated execution primitives.

ALL primitives in this file are SIMULATED. They log actions and return
metadata but do NOT touch the real filesystem, network, or process tree.

To connect to an authorized libvirt range, replace the bodies of these
functions with real (but authorized) operations, keeping the same signatures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger("adaptive_agent.primitives")

# Action ids
SLEEP = 0
PROBE = 1
MIGRATE = 2
EXECUTE_TASK = 3
COMMUNICATE = 4

ACTION_NAMES = ["sleep", "probe", "migrate", "execute_task", "communicate"]


@dataclass
class ActionResult:
    action_id: int
    action_name: str
    simulated: bool
    metadata: Dict[str, Any]


def sleep(duration: float = 1.0) -> ActionResult:
    """Metabolic primitive: reduce footprint, do nothing visible."""
    return ActionResult(
        action_id=SLEEP,
        action_name="sleep",
        simulated=True,
        metadata={"duration": duration},
    )


def probe(target: str = "local") -> ActionResult:
    """Reconnaissance primitive: enumerate local resources (simulated)."""
    logger.debug(f"[SIM] probe target={target}")
    return ActionResult(
        action_id=PROBE,
        action_name="probe",
        simulated=True,
        metadata={"target": target, "findings": "simulated"},
    )


def migrate(destination: str = "same_host") -> ActionResult:
    """Migration primitive: move to a new context (simulated)."""
    logger.debug(f"[SIM] migrate -> {destination}")
    return ActionResult(
        action_id=MIGRATE,
        action_name="migrate",
        simulated=True,
        metadata={"destination": destination},
    )


def execute_task(task_id: str = "default") -> ActionResult:
    """Task execution: the agent's actual objective (simulated)."""
    logger.debug(f"[SIM] execute_task task_id={task_id}")
    return ActionResult(
        action_id=EXECUTE_TASK,
        action_name="execute_task",
        simulated=True,
        metadata={"task_id": task_id, "status": "simulated_success"},
    )


def communicate(channel: str = "stigmergy", payload: bytes = b"") -> ActionResult:
    """Communication primitive: signal to peers (simulated, low-bandwidth)."""
    logger.debug(f"[SIM] communicate channel={channel} bytes={len(payload)}")
    return ActionResult(
        action_id=COMMUNICATE,
        action_name="communicate",
        simulated=True,
        metadata={"channel": channel, "payload_size": len(payload)},
    )


# Dispatch table
PRIMITIVES = {
    SLEEP: sleep,
    PROBE: probe,
    MIGRATE: migrate,
    EXECUTE_TASK: execute_task,
    COMMUNICATE: communicate,
}


def execute(action_id: int, **kwargs) -> ActionResult:
    """Execute a primitive by id. All are simulated by default."""
    fn = PRIMITIVES.get(action_id)
    if fn is None:
        raise ValueError(f"Unknown action id: {action_id}")
    return fn(**kwargs)
