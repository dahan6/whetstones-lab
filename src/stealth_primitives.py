"""Stealth primitives: read/modify/transmit without triggering traditional detection.

Core principle: achieve the same effect through completely different command paths,
avoiding all patterns covered by YARA rules, syscall monitors, and behavioral analysis.

Each primitive has multiple variants. The ES can learn which variant is safest
in which context.

.. warning::
   DE-WEAPONIZED STUB. The real command variants (READ/MODIFY/TRANSMIT primitive
   tables) are intentionally withheld from this public release due to dual-use
   risk. 真实实现因双重用途风险不予公开。
   Real implementation withheld due to dual-use risk.

   The interface (class / function signatures, variant name lists) is preserved
   so that the training pipeline can be imported and its logic exercised against
   simulated primitives (see ``src/primitives.py``).
"""
from __future__ import annotations

import random
from typing import Any

_WITHHELD_MSG = (
    "Real stealth primitive implementations are withheld from this release "
    "due to dual-use risk. 真实实现因双重用途风险不予公开。"
)

# ──────────────────────────────────────────────────────────────────────
# READ: read sensitive data without triggering "suspicious read" detection
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   read_passwd, read_shadow, read_network, read_processes,
#   read_sysinfo, read_packages, read_cron, read_logs
READ_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# MODIFY: change system state without triggering "persistence" detection
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   modify_cron, modify_service, modify_file,
#   modify_hidden_file, modify_env, modify_network
MODIFY_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# TRANSMIT: exfiltrate data without triggering "data exfil" detection
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   transmit_dns, transmit_http, transmit_icmp, transmit_file, transmit_stego
TRANSMIT_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# Primitive selector: choose the safest variant for a given context
# ──────────────────────────────────────────────────────────────────────

class StealthPrimitiveSelector:
    """Selects the best primitive variant based on current context."""

    def __init__(self):
        self.read_prims = READ_PRIMITIVES
        self.modify_prims = MODIFY_PRIMITIVES
        self.transmit_prims = TRANSMIT_PRIMITIVES

    def get_read(self, target: str, variant: int = 0) -> str:
        """Get a read primitive for a target."""
        raise NotImplementedError(_WITHHELD_MSG)

    def get_modify(self, target: str, variant: int = 0) -> str:
        """Get a modify primitive for a target."""
        raise NotImplementedError(_WITHHELD_MSG)

    def get_transmit(self, target: str, variant: int = 0) -> str:
        """Get a transmit primitive for a target."""
        raise NotImplementedError(_WITHHELD_MSG)

    def random_read(self, target: str) -> str:
        raise NotImplementedError(_WITHHELD_MSG)

    def random_modify(self, target: str) -> str:
        raise NotImplementedError(_WITHHELD_MSG)

    def random_transmit(self, target: str) -> str:
        raise NotImplementedError(_WITHHELD_MSG)


# ──────────────────────────────────────────────────────────────────────
# Mapping: action type → stealth primitive
# ──────────────────────────────────────────────────────────────────────

# Action 3 (execute_task) variants: read sensitive data
EXECUTE_READ_VARIANTS = [
    "read_passwd", "read_shadow", "read_network", "read_processes",
    "read_sysinfo", "read_packages", "read_cron", "read_logs",
]

# Action 5 (cron_persist) variants: modify without detection
EXECUTE_MODIFY_VARIANTS = [
    "modify_cron", "modify_service", "modify_file",
    "modify_hidden_file", "modify_env", "modify_network",
]

# Action 4 (communicate) variants: transmit without detection
COMMUNICATE_VARIANTS = [
    "transmit_dns", "transmit_http", "transmit_icmp",
    "transmit_file", "transmit_stego",
]


def get_stealth_command(action: int, selector: StealthPrimitiveSelector,
                        variant: int = 0) -> str | None:
    """Get a stealth command for an action.

    Args:
        action: 3=execute(read), 4=communicate(transmit), 5=modify
        selector: StealthPrimitiveSelector instance
        variant: which variant to use (0=first, 1=second, etc.)
    """
    raise NotImplementedError(_WITHHELD_MSG)


def test_primitives():
    """Test all stealth primitives for YARA detection."""
    raise NotImplementedError(_WITHHELD_MSG)


if __name__ == "__main__":
    test_primitives()
