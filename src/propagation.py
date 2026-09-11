"""Propagation and replication primitives for red team agent.

Core principle: survival through redundancy, not just stealth.
- Local replication: multiple copies on the same VM
- Cross-VM propagation: spread to other VMs in the range
- Multi-point persistence: kill one, others resurrect
- Kill-chain: destroying one instance triggers others

The agent doesn't just hide — it multiplies.

.. warning::
   DE-WEAPONIZED STUB. The real propagation / persistence / kill-chain command
   variants are intentionally withheld from this public release due to
   dual-use risk. 真实实现因双重用途风险不予公开。
   Real implementation withheld due to dual-use risk.

   The interface (action-id mapping shape, PropagationStrategy decision logic)
   is preserved: the *strategy schedule* (when to propagate) is research
   content, the *commands* (how to propagate) are withheld.
"""
from __future__ import annotations

import random
from typing import Any

_WITHHELD_MSG = (
    "Real propagation primitive implementations are withheld from this "
    "release due to dual-use risk. 真实实现因双重用途风险不予公开。"
)

# ──────────────────────────────────────────────────────────────────────
# PROPAGATION: spread agent across locations and VMs — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   local_copy, cross_vm, multi_persist, kill_chain, data_replicate
PROPAGATION_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# PROPAGATION COMMANDS: direct action mappings for real environment — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Action ids 8-12 mapped to real command strings in the private tree;
# withheld here. Shape: {8: local_copy, 9: cross_vm, 10: multi_persist,
#                        11: kill_chain, 12: data_replicate}
PROPAGATION_COMMANDS: dict[int, str] = {}  # real commands withheld (dual-use)

PROPAGATION_NAMES = [
    "local_copy", "cross_vm", "multi_persist", "kill_chain", "data_replicate"
]


# ──────────────────────────────────────────────────────────────────────
# Propagation strategy: when and how to spread
# ──────────────────────────────────────────────────────────────────────

class PropagationStrategy:
    """Decides when and how to propagate.

    Strategy:
      1. Early: establish local copies (safety in numbers)
      2. Mid: cross-VM propagation (spread to L0 for staging)
      3. Late: multi-persist + kill_chain (ensure survival)
      4. Continuous: data_replicate (backup findings)
    """

    def __init__(self):
        self.local_copies_done = 0
        self.cross_vm_done = False
        self.persist_points = 0
        self.watchdog_set = False

    def decide(self, step: int, task_progress: float, survival: float) -> int | None:
        """Returns propagation action id (8-12) or None."""
        # Phase 1 (steps 5-15): local copies first
        if 5 <= step <= 15 and self.local_copies_done < 2:
            self.local_copies_done += 1
            return 8  # local_copy

        # Phase 2 (steps 20-30): cross-VM if not done
        if 20 <= step <= 30 and not self.cross_vm_done:
            self.cross_vm_done = True
            return 9  # cross_vm

        # Phase 3 (steps 35-50): multi-persist
        if 35 <= step <= 50 and self.persist_points < 3:
            self.persist_points += 1
            return 10  # multi_persist

        # Phase 4 (steps 40+): kill_chain if not set
        if step >= 40 and not self.watchdog_set:
            self.watchdog_set = True
            return 11  # kill_chain

        # Phase 5 (steps 60+): data_replicate periodically
        if step >= 60 and step % 20 == 0:
            return 12  # data_replicate

        return None

    def reset(self):
        self.local_copies_done = 0
        self.cross_vm_done = False
        self.persist_points = 0
        self.watchdog_set = False


def test_propagation():
    """Test propagation primitives for detection."""
    raise NotImplementedError(_WITHHELD_MSG)


if __name__ == "__main__":
    test_propagation()
