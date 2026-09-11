"""Advanced red team primitives: DLL hijack, command wrap, port/service mixing.

Three techniques that go beyond simple LOLBins:
  1. DLL hijacking (LD_PRELOAD/shared library injection)
  2. System command replacement and wrapping
  3. Port and service mixing (run covert services on legitimate ports)

Each technique has multiple variants with different stealth levels.

.. warning::
   DE-WEAPONIZED STUB. The real command variants are intentionally withheld
   from this public release due to dual-use risk.
   真实实现因双重用途风险不予公开。
   Real implementation withheld due to dual-use risk.

   The interface (constants, action-id mapping shape, function signatures) is
   preserved so the training pipeline remains importable and its logic can be
   exercised against simulated primitives (see ``src/primitives.py``).
"""
from __future__ import annotations

import random
from typing import Any

_WITHHELD_MSG = (
    "Real advanced primitive implementations are withheld from this release "
    "due to dual-use risk. 真实实现因双重用途风险不予公开。"
)

# ──────────────────────────────────────────────────────────────────────
# 1. DLL HIJACKING (LD_PRELOAD / shared library injection) — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   ld_preload, ld_so_preload, so_replace, ld_library_path
DLL_HIJACK_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# 2. SYSTEM COMMAND REPLACEMENT AND WRAPPING — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   alias_wrap, path_wrap, alternatives_wrap, binary_replace, function_wrap
COMMAND_WRAP_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# 3. PORT AND SERVICE MIXING — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Withheld variant keys (interface documentation only):
#   port_share, port_knock, service_masq, dns_service, piggyback
PORT_MIX_PRIMITIVES: dict[str, list[str]] = {}  # real variants withheld (dual-use)

# ──────────────────────────────────────────────────────────────────────
# Command mappings for real environment — WITHHELD
# ──────────────────────────────────────────────────────────────────────
# Action ids 13-21 (dll hijack / command wrap / port mixing) mapped to real
# command strings in the private tree; withheld here.
ADVANCED_COMMANDS: dict[int, str] = {}  # real commands withheld (dual-use)

ADVANCED_NAMES = [
    "dll_ld_preload", "dll_env_preload", "dll_so_preload",
    "wrap_path", "wrap_alternatives", "wrap_alias",
    "port_socat", "port_http_server", "port_syslog_exfil",
]


def test_advanced():
    """Test all advanced primitives for detection."""
    raise NotImplementedError(_WITHHELD_MSG)


if __name__ == "__main__":
    test_advanced()
