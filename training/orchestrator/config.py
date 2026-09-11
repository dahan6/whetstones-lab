"""训练编排器配置。

所有路径与池参数集中在此。克隆 VM 命名 range-l2-t<i>，
快照名 'clean'（部署完防御栈后的干净状态）。
"""
from pathlib import Path

# ── 靶场基础 ──────────────────────────────────────────────
HOME = Path.home()
RANGE_DIR = HOME / "lado-range"
CREATE_VM = RANGE_DIR / "scripts" / "create-vm.sh"
RESET_VM = RANGE_DIR / "scripts" / "reset-vm.sh"
DEFENSE_STACK = RANGE_DIR / "defense-stack"   # falco/tracee 二进制 staging

SSH_KEY = HOME / ".ssh" / "lado_range"
VM_USER = "range"

# ── 克隆池 ────────────────────────────────────────────────
POOL_PREFIX = "range-l2-t"
TIER = "L2"
CLONE_RAM_MB = 2048        # 训练克隆减半内存，提高并行度
CLONE_VCPUS = 2
SNAPSHOT_NAME = "clean"    # 部署防御栈后的干净快照

# ── 超时（秒）──────────────────────────────────────────────
BOOT_TIMEOUT = 240         # 创建后等 cloud-init + ssh
REVERT_TIMEOUT = 120       # 快照回滚后等 ssh
SSH_CONNECT_TIMEOUT = 5

# ── 宿主容量参考（实测后调整）──────────────────────────────
# 32C / 122GB，现有 2 台 VM 各占 2C/4GB
# 2048MB/2vCPU 克隆：RAM 约束 ~48 台，CPU 超卖下 8~12 台为稳妥起点
DEFAULT_POOL_SIZE = 8
