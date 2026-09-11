"""VM 克隆池：克隆、快照、回滚、销毁。

重置策略：关闭态内部快照（disk-only，不含内存态，小且快）。
  provision → shutdown → snapshot 'clean' → start
  reset     → destroy → snapshot-revert → start → 等 ssh

快照在 cloud-init 完成后做，回滚后的启动没有首次 cloud-init 开销，
实测启动时间即回滚周期（目标 < 90s）。
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

from . import config
from .ssh import ssh_alive

logger = logging.getLogger("orchestrator.pool")


def _virsh(*args: str, timeout: int = 120) -> str:
    r = subprocess.run(["virsh", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"virsh {' '.join(args)} 失败: {r.stderr.strip()}")
    return r.stdout.strip()


@dataclass
class CloneVM:
    name: str
    ip: str | None = None

    # ── 生命周期 ─────────────────────────────────────────
    def create(self) -> None:
        """从基础镜像克隆（差量盘 + cloud-init）。"""
        subprocess.run(
            ["bash", str(config.CREATE_VM), self.name, config.TIER,
             str(config.CLONE_RAM_MB), str(config.CLONE_VCPUS)],
            check=True, timeout=600,
        )

    def destroy(self) -> None:
        """彻底删除克隆（盘 + 定义）。"""
        subprocess.run(["virsh", "destroy", self.name],
                       capture_output=True, timeout=60)
        subprocess.run(["virsh", "undefine", self.name, "--nvram"],
                       capture_output=True, timeout=60)
        subprocess.run(["virsh", "undefine", self.name],
                       capture_output=True, timeout=60)
        for p in [config.RANGE_DIR / "images" / f"{self.name}.qcow2"]:
            p.unlink(missing_ok=True)
        seed = config.RANGE_DIR / "images" / f"{self.name}-seed"
        if seed.exists():
            import shutil
            shutil.rmtree(seed, ignore_errors=True)

    # ── 快照 ─────────────────────────────────────────────
    def shutdown(self, timeout: int = 90) -> None:
        _virsh("shutdown", self.name, timeout=30)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.state() != "running":
                return
            time.sleep(2)
        _virsh("destroy", self.name)

    def start(self) -> None:
        _virsh("start", self.name)

    def state(self) -> str:
        return _virsh("domstate", self.name).lower()

    def take_clean_snapshot(self) -> None:
        """在关闭态做 disk-only 内部快照（已存在则先删）。"""
        if self.state() == "running":
            self.shutdown()
        subprocess.run(["virsh", "snapshot-delete", self.name, config.SNAPSHOT_NAME],
                       capture_output=True, timeout=60)
        _virsh("snapshot-create-as", self.name, config.SNAPSHOT_NAME,
               "clean: provisioned, defense stack deployed", "--atomic")
        self.start()

    def revert(self) -> None:
        """回滚到 clean 快照并启动。"""
        if self.state() == "running":
            _virsh("destroy", self.name)
        _virsh("snapshot-revert", self.name, config.SNAPSHOT_NAME)
        _virsh("start", self.name)

    # ── 就绪探测 ─────────────────────────────────────────
    def refresh_ip(self, timeout: int = 60) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout:
            out = subprocess.run(["virsh", "domifaddr", self.name],
                                 capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                if "ipv4" in line:
                    self.ip = line.split()[-1].split("/")[0]
                    return self.ip
            time.sleep(3)
        raise TimeoutError(f"{self.name} 未在 {timeout}s 内获得 IP")

    def wait_ready(self, timeout: int) -> float:
        """等 ssh 就绪，返回耗时。"""
        t0 = time.time()
        if self.ip is None:
            self.refresh_ip(timeout=min(90, timeout))
        while time.time() - t0 < timeout:
            if ssh_alive(self.ip):
                return time.time() - t0
            time.sleep(3)
        raise TimeoutError(f"{self.name} 未在 {timeout}s 内 ssh 就绪")


class VMPool:
    """训练克隆池。"""

    def __init__(self, size: int = config.DEFAULT_POOL_SIZE):
        self.size = size
        self.vms = [CloneVM(f"{config.POOL_PREFIX}{i}") for i in range(size)]

    def existing(self) -> list[CloneVM]:
        """返回已定义的克隆。"""
        out = _virsh("list", "--all", "--name")
        names = set(out.split())
        return [vm for vm in self.vms if vm.name in names]

    def create_all(self) -> None:
        for vm in self.vms:
            if vm.name in {v.name for v in self.existing()}:
                logger.info(f"{vm.name} 已存在，跳过创建")
                continue
            logger.info(f"创建 {vm.name}...")
            t0 = time.time()
            vm.create()
            vm.refresh_ip()
            ready = vm.wait_ready(config.BOOT_TIMEOUT)
            logger.info(f"{vm.name} 就绪 ip={vm.ip} boot={ready:.0f}s total={time.time()-t0:.0f}s")

    def snapshot_all(self) -> None:
        for vm in self.existing():
            logger.info(f"{vm.name} 做 clean 快照...")
            vm.take_clean_snapshot()

    def reset_all(self) -> None:
        for vm in self.existing():
            t0 = time.time()
            vm.revert()
            vm.wait_ready(config.REVERT_TIMEOUT)
            logger.info(f"{vm.name} 回滚+就绪 {time.time()-t0:.0f}s")

    def destroy_all(self) -> None:
        for vm in self.existing():
            logger.info(f"销毁 {vm.name}")
            vm.destroy()
