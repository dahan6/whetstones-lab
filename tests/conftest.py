"""公共 fixture：路径、权重加载、bee 二进制路径。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEE_DIR = PROJECT_ROOT / "bee"
BEE_BIN = BEE_DIR / "target" / "release" / "sysmon"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
BEE_WEIGHTS = BEE_DIR / "weights"


@pytest.fixture(scope="session")
def bee_binary():
    """bee/sysmon 二进制路径（需要先 cargo build --release）。"""
    if not BEE_BIN.exists():
        pytest.skip("bee 二进制未编译，先 cd bee && cargo build --release")
    return str(BEE_BIN)


@pytest.fixture(scope="session")
def weights_dir():
    """bee 部署权重目录。"""
    if not (BEE_WEIGHTS / "manifest.json").exists():
        pytest.skip("bee/weights/manifest.json 不存在")
    return str(BEE_WEIGHTS)
