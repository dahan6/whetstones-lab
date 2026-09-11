#!/bin/bash
# bee CI：编译检查 + Rust 测试 + Python 一致性 + 体积基线
# 用法: bash scripts/ci.sh
set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
PASS=0
FAIL=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ $1${NC}"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══ Phase 1: Rust 编译 ═══"
cd bee
cargo build --release 2>&1 | grep -E "^error" && FAIL=1 || true
check "cargo build --release"
cd ..

echo "═══ Phase 2: Rust 测试 ═══"
cd bee
cargo test --release --test integration --bin sysmon 2>&1 | grep "test result: ok" > /dev/null
check "Rust 集成测试 (integration + sysmon unit)"
cd ..

echo "═══ Phase 3: Python 一致性 (Gate 5) ═══"
python3 -m pytest tests/ -q 2>&1 | grep "passed" > /dev/null
check "Python 测试套件 (forward + crypto)"

echo "═══ Phase 4: 体积基线检查 ═══"
BEE_BIN="bee/target/release/sysmon"
if [ -f "$BEE_BIN" ]; then
    ACTUAL=$(stat --format=%s "$BEE_BIN")
    BASELINE=$(grep "^sysmon" .size_baseline | awk '{print $2}')
    THRESHOLD=$((BASELINE + BASELINE / 10))
    if [ "$ACTUAL" -le "$THRESHOLD" ]; then
        echo -e "${GREEN}✓ sysmon 体积 ${ACTUAL}B ≤ 基线+10% (${THRESHOLD}B)${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ sysmon 体积 ${ACTUAL}B > 基线+10% (${THRESHOLD}B)${NC}"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "${RED}✗ sysmon 二进制不存在${NC}"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "═══ 结果: ${PASS} passed, ${FAIL} failed ═══"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
