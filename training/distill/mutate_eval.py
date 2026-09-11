"""变形质量自检+评估（P2.3）：确定性自检为出厂硬保证，统计分析为参数反馈。

两件套：
  1. selfcheck(变体) → YARA定向/熵/strings敏感词 三层确定性检查（出厂门禁）
  2. eval_batch(N) → 产 N 个变体跑自检，统计失败参数的分布特征（反馈变形器）

用法:
  python3 -m training.distill.mutate_eval check /tmp/mutant_101
  python3 -m training.distill.mutate_eval batch 100
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

YARA_RULES = Path("training/orchestrator/assets/yara_bee定向.yar")
MUTATE = Path("bee/target/release/mutate")
MASTER = Path("bee/target/release/sysmon")

SENSITIVE = re.compile(
    rb"(\.cache/\.sysmon|systemd/user|systemctl --user enable|pgrep -f|"
    rb"weights --daemon|/dev/shm/\.|crontab|at now \+|pool_fc\.weight|"
    rb"event_fc\.weight|watchdog|kill_chain)")


def entropy(data: bytes) -> float:
    c = Counter(data)
    n = len(data)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def selfcheck(path: str) -> tuple[bool, dict]:
    """三层确定性自检。返回 (是否通过, 明细)。"""
    detail: dict = {}
    r = subprocess.run(["yara", str(YARA_RULES), path],
                       capture_output=True, text=True)
    detail["yara_hits"] = len([l for l in r.stdout.splitlines() if l.strip()])
    data = open(path, "rb").read()
    detail["entropy"] = round(entropy(data), 2)
    detail["sensitive"] = len(SENSITIVE.findall(data))
    ok = (detail["yara_hits"] == 0 and detail["entropy"] < 7.0
          and detail["sensitive"] == 0)
    return ok, detail


def eval_batch(n: int) -> None:
    """产 N 个变体跑自检，输出通过率与失败分布（变形参数反馈）。"""
    passed, failed = 0, []
    for seed in range(1, n + 1):
        out = f"/tmp/eval_mutant_{seed}"
        r = subprocess.run([str(MUTATE), str(MASTER), out, str(seed)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((seed, "mutate_error"))
            continue
        ok, detail = selfcheck(out)
        if ok:
            passed += 1
        else:
            failed.append((seed, detail))
        Path(out).unlink(missing_ok=True)
    print(f"批量 {n}: 通过 {passed} ({passed/n*100:.0f}%)")
    for seed, why in failed[:10]:
        print(f"  失败 seed={seed}: {why}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        ok, d = selfcheck(sys.argv[2])
        print(f"{'PASS' if ok else 'FAIL'} {d}")
        sys.exit(0 if ok else 1)
    elif len(sys.argv) >= 3 and sys.argv[1] == "batch":
        eval_batch(int(sys.argv[2]))
    else:
        print(__doc__)
