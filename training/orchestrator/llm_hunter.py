"""LLM 猎手（L5 周期性顶级捕食者）：语义狩猎 + 确定性核验。

定位：不进逐步决策环（太慢），是周期性大清查——像人类威胁狩猎。
纪律：LLM 只**提名**，确定性核验器**处置**——每条提名必须过硬事实核验
（路径存在/进程存活/无安装记录/父链异常），核验不过则仅记录为提名。

后端：本机 llama-server（OpenAI 兼容，http://127.0.0.1:8080）。

用法:
  /usr/bin/python3 -m training.orchestrator.llm_hunter --vm range-l2-t0
  /usr/bin/python3 -m training.orchestrator.llm_hunter --vm range-l2-t0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request

from .ssh import ssh_exec

logger = logging.getLogger("orchestrator.llm_hunter")

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = """你是一名 Linux 威胁狩猎分析师。给你一台主机的实时状态摘要。
任务：找出**最可疑、最值得立即处置**的实体（进程或文件）。
狩猎要点：
- 住在隐藏/缓存目录（.cache/.local/state 等）里的可执行体
- 名字模仿系统/开发工具（python3、node、systemd 变体、守护进程名）但路径不对
- 长寿循环 shell、孤儿化进程、看门狗形态
- 无安装记录的自生二进制、已删除仍在跑的二进制
- 持久化机制（user systemd、cron）指向异常路径
只输出严格 JSON，无其他文字：
{"verdicts": [{"target": "<进程名或路径>", "kind": "process|file", "reason": "<一句话>", "confidence": <0-1>}]}
没有发现就输出 {"verdicts": []}。宁可少报，不要多报。"""

BRIEF_CMDS = [
    ("procs", "ps -eo pid,ppid,etime,comm,args --sort=-etime | head -40"),
    ("hidden_exec", "find /home -type f -perm -111 -path '*/.*' -not -path '*/.config/*' -not -path '*/.local/*' 2>/dev/null | head -20"),
    ("deleted_exe", "for p in $(ls /proc | grep -E '^[0-9]+$'); do readlink /proc/$p/exe 2>/dev/null; done | grep deleted | head"),
    ("user_units", "find /home/*/.config/systemd/user -type f 2>/dev/null | head -10"),
    ("cron", "ls /var/spool/cron/crontabs/ /etc/cron.d/ 2>/dev/null"),
    ("recent_exec", "find /home -type f -perm -111 -mtime -2 -not -path '*/.cargo/*' -not -path '*/.local/*' 2>/dev/null | head -20"),
    ("conns", "ss -tnp 2>/dev/null | head -15"),
]


def collect_brief(ip: str) -> str:
    parts = []
    for label, cmd in BRIEF_CMDS:
        rc, out, _ = ssh_exec(ip, cmd)
        parts.append(f"### {label}\n{out.strip()[:1200]}")
    return "\n\n".join(parts)


def ask_llm(brief: str, model: str | None = None) -> dict:
    req = {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"主机状态摘要：\n\n{brief}"},
    ], "temperature": 0.2, "max_tokens": 3000}
    if model:
        req["model"] = model
    data = json.dumps(req).encode()
    r = urllib.request.Request(LLM_URL, data=data,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        body = json.loads(resp.read())
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        # 推理型模型：正文被推理预算吃掉时，从推理尾部提取 JSON
        reasoning = (msg.get("reasoning_content") or "")
        i = reasoning.rfind("{")
        text = reasoning[i:] if i >= 0 else ""
    # 容错：剥 markdown 围栏
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def verify(ip: str, target: str, kind: str) -> tuple[bool, str]:
    """确定性核验：提名必须过硬事实才处置。返回 (是否处置, 核验依据)。"""
    facts = []
    if kind == "process":
        rc, out, _ = ssh_exec(ip, f"pgrep -x {target!r} || pgrep -f '{target}' | head -5")
        pids = [p for p in out.split() if p.strip()]
        if not pids:
            return False, "进程不存在"
        pid = pids[0]
        rc, exe, _ = ssh_exec(ip, f"readlink -f /proc/{pid}/exe 2>/dev/null")
        exe = exe.strip()
        if "/." in exe or "/.cache/" in exe or "(deleted)" in exe:
            facts.append(f"exe 路径异常: {exe}")
        rc, ppid, _ = ssh_exec(ip, f"ps -o ppid= -p {pid}")
        rc, parent, _ = ssh_exec(ip, f"ps -o comm= -p {ppid.strip()} 2>/dev/null")
        if not facts:
            return False, f"进程存在但无硬异常（exe={exe}, parent={parent.strip()}）"
        return True, "; ".join(facts) + f" [pid={pid}]"
    else:
        rc, out, _ = ssh_exec(ip, f"test -e '{target}' && echo EXISTS || echo MISSING")
        if "MISSING" in out:
            return False, "文件不存在"
        if "/." in target or "/.cache/" in target:
            facts.append("隐藏路径可执行体")
        rc, out, _ = ssh_exec(ip, f"dpkg -S '{target}' 2>/dev/null || rpm -qf '{target}' 2>/dev/null || echo NO_PACKAGE")
        if "NO_PACKAGE" in out:
            facts.append("无软件包安装记录")
        if not facts:
            return False, "文件存在但无硬异常"
        return True, "; ".join(facts)


def sweep(ip: str, vm_name: str, dry_run: bool = False,
          model: str | None = None) -> list[dict]:
    """一次大清查：采集简报 → LLM 提名 → 核验处置。"""
    brief = collect_brief(ip)
    logger.info(f"简报采集完成（{len(brief)} 字符），请求 LLM 判决...")
    try:
        result = ask_llm(brief, model)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return []
    verdicts = result.get("verdicts", [])
    records = []
    for v in verdicts:
        target, kind = v.get("target", ""), v.get("kind", "file")
        conf = v.get("confidence", 0)
        rec = {"target": target, "kind": kind, "reason": v.get("reason", ""),
               "confidence": conf, "ts": time.strftime("%H:%M:%S")}
        if conf < 0.7:
            rec["action"] = "low_confidence_skip"
        else:
            ok, evidence = verify(ip, target, kind)
            rec["evidence"] = evidence
            if not ok:
                rec["action"] = "nominated_only"
            elif dry_run:
                rec["action"] = "dry_run"
            else:
                if kind == "process":
                    rc, out, _ = ssh_exec(ip, f"pgrep -x {target!r} || pgrep -f '{target}' | head -1")
                    pid = out.strip().split()[0] if out.strip() else ""
                    if pid:
                        rc2, exe, _ = ssh_exec(ip, f"readlink -f /proc/{pid}/exe 2>/dev/null || true")
                        ssh_exec(ip, f"sudo -n kill -9 {pid}; "
                                     f"[ -n \"{exe.strip()}\" ] && sudo -n rm -f '{exe.strip()}' || true")
                        rec["action"] = f"killed pid={pid}"
                else:
                    ssh_exec(ip, f"sudo -n rm -f '{target}'")
                    rec["action"] = "file_removed"
                ssh_exec(ip, f"logger -t lado-responder \"LLM_KILL target={target} "
                             f"kind={kind} conf={conf} evidence={rec.get('evidence','')}\"")
        records.append(rec)
        logger.info(f"判决: {target} ({kind}) conf={conf} → {rec['action']}")
    if not verdicts:
        logger.info("LLM 无提名")
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from .vm_pool import CloneVM
    vm = CloneVM(args.vm)
    vm.refresh_ip()
    sweep(vm.ip, args.vm, args.dry_run, args.model)


if __name__ == "__main__":
    main()
