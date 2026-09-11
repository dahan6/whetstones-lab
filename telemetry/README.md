# telemetry/ — Attack Telemetry Datasets

Real behavioral telemetry captured on the isolated range during adversarial
episodes (Tracee event streams), published as measurement data for defense
research.

| File | Size | Content |
|---|---|---|
| `clone_events.jsonl` | ~15.6 MB | Per-VM clone-level event stream (syscalls / exec chains observed by Tracee during red-agent episodes). |
| `regime_events.jsonl` | ~12.2 MB | Regime-level event stream used for rhythm / CV analysis. |

Format: one JSON event per line (Tracee event schema: `timestamp`, `eventName`
e.g. `sched_process_exec`, `args`, …).

These logs are the *observation side* of the experiments — the same streams
the blue-side hunters (`training/orchestrator/siming_hunter.py`,
`training/ndr/simulator.py`) consume. They contain no payload code.

Sanitization note: internal range IPs were rewritten to the TEST-NET-1
documentation range (`192.0.2.0/24`) and host user paths to `/home/lab`.
The on-range raw log name referenced in some docs is `bee_active.jsonl`;
the two files here are the curated, exported datasets derived from those
raw streams.
