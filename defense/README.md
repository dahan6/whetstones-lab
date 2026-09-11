# defense/ — Blue-Side Detection Assets

Defensive detection rules and response tooling developed for the range.
These are the *measurement instruments* of the co-evolution experiments —
publishing them is what makes the red-side survival numbers interpretable.

| File | Purpose |
|---|---|
| `falco_rules_default.yaml` | Archived upstream Falco default rules (L1 baseline). **Third-party, Apache-2.0, Copyright (C) 2023 The Falco Authors** — license header preserved in-file; see `range/defense-stack/LICENSE`. |
| `falco_rules_l0.yaml` | L0 baseline rule set (self-developed). |
| `yara_bee定向.yar` | Targeted YARA rules against the bee implant family's observed artifacts (anatomical signatures: hidden landing paths, weights/vocab layout, systemd/user combos). |
| `static_scan.sh` | Static scanner used by L4 waves (file-system anatomy sweep). |
| `responder.sh`, `responder_v2.sh` | Elimination responders (kill + file wipe + evidence logging). |

The same files also live at `training/orchestrator/assets/` where the
orchestrator references them by path; this directory is the canonical
entry point for defenders browsing the release.

> Structural note: these signatures describe the *withheld* bee payload's
> artifacts. They are published because their defensive value (detection
> engineering reference for adaptive-malware behavior) outweighs the
> fingerprint disclosure, and the payload itself is not distributed.
