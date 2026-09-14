# Rounds Addendum — 2026-09-14/15 (E-L → E-H′)

> Sanitized experiment summaries. Analysis unit and numbers are verbatim;
> infrastructure specifics (hosts, addresses, credentials) are omitted.

## E-L Red Queen Round (4 arms, 80 rounds total)

Confirmatory pre-registered round. Conditions per 10-min round: full blue
stack (90 s hunter patrol, 480 s responder wave, +60 s post-wave rescan).

| Arm | Condition | n | Survival | Death time |
|---|---|---|---|---|
| 1 | H7-full (jitter+burst+silence+landing-camouflage) | 20 | **0/20** | 542–552 s |
| 1 | H5-landing-open | 20 | **0/20** | 544–548 s |
| 2 | H7 + post-wave rescan | 20 | **0/20** | 548–550 s |
| 4 | H4 tmp-landing | 20 | **0/20** | **92–183 s (median 92)** |

- Post-wave rescan: 100% lethal (60/60). Chain hunter: 0 kills / 360 patrols.
  Single L4 wave: 0 kills. Implementation note: rescan applied
  unconditionally → arms 1/2 are one 60-round condition.
- Arm 3 (exact-NLL full fusion, offline): G5-elite escapes CV (1.91/2.04) →
  grammar-face exposure **60.2% / 100%** (UNK/context/rare).
- Arm 4: landing camouflage of the champion is worth **+450 s (5.9×)**.

## E-M Heterogeneous Intrusion (LLM red brain + 7 trojan behavior batteries)

- LLM agent (local model, one shell command per turn): full kill chain in
  **43 s** (foothold 6 s / secret read 31 s / exfil 43 s / persistence 85 s),
  exfil confirmed at sink 4×, not caught in 265 s.
- v1 scorecard: 7/7 trojan behaviors survived — verification step only
  checked `/proc/pid/exe` path; script-based trojans inherit the interpreter
  binary (system path) → structurally exempt.
- v2 (evidence-based authorization: exe+cmdline dual check, beacon-interp):
  **3/3 killed** (reverse-shell first-hunt kill; miner kill+seed-wipe;
  RAT killed by the new connect-beacon face), **0 false positives** on four
  benign role hosts.
- First model-convicted kill in the live loop: rhythm classifier
  `machine_prob = 0.997`.

## E-N Real-Sample Round

- Real XMRig v6.26.0 binary (SHA256-audited official release): beacon face
  kills ×3 (n up to 213 connects / 3 min), first use of beacon-interp
  authorization; 3 instances ultimately survived (PID-resolution gap for
  bare-exec processes — registered fix).
- Real-source Kaiten (compiled from public source): ran, C2 beacon absent —
  source-level debugging pending. Mirai bot build deps missing; Diamorphine
  needs kbuild — both honestly recorded as not run.
- LLM no-hint intrusion (76 moves / 15 min): secret recovered at 40 s (via
  leftover artifact — cleanup-verification lesson), 3 exfil styles, no
  persistence, **not caught**.

## E-H′ Atomic Red Team scorecard (external opponent, replaces CALDERA)

12 cross-tactic public TTPs, all executed successfully:

- v1: **0/12 detected** (vs offline pipeline 22–87% on OpTC — capability
  exists, live wiring is the gap).
- v2 (three wirings: sensitive-read falco rule incl. sudo path, network
  fan-out face, one-shot combo face): **5/12 detected** — the causal
  evidence that offline capability materializes with wiring.

## Calibration & economics

- Burstiness-face FPR: **85.5%** on scheduled-workload benign (CI jobs are
  more machine-like than most beacons) → class-aware retrain **1.8% @ 45%
  recall**. Reference-class scoping is mandatory for the paper claim.
- Axelsson curves (E1 σ-ladder + G5): **floor + slope** structure — discrete
  faces (UNK/rare) give a budget-independent detection floor (59.6%),
  continuous NLL gives the budget-sensitive segment.
- Silence×window matrix (6 depths): all die at the same rescan instant —
  silence evades rhythm detection, not file-scan elimination.

## Known gaps (registered)

falco-alert consumption in the live loop · PID resolution for bare-exec
beacons · meta-scorer proxy slots empty · LLM-opponent detection (network
fan-out + receive-side alerts are wired, retest pending).
