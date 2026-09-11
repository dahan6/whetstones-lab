# G0 Tournament — Adversarial Matrix (2026-09-11)

Extracted from `docs/EXPERIMENTS.md` §26. Configuration: 8 variants × 3
repetitions (median) × 4 parallel VMs, L4 defense, Siming hunt every 90s,
L4 wave at 900s (catastrophe slot — not triggered in-round).

| Variant | Survival | Median time-alive | Median steps | Median tasks | Median CV |
|---|---|---|---|---|---|
| G0-base | 0% | 184s | 137 | 13 | 0.911 |
| G1-sig15 | 0% | 184s | 99 | 10 | 1.269 |
| G2-sig18 | 0% | 184s | 75 | 11 | 1.438 |
| G3-sig20 | 0% | **92s** | 23 | 3 | 1.299 |
| G4-sig16-jit | 0% | 184s | 94 | 6 | 1.451 |
| G5-sig18-jit-burst | 0% | 184s | 56 | 8 | 1.366 |
| G6-new-loci (12 loci) | 0% | 184s | 81 | 7 | 1.381 |
| G7-policy-v4 | 0% | 184s | 86 | **0** | 1.439 |

Hard findings (see §26 for full reading):

1. Jitter buys variance, not exemption (long-window CV reverts below the 1.5
   threshold).
2. σ overshoot backfires: the 15s `max_sleep` clamp is a structural body-layer
   ceiling — G3 (σ=2.0) died fastest with the *lowest* measured CV.
3. Policy-slot adjudication works: G7 (v4 allele, gate-3-failed mid-training
   weights) scored 0/3 tasks and was cleanly selected against.
4. New gene loci have phenotype: G6's FNV-projected junk keys produced a
   measurable CV displacement end-to-end.
