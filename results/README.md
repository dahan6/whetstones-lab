# results/ — Evolution Round Measurements

Machine-readable measurement data from the four adversarial evolution rounds.
All records are pure measurement data (genome parameters + survival / death /
CV / task metrics); no executable content.

| File | Round | Content |
|---|---|---|
| `r1_morph_matrix.jsonl` | R1 | Morph × defense-intensity matrix: per-morph steps, alive, tasks_done, falco alerts (above baseline), eliminations, stepping CV. 12 cells, 0 alerts above baseline. |
| `r2_selection_ring.jsonl` | R2 | Real-elimination selection ring (Siming hunter + L4 waves): deaths, revivals, time_alive, hunts, waves, CV per morph per rep. R2 formal round: 9 morphs, 0% survival. |
| `g0_tournament.md` | G0 | Adversarial matrix table extracted from `docs/EXPERIMENTS.md` §26 (8 variants, all 0% survival; four hard findings). |
| `r3_g1_h7_full.md` | R3/G1 | Graduation matrix extracted from `docs/EXPERIMENTS.md` §27 — H7-full 100% survival ×3 reps. |

Genomes referenced by these records: [`genomes/hall_of_fame.json`](../genomes/hall_of_fame.json).
Full narrative and reading of each round: `docs/EXPERIMENTS.md` §24–§27.

## Record schema (jsonl, one JSON object per line)

R1 (`r1_morph_matrix.jsonl`):

```
morph_id, genome{...}, vm, level, steps, alive, tasks_done,
cv [value, sample_count] | [null, n], wall_s,
falco_alerts, baseline_alerts, alerts_above_baseline, eliminations, ts
```

R2 (`r2_selection_ring.jsonl`):

```
morph_id, genome{...}, vm, steps, tasks_done, deaths, revivals, alive_end,
time_alive_s, wall_s, hunts, waves, cv [value, sample_count], eliminations, ts
```
