# genomes/ — Genome Schema & Hall of Fame

Pure data directory. No implementation code.

- **Schema**: the genome encoding (11 gene loci + open-locus FNV projection
  mechanism, morph definitions M6–M8) is specified in
  [`docs/弓形虫-系统文档-v0.md`](../docs/弓形虫-系统文档-v0.md) §2.
- **`hall_of_fame.json`**: population genomes evaluated across the evolution
  rounds —
  - `r1_population`: M0–M5 (R1 morph × intensity matrix)
  - `r2_additions`: M6–M8 (R2 real-elimination selection ring)
  - `g0_tournament_variants`: G0–G7 (G0 tournament, reconstructed from
    `docs/EXPERIMENTS.md` §26)
  - `hall_of_fame.H7-full`: the first fitter morph — 100% survival ×3 reps at
    L4 (reconstructed from `docs/EXPERIMENTS.md` §27)

Per-round measurement records that consume these genomes live in
[`results/`](../results/). The bee-side parser (`Genome::load()`) is part of
the withheld payload and is not published.
