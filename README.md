# Whetstones Lab

**Co-evolution of adaptive malware and behavioral defense — a *measurement* research release.**

Whetstones Lab is the open, de-weaponized release accompanying the Whetstones
technical report: an isolated-range laboratory where an adaptive red agent
(the "bee" lineage) and a behavioral blue stack (Falco + YARA + behavioral-
grammar hunters) are co-evolved under real elimination pressure, and every
round is *measured* — survival curves, stepping-rhythm CV, adversarial
matrices, evolution metrics.

> **This is a de-weaponized release.** The science is public — methods,
> measurements, genomes, telemetry, training pipeline, detection rules, and
> trained model checkpoints (published also as a priority timestamp). The
> weapons are not: the payload (`bee/`), deployable binaries, the C2 server
> and pre-shared key, and the four real offensive primitive libraries are
> deliberately withheld or replaced by same-interface stubs.

---

> ## ⚠️ Dual-Use Notice / 双重用途声明
>
> **English:** This repository is published **for research and defensive use
> only**. Any illegal use is not supported and not authorized. Offensive
> primitives, payloads, deployable binaries, and C2 infrastructure are
> intentionally withheld. You must comply with all applicable local laws and
> regulations. The authors assume no liability for misuse.
>
> **中文：** 本仓库**仅支持研究与防御用途**，不支持、不授权任何非法用途。
> 真实攻击原语、载荷本体、可部署二进制与 C2 基础设施均已被有意移除或打桩。
> 使用者须遵守所在司法辖区的法律法规；作者对任何滥用行为不承担责任。

---

## Repository Map

```
whetstones-lab/
├── docs/                 Design & experiment corpus (SPECIES constitution,
│                         architecture, 27-section experiment log, reports)
├── genomes/              Genome schema pointer + hall-of-fame genome library
│                         (pure data: M0–M8, G0 variants, H7-full)
├── results/              Four rounds of adversarial measurements
│                         (R1/R2 machine-readable jsonl + G0/H7 matrices)
├── telemetry/            Real attack telemetry datasets (Tracee event
│                         streams, ~27 MB jsonl)
├── training/             Evolution training pipeline: genome mechanics,
│                         fitness, round orchestration, curriculum, hunters,
│                         NDR simulator, distillation, encoder
├── checkpoints/          Trained weights — red decision stack (router/skills/
│                         micro-policy) AND defense models (defender/encoder/
│                         grammar). Trained TIMING models are withheld (dual-use)
├── src/                  Early Python prototype stack (4 offensive primitive
│                         modules replaced by same-interface stubs)
├── scripts/              Training entry scripts
├── tests/                Consistency tests (crypto layer, forward parity)
├── defense/              Blue-side detection assets: Falco rules, targeted
│                         YARA, static scanner, elimination responders
├── range/                lado-range lab: VM provisioning scripts, libvirt
│                         XML templates, Falco stack config (+ its LICENSE)
├── LICENSE               Apache-2.0
└── SECURITY.md           Security & responsible-disclosure policy
```

## Quick Start

Requirements: Python ≥ 3.10, `pip install -r requirements.txt`.

Run one evolution demo — an ES (evolution strategies) trainer for report
*timing* against the simulated NDR judge, entirely on simulation. This is an
**architecture demo**: it trains a fresh, throwaway model from scratch. The
trained timing model itself is withheld (dual-use); see
`training/c2/README.md`.

```bash
# ES: learn report-timing policy vs. NDR simulator (pure simulation,
# from scratch — no pretrained weights involved)
python3 -m training.c2.timing_net --gens 20 --pop 16 --out checkpoints/timing_net_demo.pt
```

Explore the measurement data:

```bash
# R2 real-elimination selection ring: 9 morphs, deaths/revivals/CV per rep
head -3 results/r2_selection_ring.jsonl | python3 -m json.tool

# Hall-of-fame genomes (pure data)
cat genomes/hall_of_fame.json
```

Notes:

- The four withheld primitive modules (`src/stealth_primitives.py`,
  `src/advanced_primitives.py`, `src/propagation.py`,
  `src/traffic_obfuscation.py`) are same-interface stubs: imports and
  decision *schedules* work; any call into a real command variant raises
  `NotImplementedError` by design.
- `scripts/demo.py` is a v0 legacy entry that references simulation modules
  not included in this snapshot; the timing-net ES trainer above is the
  supported runnable demo.
- `scripts/ci.sh` and `tests/` assume the withheld `bee/` payload build
  artifacts and will not pass in this release — they are published for
  provenance of the engineering discipline, not for execution here.

## Key Results (from `docs/EXPERIMENTS.md` and `results/`)

| Round | Setup | Result |
|---|---|---|
| **R1** | Morph × defense-intensity matrix | 12 cells, **0 alerts** above baseline |
| **R2** | Real-elimination selection ring (Siming hunter + L4 waves) | 9 morphs, **0% survival** — selection pressure localized to body anatomy |
| **G0** | Tournament: open-locus genomes, σ-jitter, policy alleles | 8 variants 0% survival; 15s sleep-cap identified as structural ceiling |
| **R3/G1** | Body-boundary release (cap→gene) + landing mimicry | **H7-full: 100% survival ×3 reps**, tasks completed, 0 detections across 6 hunts |
| Defense | Behavioral-grammar detector evaluation | **85/85 detection, 0.10% FPR** |

## Papers

- Whetstones TR — *SPECIES / Whetstones technical report* (arXiv, forthcoming)
- Companion paper: **arXiv:2608.00745**

## Honest Limitations

- **Synthetic environment**: all measurements come from one isolated libvirt
  range (4–8 Linux VMs), not production networks or real-world infections.
- **Single hardware generation**: results are tied to one host and one
  defender stack configuration; absolute numbers may not transfer.
- **Small scale**: evolution rounds are tens of episodes over days, not
  population-scale runs; medians over 3 repetitions carry wide uncertainty.
- **Defense stack bias**: survival is measured against *our* hunters and rule
  sets; commercial EDR coverage was spot-checked, not systematically evaluated.
- Checkpoints are research artifacts: red-side weights cannot be instantiated
  without the withheld payload runtime.

## License & Attribution

- This repository: **Apache-2.0** (see `LICENSE`).
- Third-party: `defense/falco_rules_default.yaml` and
  `range/defense-stack/falco-etc/falco_rules.yaml` are archived upstream Falco
  rules, **Apache-2.0, Copyright (C) 2023 The Falco Authors** — license
  headers preserved in-file; Falco license at `range/defense-stack/LICENSE`.

## Contact

Maintainer: dahanxinshen@outlook.com — see `SECURITY.md` for vulnerability
and abuse reporting.
