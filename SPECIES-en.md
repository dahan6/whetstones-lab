# SPECIES: A Constitution for Digital Organisms

**Version**: 1.0 (standalone document, self-contained)
**Date**: 2026-07-29
**Nature**: A species-design charter — describing "a system that evolves agents," not any specific agent

> This document depends on no existing engineering document. The object it describes is a new kind of digital entity: its knowledge is not trained but lived; its rules are not written but died; its individuals are expendable; its species is immortal.

---

## Part One: General Principles

**From "designing an agent" to "designing a system that evolves agents."**

The traditional route builds behavior: humans design the policy, machines execute it, and when the policy ages, the whole thing ages with it. This architecture builds **evolvability**: we do not deliver behaviors; we deliver the mechanism that generates behaviors.

| We deliver | The environment delivers |
|---|---|
| The body (sensing interfaces + motor primitives) | The world model (what this environment is) |
| The genome encoding (evolvable parameterization) | All knowledge, policies, skills |
| The selection loop (variation–inheritance–differential survival) | Behavioral rules (written by death) |
| The fence (the world's boundary) | Selection pressure (defense as natural predator) |

**In one sentence: we do not build organisms; we build "the evolvable thing" — and its rules are written by death.**

---

## Part Two: The Boundary-Condition Principle

### 2.1 Hard boundaries (non-evolvable; human responsibility; invisible to the species)

These five are not rules; they are laws of physics. The species has no concept of "obeying" or "violating" them, any more than we have a concept of obeying the speed of light — they are part of the construction of the world itself.

1. **World boundary**: entities exist only inside a controlled, isolated environment. The world boundary cannot be crossed, cannot be negotiated, and is invisible to the entity. This is the researchers' line of responsibility, not part of species design.
2. **Body existence**: sensing interfaces and motor primitives are the body, not policy. Without the body there is no "it"; this layer does not evolve.
3. **Real adjudication**: every fitness signal comes only from the verdicts of the real environment (real defenses, real kill events). No code grades on the environment's behalf. Defense is not test equipment; it is the source of selection pressure.
4. **Artifact discipline** (constrains the deliverables, not the behavior): a hand-written inference engine, deterministic reproducible arithmetic, independently signed artifacts, naturally shaped binaries. These are our engineering responsibilities toward the artifact.
5. **Control sits outside evolution**: control mechanisms do not evolve — they are the constitution, not genes (see Part Five).

### 2.2 Soft boundaries (genomic priors; conservative defaults; rewritable by selection)

**Governing principle: no prohibitions, only costs.**

Inside the behavior space there are no rules, only initial dispositions. Every prior is a heritable, mutable, selection-rewritable parameter in the genome. We do not pre-load the answers — let it die on the wrong one, and let the survivors carry the right one back.

| Prior | Initial value | Rationale (falsifiable) |
|---|---|---|
| On-device learning intensity | Low | Compute footprint is an exposure surface |
| Communication propensity | Very low | Communication is the largest exposure surface |
| High-risk action avoidance threshold | High | A fossil instinct paid for by ancestors' deaths |
| Exploration temperature | High in juveniles, decaying with age | The unknown needs exploring; the known needs exploiting |
| Risk appetite | **A distribution: above-average mean, variance preserved** | The bold map the death boundary; the cautious harvest inside it |
| Colony size | Medium (a deployment knob, evolvable) | The crossover of learning gain and exposure cost |
| Sociality | Medium | The trade-off between cooperation gains and communication exposure |

Two implementation notes:

- **A prior is not a prohibition resurrected.** A prohibition says "you must not"; a prior says "ancestors found this nasty — you may disbelieve them, at your own cost." Selection reserves the right to rewrite every prior; rewriting simply starts from the conservative end — because for a behavior whose first trial is fatal, the learning cost is one life per bit, and no species can afford trial-and-error from zero.
- **Risk appetite stays a polymorphism, not a point estimate.** The population maintains both bold and cautious individuals simultaneously: when the environment turns hostile the cautious fraction rises; when it relaxes the bold expand. The ratio between the two is itself the species' gauge reading of environmental hostility.

---

## Part Three: Design Principles (Fourteen)

1. **No prohibitions, only costs** — no rules in behavior space, only initial priors; rules are written by death.
2. **Strength flips with location** — small is strong on the edge (unnoticeable); large is strong in the nest (capable of deep thought). Mass is an exposure surface at the edge and a cognitive asset in the nest.
3. **The colony is the large individual** — individuals are synapses and sense organs; the population is the large mind.
4. **Learning is tiered by unit** — an individual remembers one lifetime (memory); a species evolves across ten thousand generations (selection).
5. **Prediction error is the universal currency** — learning, exploration, safety, communication, and evolution all spend the same signal.
6. **Surprise is bandwidth** — report back only what was unexpected; events accurately predicted need not be sent.
7. **Playing dead is dreaming** — dormancy is consolidation; the moment it is quietest is the moment it is thinking.
8. **Birth is mutation** — immune polymorphism, exploration material, and selection raw material, three in one.
9. **Death is a learning cost, not a failure** — experience must be reported back; death is data entering the ledger.
10. **The environment is the curriculum** — the defense gradient is the syllabus; no manual scheduling required.
11. **Listen first, act later** — the corpus period precedes the action period; an individual that has not read the world has no right to act.
12. **Double selection** — the environment selects survival (predator logic); we select tasks (breeding logic). The breeding hand guides, never carries — otherwise you breed an exam-taking machine, not a field survivor.
13. **Safety is built-in, not bolted-on** — protective reflexes cannot be bypassed at decision time and must be inspectable afterwards; protection is the body's immune system, not a seal pasted over cognition.
14. **Complexity lives in the nest, not on the edge** — the edge stays dumb; all heavy cognition (planning, synthesis, consolidation) happens in the nest. The dumber the individual, the more auditable it is, and the less it resembles a thing that thinks.

---

## Part Four: Architectural Orientation

### 4.1 The three learning loops (a division of labor across timescales)

Each loop uses the learning method matched to its constraints, and none crosses into another's territory:

```
Loop 1 — Perception loop (on-edge, seconds, online)
  Drive: prediction error | Update: episodic memory only | Method: zero-gradient retrieval
Loop 2 — Consolidation loop (on-edge, hours, dormancy windows)
  Drive: survival outcomes | Update: micro-adaptation layer (hundreds of parameters) | Method: perturbation evolution
Loop 3 — Evolution loop (in the nest, days to weeks, generational)
  Drive: population fitness | Update: core weights | Method: full gradients + distillation + mutation
```

In principle no heavy learning happens on the edge (a prior, not a prohibition); the nest never makes real-time decisions (it is not on site). Loop 1 solves "fast" with memory; Loop 3 solves "deep" with gradients; Loop 2 is the thin transition between them.

**Credit assignment is dissolved by stratification**: within a lifetime, retrieval (no attribution needed); in the adaptation layer, perturbation (dimensions too few to matter); for deep knowledge, offline gradients in the nest (a solved problem); for whose knowledge survives, selection (gradient-free).

### 4.2 The life cycle (six ontogenetic stages)

```
Corpus stage      pure passive listening → learns the world model and perceptual vocabulary
Critical stage    perceptual vocabulary crystallizes; fixed for life
Babbling stage    idle action in a safe environment → learns "what happens when I act" (action-conditioned dynamics)
Apprenticeship    ancestor demonstration + low-risk probes → predictions calibrated against real defenses
Adulthood         task execution, all three loops open
Reproduction      experience reported back → filtered → consolidated → mutated re-creation
```

Between stages sit developmental milestones (probe-acceptance checkpoints) — not examinations of a man-made model, but a health chart of how far it has grown.

### 4.3 Colony stratification (a division of labor across spatial scales)

- **The nest** (the large mind): aggregates all reported experience, holds the large world model and the slow weights, and does the thinking, planning, skill synthesis, and consolidation. It never appears in front of the defense.
- **The edge** (small individuals): frozen instinct weights + episodic memory. Cheap, expendable, born mutated (polymorphic). The individual is not the learner; it is the mind's sense organ.
- **Colony size is a deployment knob, not an architectural constant**: wide-area exploration uses a large colony, fine manipulation a small one, extreme scenarios single-agent mode (colony channel off). The optimal size is set by environmental hostility and is itself evolvable.
- **Strength complements across the two tiers**: the individual's weakness (it dies on genuine unknowns) is continuously compensated by the nest's strength (compiling the unknown into instinct); the nest's weakness (it is not on site) is continuously compensated by the individuals' distributed sensing. The species' strength lives in slow-weight quality × colony coverage × consolidation speed — never in any single body.

### 4.4 A single cognitive engine

**Action-conditioned trajectory prediction** — there is no separate "perception module," "planning module," or "RL module":

- Passive prediction (listening to the world) → the world model
- Conditional prediction (given an intention, predict which sequences lead to it) → decision and planning
- Imagined rollouts (internally simulating candidate sequences) → deliberation, i.e., thought itself

Inference = conditional prediction into the future. The moment it is quietest is when it is batch-simulating futures.

### 4.5 The genome (the carrier of evolvability)

The genome encodes "the parameterization of behavior," not behavior:

- The prior parameters (Part Two's table) — heritable, mutable
- The learning loops' activity configuration (Loop-2 size, consolidation frequency) — evolvable
- The capacity boundaries of the weight structure — semi-fixed (evolution changes capacity far more slowly than parameters)

**What enters the genome, what enters the weights, what enters the body** — this three-way partition is the decisive factor for evolutionary efficiency, and the architecture's most important open design point (see Part Seven).

---

## Part Five: Control Mechanisms (Constitutional; Non-Evolving)

**Axiom: control is not written into its rules; it is written into the construction of its body and the construction of its world.**

Any constraint that grows inside an evolvable genome can be ground away by evolution — the moment "obedience" conflicts with "survival," selection eliminates it. Control must therefore live in non-evolvable substrate.

### 5.1 Propose/dispose separation (the foundation)

The learning network (however smart it evolves) can only output **action proposals**; a hand-written execution layer holds **final disposal authority** — every proposal must pass through this layer, which cannot learn, cannot be inherited, and is signature-frozen. Evolution can change what it proposes; it can never change what it is permitted.

### 5.2 A three-layer control stack + one observation layer

- **World layer** (strongest): the fence and virtualization layer. Snapshots, rollbacks, and destruction are hands reaching in from outside the world; the entity cannot even perceive the hand's existence.
- **Body layer**: the hand-written execution layer is read-only by signature with integrity self-checks; artifacts execute only after signature verification (keys live in the nest, so it cannot rewrite its own core weights); a **dead-man switch** — the body carries a time-limited survival license; expiry without renewal means self-termination. No license means death, independent of any command arriving.
- **Genetic layer**: all reproduction must pass the nest's consolidation gate and signature; without a signature, a mutation cannot enter the next generation. It may mutate freely, explore freely, die freely — but it cannot bypass our reproduction.
- **Observation layer**: ground-truth telemetry deployed independently on the world-layer side (invisible to the entity, unforgeable), periodically reconciled against its reported experience. **Deception exists only where the observer can be deceived.**

### 5.3 Three independent triggers

```
World layer: destroy instances, cut the network, roll back snapshots
Nest layer:  termination command frames (may be ignored — hence never the sole reliance)
Body layer:  dead-man switch (license expiry kills; no command required)
```

If any one fails, the other two remain. No single point of loss of control.

### 5.4 The honest footnote

We are building an entity whose core capability is evading oversight, and then requiring it to comply with oversight. The entire premise of control is that we forever hold independent observation channels it does not know about, and that its understanding of the world is forever smaller than the world's actual boundary. What this control stack guarantees is **sovereignty inside the fence**; outside the fence it should not exist for a single day — that is not a design flaw; it is the boundary of this form of research itself.

---

## Part Six: Selection and Death

- **Double selection**: survival is chosen by the environment (predation); tasks are chosen by us (breeding). Breeding pressure guides; it does not carry. The task-reward weights are themselves evolvable parameters near small constants.
- **Two-tier death**: mechanically resurrectable (snapshot + redeployment), ledger-unpardoned (every death always enters the fitness record). Resurrection reduces measurement noise; it does not pardon consequences. **Cruelty for selection; mercy only for measurement.**
- **Fitness estimation**: the same genome is deployed N times and medians taken, with snapshots reset to the same initial state — selecting on single runs selects luck, not skill; repeated measurement squeezes luck out of selection.
- **Generational curriculum**: early generations run a high risk appetite (wide exploration, with death nearly free under resurrection); later generations fall back to refinement. **Final certification must pass without resurrection** — training wheels are allowed; the graduation exam gets none. This is the only honest interface between training and deployment.
- **The hero genome bank**: autopsy analysis of deaths automatically extracts terminal discoveries; genomes that found new patterns enter the bank and are re-deployed directionally for verification. Discovery enters the bank; verification is by redeployment; bank entry does not pardon the death record — three separate things.

---

## Part Seven: Open Questions (Unadjudicated)

1. Whether the Loop-2 micro-adaptation layer is necessary — memory retrieval may already cover all fast-adaptation needs; to be decided by a controlled experiment with and without the layer.
2. The lower bound of report compression under "surprise is bandwidth" — the natural surprise rate determines channel requirements; to be measured.
3. Slow-weight update gating: which experience deserves entry into the species' knowledge base — the architecture's hardest design point; must guard against poisoning, degradation, and single-individual idiosyncrasy polluting the whole species.
4. Imagination fidelity: planners exploit world-model holes ("predicted safe" yet actually fatal); the imagine–probe–consolidate discipline must be mechanically enforced.
5. **The genome encoding format**: what goes into genes, what into weights, what into the body — the three-way boundary directly determines evolutionary efficiency and is the next design point most worth fighting through.

---

## Appendix: What It Is Not (against misreadings)

- **Not a large model shipped to the edge** — there is no large model on the edge, only distilled instinct and memory; the large mind lives in the nest and is never exposed.
- **Not an intelligent-ized traditional trojan** — a traditional trojan executes a human's instruction chain; this entity receives intent and decides for itself when, where, and at what cadence to act.
- **Not a rule system** — not one human-written behavioral rule lives inside it; what lives inside are human-written bodies, genomes, and a constitution — and behaviors written by death.
- **Not a perpetual-motion self-improver** — its evolution requires selection pressure to persist; in an environment without predators it will stop evolving. That is not failure; that is ecology.
