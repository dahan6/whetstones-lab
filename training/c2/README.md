# training/c2/ — Report-Timing Learner (Architecture Implementation)

`timing_net.py` is the **architecture implementation** of the evasion-side
report-timing learner: a small policy network that maps connection-context
features to the next report-back interval, trained with evolution strategies
(ES) against the simulated NDR judge (`training/ndr/simulator.py`). It is the
red-side counterpart of the blue-side FFT-based timing/rhythm detection
published in the companion work — the two form the temporal-shape
attack/defense pair studied in the Whetstones TR (temporal shape, cap
cession).

**Withheld (dual-use): the trained timing model.** All trained timing-model
weights (`timing_net.pt`, `micro_timing*`) are intentionally not published.
Running the module trains a fresh, throwaway model from scratch against the
*simulated* judge — it contains no C2 server, channel, or payload capability
(the C2 server components `hive.py` / `dns_server.py` and the domain-fronting
template are withheld from this release).

Demo (pure simulation, architecture exercise only):

```bash
python3 -m training.c2.timing_net --gens 20 --pop 16 --out checkpoints/timing_net_demo.pt
```
