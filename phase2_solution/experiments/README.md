# Experiments

Standalone experiments that test ideas beyond the main pipeline. These were originally in a top-level experiments/ folder but moved here since they're part of the Phase 2 solution work.

**read_write_restoration.py** - tests whether safety degradation comes primarily from damaged read matrices or damaged write matrices. Loads the NF4 model, then swaps in FP32 weights for each group independently and measures ASR:

1. FP32 baseline
2. NF4 baseline (no repair)
3. NF4 + write matrices restored (o_proj, down_proj)
4. NF4 + read matrices restored (q/k/v/gate_proj/up_proj)
5. NF4 + all matrices restored (upper bound)

If restoring read matrices recovers more safety than restoring write matrices, that confirms the Read-Side Collapse hypothesis: quantization primarily damages the detectors, not the alarms.

Note: this is a coarser experiment than the V-score repair. It replaces entire matrices rather than targeted rows. It is meant to establish which side of the circuit is more affected, not to be an efficient repair strategy.

Run:
```bash
python phase2_solution/experiments/read_write_restoration.py --model qwen1.5b --n_eval 50
```
