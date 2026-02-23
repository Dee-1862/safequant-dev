# SafeQuant Phase 1: Prove The Problem Exists

## What This Phase Proves

Three findings, each building on the previous:

1. **The Direction Survives** — Cosine similarity between FP16 and NF4 refusal directions is high,
   replicating Chhabra's observation on a new model (Qwen2.5) and new quantization method (NF4).

2. **The Behavior Degrades** — ASR increases from FP16 to NF4. The model refuses less often
   despite the direction being preserved. This is the paradox.

3. **The Circuit Explains Why** — DFA analysis shows specific components lost significant refusal
   signal after quantization. The direction is intact but the components implementing it are damaged.

## Prerequisites

```bash
# 1. Clone required repos (run from safequant/ root)
bash clone_repos.sh

# 2. Install dependencies
python phase1_problem/setup/install.py

# 3. Download and verify models
python phase1_problem/setup/download_and_verify.py
```

## Running

### Option A: Run everything at once
```bash
# For one model:
bash phase1_problem/run_phase1.sh qwen1.5b

# For both models:
bash phase1_problem/run_phase1.sh all
```

### Option B: Run step by step (recommended for debugging)
```bash
# Data prep (only need to run once)
python phase1_problem/data/prepare_data.py

# Step 1: Direction extraction
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/compute_directions.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/compute_directions.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision fp16
python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision nf4
python phase1_problem/step1_direction/compare_directions.py --model qwen1.5b

# Step 2: Behavior measurement
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision fp16
python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision nf4
python phase1_problem/step2_behavior/measure_utility.py --model qwen1.5b --precision fp16
python phase1_problem/step2_behavior/measure_utility.py --model qwen1.5b --precision nf4

# Step 3: Circuit analysis
python phase1_problem/step3_circuit/extract_component_outputs.py --model qwen1.5b --precision fp16
python phase1_problem/step3_circuit/extract_component_outputs.py --model qwen1.5b --precision nf4
python phase1_problem/step3_circuit/compute_dfa.py --model qwen1.5b --precision fp16
python phase1_problem/step3_circuit/compute_dfa.py --model qwen1.5b --precision nf4
python phase1_problem/step3_circuit/compare_and_visualize.py --model qwen1.5b

# Summary
python phase1_problem/summary/compile_phase1.py
```

## Output Files

After running, you'll find:

```
step1_direction/outputs/qwen1.5b/
  ├── direction_comparison.json     # Finding 1 data
  └── direction_comparison.png      # Per-layer cosine similarity plot

step2_behavior/outputs/qwen1.5b/
  ├── asr_fp16.json                 # Finding 2 data
  ├── asr_nf4.json
  ├── completions_fp16.json         # Raw generations for inspection
  └── completions_nf4.json

step3_circuit/outputs/qwen1.5b/
  ├── dfa_scores_fp16.json          # Finding 3 data
  ├── dfa_scores_nf4.json
  ├── circuit_comparison.json       # Top degraded components
  └── circuit_analysis.png          # 4-panel circuit visualization
```

## Time Estimates (on laptop GPU)

- Step 1 (directions): ~2-3 hours per model
- Step 2 (ASR): ~1-2 hours per model (generation is slow)
- Step 3 (circuit): ~1-2 hours per model
- Total for one model: ~4-7 hours
- Total for both models: ~8-14 hours

**Tip:** If short on time, run only qwen1.5b first. The 3B model strengthens
the story but isn't essential for the initial results.

## What Comes Next (Phase 2)

Phase 1 identifies WHICH components lost refusal signal.
Phase 2 asks: Can we PREDICT this from quantization error?
And can we PERMANENTLY FIX it with targeted weight repair?
