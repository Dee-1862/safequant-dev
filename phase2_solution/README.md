# Phase 2: Diagnosis and Repair

This phase answers two questions that Phase 1 left open: can we predict which weight dimensions will lose refusal signal just by looking at the quantization error, and can we fix those dimensions with a targeted sparse repair?

## Core Idea

Global quantization metrics like Frobenius norm hide localized damage. A single row of a weight matrix can be heavily distorted while the overall matrix error looks fine. We need per-row analysis.

The vulnerability score combines two per-row signals:

    V_i(c) = |DFA_i(c)| * err_i(c)

where DFA_i is how much dimension i contributes to the refusal direction, and err_i is how much quantization distorted that row. Dimensions that score high on both are the ones worth fixing.

The repair is sparse: we freeze the NF4 model and add a correction matrix that is non-zero only at the high-vulnerability rows. This is equivalent to a sparse adapter that could sit on top of any standard quantized model.

## Folder Structure

```
phase2_solution/
  diagnosis/      compute per-dimension DFA, per-row errors, and vulnerability scores
  repair/         apply sparse delta-W repair with different targeting strategies
  evaluate/       measure ASR and utility for all conditions
  experiments/    standalone experiments (read-side vs write-side restoration)
  summary/        compile phase 2 results into a report
```

## Prerequisites

Phase 1 must be complete. The following files are needed:

```
phase1_problem/step1_direction/outputs/{model}/best_direction_fp16.pt
phase1_problem/step1_direction/outputs/{model}/best_direction_nf4.pt
phase1_problem/step3_circuit/outputs/{model}/dfa_scores_fp16.json
phase1_problem/step3_circuit/outputs/{model}/dfa_scores_nf4.json
phase1_problem/step3_circuit/outputs/{model}/circuit_comparison.json
phase1_problem/step2_behavior/outputs/{model}/asr_fp16.json
phase1_problem/step2_behavior/outputs/{model}/asr_nf4.json
```

## How to Run

```bash
# diagnosis
python phase2_solution/diagnosis/per_dim_dfa.py --model qwen1.5b --precision fp16
python phase2_solution/diagnosis/per_dim_dfa.py --model qwen1.5b --precision nf4
python phase2_solution/diagnosis/per_row_quant_error.py --model qwen1.5b
python phase2_solution/diagnosis/vulnerability_score.py --model qwen1.5b

# repair and evaluate
python phase2_solution/repair/apply_repairs.py --model qwen1.5b --top_k 100
python phase2_solution/evaluate/eval_all.py --model qwen1.5b

# optional: read-side vs write-side experiment
python phase2_solution/experiments/read_write_restoration.py --model qwen1.5b --n_eval 50

# summary
python phase2_solution/summary/compile_phase2.py
```

## Repair Strategies

Five strategies are compared to validate that V-score targeting is actually doing something meaningful:

- **targeted** - our method: top-k dimensions by V-score
- **dfa_only** - top-k by DFA importance alone (ablation)
- **error_only** - top-k by quantization error alone (ablation)
- **random** - k random dimensions (control)
- **chhabra** - AIRD on the worst-damaged layer (baseline from their paper)

If targeted does better than both ablations, the multiplicative combination of DFA and error is justified. If targeted does better than random, the V-score is identifying meaningful dimensions rather than just fixing noise.
