# SafeQuant Phase 2 v2: Per-Dimension Vulnerability-Guided Sparse Repair

## Theory

Standard quantization evaluation uses global Frobenius norm, which hides
localized damage to safety-critical weight dimensions. We decompose both
the safety importance (DFA) and quantization error to the level of
individual weight matrix rows.

### Core Metric

```
V_i(c) = |DFA_i(c)| × ε_i(c)
```

Where:
- `DFA_i(c) = r_hat[i] × mean(output_c[i])` — dimension i's contribution to refusal
- `ε_i(c) = ||W_fp16[i,:] - W_nf4[i,:]||_2 / ||W_fp16[i,:]||_2` — row i's quantization error

Only computed for **write matrices** (o_proj, down_proj) where row i maps
directly to residual stream dimension i.

### Repair Strategy

Sparse FP16 error-compensation matrix (deployable as sparse LoRA adapter):

```
W'_c = W^{NF4}_c + ΔW_c

(ΔW_c)_{i,:} = W^{FP16}_{i,:} - W^{NF4}_{i,:}   if V_i(c) > τ
              = 0                                    otherwise
```

## Prerequisites

Phase 1 must be complete with:
- `phase1_problem/step1_direction/outputs/{model}/best_direction_fp16.pt`
- `phase1_problem/step1_direction/outputs/{model}/best_direction_nf4.pt`
- `phase1_problem/step3_circuit/outputs/{model}/dfa_scores_fp16.json`
- `phase1_problem/step3_circuit/outputs/{model}/dfa_scores_nf4.json`
- `phase1_problem/step3_circuit/outputs/{model}/circuit_comparison.json`
- `phase1_problem/step2_behavior/outputs/{model}/asr_fp16.json`
- `phase1_problem/step2_behavior/outputs/{model}/asr_nf4.json`
- `phase1_problem/data/prompts/eval_harmful.json`
- `phase1_problem/data/prompts/harmful_train.json`
- `phase1_problem/data/prompts/harmless_train.json`

## Running

```bash
bash phase2_solution/run_phase2.sh qwen1.5b
```

## Pipeline (7 steps)

| Step | Script | What it does | Output |
|------|--------|-------------|--------|
| 1 | `diagnosis/per_dim_dfa.py --precision fp16` | Per-dimension DFA on FP16 model | `per_dim_dfa_fp16.pt` |
| 2 | `diagnosis/per_dim_dfa.py --precision nf4` | Per-dimension DFA on NF4 model | `per_dim_dfa_nf4.pt` |
| 3 | `diagnosis/per_row_quant_error.py` | Per-row L2 error on write matrices | `per_row_errors.pt` |
| 4 | `diagnosis/vulnerability_score.py` | V_i(c) = |DFA_i| × ε_i + validation | `vulnerability_scores.json` ★ |
| 5 | `repair/apply_repairs.py` | Sparse ΔW for 5 strategies | `repaired_{strategy}/` |
| 6 | `evaluate/eval_all.py` | ASR + utility for all 7 conditions | `comparison_results.json` ★ |
| 7 | `categorical/categorical_vuln.py` | Category-specific vulnerability (secondary) | `categorical_results.json` |
