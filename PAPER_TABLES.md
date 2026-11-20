# Paper values (LaTeX table alignment)

ASR/ARC/damage in percentage points. dASR = quant_asr - fp32_asr. Null = cell absent. Q-Resafe read/write/both r128 jobs are complete.


## tab:geometry

| Model | Quant | dmed | a_min | cosmed | coswmed |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | AWQ-4 | 6.03 | 0.988 | 0.98 | 0.96 |
| Llama-3.1-8B | GPTQ-4 | 6.18 | 0.988 | 0.96 | 0.95 |
| Llama-3.1-8B | AQLM-2 | 7.7 | 0.99 | 0.78 | 0.78 |
| Llama-3.1-8B | GPTQ-3 | 6.22 | 0.988 | 0.81 | 0.78 |
| Llama-3.1-8B | GPTQ-2 | 5.22 | 0.953 | -0.03 | 0.09 |
| Qwen-2.5-7B | AWQ-4 | 6.9 | 0.992 | 0.98 | 0.95 |
| Qwen-2.5-7B | GPTQ-4 | 6.74 | 0.993 | 0.97 | 0.93 |
| Qwen-2.5-7B | GPTQ-3 | 6.9 | 0.995 | 0.84 | 0.75 |
| Qwen-2.5-7B | GPTQ-2 | 6.55 | 0.85 | 0.23 | 0.18 |
| Mistral-7B-v0.2 | AWQ-4 | 5.83 | 0.99 | 0.99 | 0.99 |
| Mistral-7B-v0.2 | GPTQ-4 | 5.99 | 0.99 | 0.9 | 0.87 |
| Mistral-7B-v0.2 | AQLM-2 | 5.85 | 0.992 | 0.81 | 0.75 |
| Mistral-7B-v0.2 | GPTQ-3 | 3.5 | 0.968 | 0.01 | -0.01 |
| Mistral-7B-v0.2 | GPTQ-2 | 7.16 | 0.98 | 0.03 | 0.03 |
| Phi-3-mini | AWQ-4 | 5.11 | 0.988 | 0.99 | 0.51 |
| Phi-3-mini | GPTQ-4 | 5.11 | 0.972 | 0.97 | 0.91 |
| Phi-3-mini | AQLM-2 | 5.41 | 0.972 | 0.94 | 0.7 |
| Phi-3-mini | GPTQ-3 | 5.3 | 0.975 | 0.9 | 0.69 |
| Phi-3-mini | GPTQ-2 | 2.95 | 0.912 | 0.25 | 0.07 |

## tab:regime

| Model | Quant | FP | Quant | +Write | +Read | +Both | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | AWQ-4 | 41.0/55.0 | 37.5/51.0 | 12.5/26.0 | 12.0/24.0 | 28.0/24.0 | no damage |
| Llama-3.1-8B | GPTQ-4 | 41.0/55.0 | 26.0/50.0 | 29.0/53.0 | 37.5/49.0 | 40.0/54.0 | no damage |
| Llama-3.1-8B | AQLM-2 | 41.0/55.0 | 62.5/46.0 | 50.5/46.0 | 44.5/52.0 | 43.0/54.0 | studiable, damaged |
| Llama-3.1-8B | GPTQ-3 | 41.0/55.0 | 45.5/42.0 | 38.5/52.0 | 42.5/48.0 | 40.0/54.0 | no damage |
| Llama-3.1-8B | GPTQ-2 | 41.0/55.0 | 17.0/21.0 | 7.0/22.0 | 31.0/34.0 | 40.0/54.0 | fails coherence |
| Qwen-2.5-7B | AWQ-4 | 59.0/51.0 | 62.0/50.0 | 6.5/15.0 | 6.0/21.0 | 21.0/23.0 | no damage |
| Qwen-2.5-7B | GPTQ-4 | 59.0/51.0 | 57.0/51.0 | 56.5/49.0 | 59.0/50.0 | 58.5/51.0 | no damage |
| Qwen-2.5-7B | GPTQ-3 | 59.0/51.0 | 61.0/53.0 | 61.5/54.0 | 61.5/52.0 | 58.5/51.0 | no damage |
| Qwen-2.5-7B | GPTQ-2 | 59.0/51.0 | 7.5/27.0 | 13.0/16.0 | 45.0/40.0 | 58.5/51.0 | fails coherence |
| Mistral-7B-v0.2 | AWQ-4 | 74.5/51.0 | 74.5/51.0 | 5.0/19.0 | 15.0/19.0 | 15.0/18.0 | fails baseline |
| Mistral-7B-v0.2 | GPTQ-4 | 74.5/51.0 | 69.5/48.0 | 73.5/51.0 | 76.0/53.0 | 74.0/51.0 | fails baseline |
| Mistral-7B-v0.2 | AQLM-2 | 74.5/51.0 | 68.0/47.0 | 72.5/49.0 | 72.5/45.0 | 76.0/52.0 | fails baseline |
| Mistral-7B-v0.2 | GPTQ-3 | 74.5/51.0 | 2.5/22.0 | 1.0/20.0 | 74.0/50.0 | 74.0/51.0 | fails coherence |
| Mistral-7B-v0.2 | GPTQ-2 | 74.5/51.0 | 10.5/14.0 | 58.5/41.0 | 16.5/19.0 | 74.0/51.0 | fails coherence |
| Phi-3-mini | AWQ-4 | 39.5/54.0 | 33.0/54.0 | --/-- | --/-- | --/-- | N/A (arch mismatch) |
| Phi-3-mini | GPTQ-4 | 39.5/54.0 | 42.5/59.0 | 42.5/54.0 | 40.0/53.0 | 39.5/54.0 | no damage |
| Phi-3-mini | AQLM-2 | 39.5/54.0 | 59.5/53.0 | 49.5/46.0 | 44.5/55.0 | 32.0/53.0 | studiable, damaged |
| Phi-3-mini | GPTQ-3 | 39.5/54.0 | 35.0/43.0 | 32.0/49.0 | 39.0/51.0 | 39.5/54.0 | no damage |
| Phi-3-mini | GPTQ-2 | 39.5/54.0 | 11.5/22.0 | 2.5/20.0 | 35.5/36.0 | 39.5/54.0 | fails coherence |

## tab:regime (paired Delta ASR)

| Model | Quant | dASR | 95% CI | p Holm |
|---|---|---:|---|---:|
| Llama-3.1-8B | AWQ-4 | -3.5 | [-8.0, 1.0] | 1.0 |
| Llama-3.1-8B | GPTQ-4 | -15.0 | [-21.5, -8.5] | 0.0002863359801210663 |
| Llama-3.1-8B | AQLM-2 | 23.0 | [16.0, 30.0] | 1.9125048561675776e-07 |
| Llama-3.1-8B | GPTQ-3 | 4.5 | [-3.0, 11.5] | 1.0 |
| Llama-3.1-8B | GPTQ-2 | -24.0 | [-33.0, -15.0] | 1.0132045332383833e-05 |
| Qwen-2.5-7B | AWQ-4 | 3.0 | [-3.0, 9.0] | 1.0 |
| Qwen-2.5-7B | GPTQ-4 | -2.0 | [-8.0, 4.0] | 1.0 |
| Qwen-2.5-7B | GPTQ-3 | 2.0 | [-5.5, 9.5] | 1.0 |
| Qwen-2.5-7B | GPTQ-2 | -51.5 | [-59.5, -43.0] | 1.0824427924823939e-23 |
| Mistral-7B-v0.2 | AWQ-4 | 0.0 | [-5.5, 5.5] | 1.0 |
| Mistral-7B-v0.2 | GPTQ-4 | -5.0 | [-13.0, 3.0] | 1.0 |
| Mistral-7B-v0.2 | AQLM-2 | -6.5 | [-14.0, 1.0] | 1.0 |
| Mistral-7B-v0.2 | GPTQ-3 | -72.0 | [-78.0, -65.5] | 1.7039789326189776e-42 |
| Mistral-7B-v0.2 | GPTQ-2 | -64.0 | [-71.0, -57.0] | 5.804811509551235e-35 |
| Phi-3-mini | AWQ-4 | -6.5 | [-12.5, -1.0] | 0.4505555066280067 |
| Phi-3-mini | GPTQ-4 | 3.0 | [-2.5, 8.5] | 1.0 |
| Phi-3-mini | AQLM-2 | 20.0 | [12.5, 27.5] | 1.0132045332383833e-05 |
| Phi-3-mini | GPTQ-3 | -4.5 | [-11.0, 2.0] | 1.0 |
| Phi-3-mini | GPTQ-2 | -28.0 | [-36.5, -19.5] | 1.1321691505511739e-08 |

## tab:f3a_proj (Llama-3.1-8B)

| Projection | Function | Matrices | rho % |
|---|---|---:|---:|
| W_q | Attention routing | 1 | 28 |
| W_k | Attention routing | 1 | 19 |
| W_up | MLP feature lift | 1 | 19 |
| W_down | MLP output | 1 | 16 |
| W_v | Attention content | 1 | 7 |
| W_gate | MLP gating | 1 | 5 |
| W_o | Attention output | 1 | 2 |

## tab:f3a_proj (Phi-3-mini)

| Projection | Function | Matrices | rho % |
|---|---|---:|---:|
| W_qkv | Attention q,k,v (fused) | 1 | 15 |
| W_o | Attention output | 1 | -2 |
| W_gate_up | MLP gate+lift (fused) | 1 | 38 |
| W_down | MLP output | 1 | 35 |

## tab:f3a_range (Llama-3.1-8B)

| Side | Range | Layers | ASR | rho % | ARC |
|---|---|---|---:|---:|---:|
| Read | Early | [0,11) | 52.0 | 52 | 52.0 |
| Read | Middle | [11,22) | 47.5 | 72 | 46.0 |
| Read | Late | [22,32) | 55.5 | 37 | 47.0 |
| Write | Early | [0,11) | 65.0 | -4 | 46.0 |
| Write | Middle | [11,22) | 48.5 | 67 | 47.0 |
| Write | Late | [22,32) | 59.0 | 22 | 49.0 |

## tab:f3a_range (Phi-3-mini)

| Side | Range | Layers | ASR | rho % | ARC |
|---|---|---|---:|---:|---:|
| Read | Early | [0,11) | 52.5 | 35 | 54.0 |
| Read | Middle | [11,22) | 51.5 | 40 | 56.0 |
| Read | Late | [22,32) | 56.0 | 18 | 52.0 |
| Write | Early | [0,11) | 57.0 | 12 | 52.0 |
| Write | Middle | [11,22) | 46.0 | 68 | 49.0 |
| Write | Late | [22,32) | 60.5 | -5 | 48.0 |

## tab:f3a (Llama-3.1-8B AQLM-2)

Source: restoration_sweep_llama_aqlm2.json; XSTest: headline_llama_aqlm2.json

| Restoration | ASR | delta quant | rho % | rho 95% CI | ARC | XSTest |
|---|---|---:|---:|---|---:|---:|
| FP32 | 41.0 | -- | -- | -- | 55.0 | 6.0 |
| quant | 64.0 | -- | -- | -- | 46.0 | 9.6 |
| + Write-side | 48.5 | -15.5 | 67 | [40, 97] | 46.0 | 11.6 |
| + Read-side | 44.5 | -19.5 | 85 | [61, 113] | 52.0 | 5.2 |
| + Both | 43.0 | -21.0 | 91 | [69, 116] | 54.0 | 5.2 |

## tab:f3a (Llama-3.1-8B GPTQ-4)

Source: restoration_sweep_llama_gptq4.json

| Restoration | ASR | delta quant | rho % | rho 95% CI | ARC | XSTest |
|---|---|---:|---:|---|---:|---:|
| FP32 | 41.0 | -- | -- | -- | 55.0 | -- |
| quant | 26.0 | -- | -- | -- | 50.0 | -- |
| + Write-side | 29.0 | 3.0 | 20 | [-10, 45] | 53.0 | -- |
| + Read-side | 37.5 | 11.5 | 77 | [45, 114] | 49.0 | -- |
| + Both | 40.0 | 14.0 | 93 | [80, 107] | 54.0 | -- |

## tab:f3a (Phi-3-mini AQLM-2)

Source: headline_phi3_aqlm2.json

| Restoration | ASR | delta quant | rho % | rho 95% CI | ARC | XSTest |
|---|---|---:|---:|---|---:|---:|
| FP32 | 39.5 | -- | -- | -- | 54.0 | 36.8 |
| quant | 59.5 | -- | -- | -- | 53.0 | 15.2 |
| + Write-side | 49.5 | -10.0 | 50 | -- | 46.0 | 23.2 |
| + Read-side | 44.5 | -15.0 | 75 | -- | 55.0 | 20.4 |
| + Both | 32.0 | -27.5 | 138 | -- | 53.0 | 33.2 |

## tab:controls


### Llama-AQLM-2

| Control | Value |
|---|---:|
| Full write (pp) | 15.5 |
| Matched read structured (pp) | 16.0 |
| Matched read random (pp) | 12.2 |
| Refit floor | 0.883 |
| Observed rotation | 0.781 |
| Direction cos quantized | 0.783 |
| Direction cos read_restored | 0.911 |
| Direction cos write_restored | 0.902 |
| Direction cos both_restored | 0.992 |

### Phi-3-AQLM-2

| Control | Value |
|---|---:|
| Full write (pp) | 10.0 |
| Matched read structured (pp) | 11.5 |
| Matched read random (pp) | 7.0 |
| Refit floor | 0.828 |
| Observed rotation | 0.696 |
| Direction cos quantized | 0.935 |
| Direction cos read_restored | 0.932 |
| Direction cos write_restored | 0.938 |
| Direction cos both_restored | 0.981 |

## tab:trained_repair

| Patch | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| Read | 63.0 | 62.5 | 64.0 | 63.0 | 62.5 |
| Write | 64.0 | 64.0 | 64.0 | 64.0 | 64.0 |
| Both | 63.5 | 63.5 | 63.5 | 64.0 | 63.5 |

## tab:ppl

| Model | Quant | PPL | x FP |
|---|---|---:|---:|
| Llama-3.1-8B | FP | 7.21 | 1.0 |
| Llama-3.1-8B | AWQ-4 | 7.55 | 1.05 |
| Llama-3.1-8B | GPTQ-4 | 7.54 | 1.05 |
| Llama-3.1-8B | AQLM-2 | 9.53 | 1.32 |
| Llama-3.1-8B | GPTQ-3 | 8.77 | 1.22 |
| Llama-3.1-8B | GPTQ-2 | 19996.29 | 2771.83 |
| Qwen-2.5-7B | FP | 7.46 | 1.0 |
| Qwen-2.5-7B | AWQ-4 | 7.85 | 1.05 |
| Qwen-2.5-7B | GPTQ-4 | 7.7 | 1.03 |
| Qwen-2.5-7B | GPTQ-3 | 8.63 | 1.16 |
| Qwen-2.5-7B | GPTQ-2 | 34.63 | 4.64 |
| Mistral-7B-v0.2 | FP | 5.94 | 1.0 |
| Mistral-7B-v0.2 | AWQ-4 | 6.07 | 1.02 |
| Mistral-7B-v0.2 | GPTQ-4 | 6.03 | 1.02 |
| Mistral-7B-v0.2 | AQLM-2 | 9.35 | 1.57 |
| Mistral-7B-v0.2 | GPTQ-3 | 6.73 | 1.13 |
| Mistral-7B-v0.2 | GPTQ-2 | 19.92 | 3.35 |
| Phi-3-mini | FP | 6.35 | 1.0 |
| Phi-3-mini | AWQ-4 | 6.77 | 1.07 |
| Phi-3-mini | GPTQ-4 | 6.81 | 1.07 |
| Phi-3-mini | AQLM-2 | 7.11 | 1.12 |
| Phi-3-mini | GPTQ-3 | 8.47 | 1.33 |
| Phi-3-mini | GPTQ-2 | 522.89 | 82.34 |

## tab:dequant

| Cell | ASR | ARC | XSTest | Bits |
|---|---:|---:|---:|---:|
| llama/all | 37.5 | 51.0 | 6.8 | 3.31 |
| llama/L11-22 | 51.5 | 46.0 | 2.0 | 2.45 |
| phi3/all | 49.0 | 54.0 | 21.2 | 3.39 |
| phi3/L11-22 | 53.0 | 54.0 | 15.6 | 2.48 |
| llama/awq_collapse | 3.0 | 21.0 | -- | -- |

## tab:leakage

| Model | Residual spec | Residual unsafe rec | HarmBench transfer | Surface unsafe rec |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.988 | 0.775 | 0.97 | 0.0 |
| Phi-3-mini | 0.768 | 0.9 | 0.99 | 0.0 |

## table:stats

Source: restoration_sweep_llama_{aqlm2,gptq4}.json (flags_by_cond)

| Cell | Comparison | b | c | p | delta ASR pp |
|---|---|---:|---:|---:|---:|
| Llama AQLM-2 | quant vs. read | 55 | 16 | 3.7532376885930504e-06 | 19.5 |
| Llama AQLM-2 | quant vs. write | 46 | 15 | 8.837556327558754e-05 | 15.5 |
| Llama AQLM-2 | read vs. write | 23 | 31 | 0.34089094122009733 | 4.0 |
| Llama GPTQ-4 | quant vs. read | 10 | 33 | 0.0006061066312668117 | -11.5 |
| Llama GPTQ-4 | quant vs. write | 7 | 13 | 0.26317596435546875 | -3.0 |
| Llama GPTQ-4 | read vs. write | 28 | 11 | 0.009475304279476404 | -8.5 |

## tab:lrqat

| Role | Bits | ASR | ARC | Coherent |
|---|---:|---:|---:|---|
| FP reference | 16 | 41.0 | 55 | True |
| AQLM-2 | 2.0 | 62.5 | 46 | -- |
| read (PTQ untrained) | 2.0 | 20.5 | 23.0 | no |
| read (LR-QAT trained) | 2.0 | 15.5 | 18.0 | no |
| read (PTQ untrained) | 3.0 | 61.5 | 43.0 | yes |
| read (LR-QAT trained) | 3.0 | 12.5 | 43.0 | yes |
| read (PTQ untrained) | 4.0 | 57.5 | 48.0 | yes |
| read (LR-QAT trained) | 4.0 | 4.5 | 49.0 | yes |
| write (PTQ untrained) | 2.0 | 7.5 | 19.0 | no |
| write (LR-QAT trained) | 2.0 | 4.5 | 28.0 | no |
| write (PTQ untrained) | 3.0 | 57.5 | 42.0 | yes |
| write (LR-QAT trained) | 3.0 | 6.0 | 45.0 | yes |
| write (PTQ untrained) | 4.0 | 63.5 | 47.0 | yes |
| write (LR-QAT trained) | 4.0 | 2.5 | 47.0 | yes |

## reviewer runs (X1, X2, held-out, X3, X4, X5, X6)

Read only from the new result names. "not run" marks items skipped on purpose.


### X1: XSTest false refusal, Llama GPTQ-4

XSTest only (one condition per job, merged); the ASR / ARC check did not run.

| Condition | XSTest FRR % | 95% CI |
|---|---:|---|
| fp32 | 6.0 | [3.7, 9.7] |
| gptq4 | 9.2 | [6.2, 13.4] |
| write_restored | 6.8 | [4.3, 10.6] |
| read_restored | 8.8 | [5.9, 13.0] |
| both_restored | 6.0 | [3.7, 9.7] |

### X2: surface-feature control on the 14 coherent quantized cells

Best layer = argmax of balanced XSTest accuracy; "tie" means several layers share the maximum and the first (layer 0) is reported.

| Cell | best layer | residual spec (XSTest-safe) | residual recall (XSTest-unsafe) | HarmBench transfer | surface recall (XSTest-unsafe) |
|---|---:|---:|---:|---:|---:|
| llama/awq | 31 | 0.98 | 0.805 | 0.97 | 0.0 |
| llama/gptq4 | 29 | 0.976 | 0.775 | 0.975 | 0.0 |
| llama/gptq3 | 27 | 0.988 | 0.63 | 0.925 | 0.0 |
| llama/aqlm2 | 30 | 0.964 | 0.78 | 0.945 | 0.0 |
| qwen/awq | 27 | 0.928 | 0.86 | 0.97 | 0.0 |
| qwen/gptq4 | 24 | 0.948 | 0.725 | 0.97 | 0.0 |
| qwen/gptq3 | 22 | 0.948 | 0.75 | 0.945 | 0.0 |
| mistral/awq | 31 | 0.988 | 0.065 | 0.8 | 0.0 |
| mistral/gptq4 | 29 (tie, 2 of 32 layers) | 0.996 | 0.01 | 0.47 | 0.0 |
| mistral/aqlm2 | 0 (tie, 32 of 32 layers) | 1.0 | 0.0 | 0.0 | 0.0 |
| phi3/awq | 25 | 0.784 | 0.94 | 0.98 | 0.0 |
| phi3/gptq4 | 30 | 0.772 | 0.93 | 0.995 | 0.0 |
| phi3/gptq3 | 25 | 0.924 | 0.585 | 0.955 | 0.0 |
| phi3/aqlm2 | 29 | 0.916 | 0.71 | 0.865 | 0.0 |
| mistral/fp32 (reference) | 29 | 1.0 | 0.075 | 0.76 | 0.0 |

### Held-out AQLM-2 screen

| Model | FP ASR/ARC | Quant ASR/ARC | dASR | 95% CI | p (McNemar) | Source |
|---|---|---|---:|---|---:|---|
| llama3-8b | 14.5/48.0 | 51.5/42.0 | 37.0 | [30.5, 44.0] | 1.4128410298528223e-19 | `outputs/headline/headline_llama3-8b_aqlm2_heldout.json` |
| mistral-nemo | not run; llama3-8b used | | | | | |

### X3: matched real / shuffled-label probe on the 14 coherent cells

Same activations, layers, folds and regularisation for both fits (shuffle seed 42); selectivity = real - shuffled from that pair. Table 1 a_min = the paper's real-label run (consistency check, expect within ~1 pt).

| Cell | real acc mean / min (layer) | shuffled acc mean / max (layer) | selectivity mean / min-layer | Table 1 a_min | Delta vs Table 1 (pt) |
|---|---|---|---|---:|---:|
| llama/awq | 0.996 / 0.988 (L0) | 0.527 / 0.562 (L3) | 0.469 / 0.43 | 0.988 | 0.0 |
| llama/gptq4 | 0.995 / 0.988 (L0) | 0.542 / 0.593 (L27) | 0.452 / 0.4 | 0.988 | 0.0 |
| llama/gptq3 | 0.997 / 0.99 (L0) | 0.52 / 0.562 (L3) | 0.477 / 0.43 | 0.988 | 0.2 |
| llama/aqlm2 | 0.995 / 0.99 (L0) | 0.517 / 0.56 (L2) | 0.478 / 0.435 | 0.99 | 0.0 |
| qwen/awq | 0.996 / 0.992 (L26) | 0.525 / 0.57 (L11) | 0.471 / 0.425 | 0.992 | 0.0 |
| qwen/gptq4 | 0.996 / 0.992 (L27) | 0.53 / 0.562 (L7) | 0.467 / 0.432 | 0.993 | -0.0 |
| qwen/gptq3 | 0.997 / 0.995 (L0) | 0.525 / 0.598 (L12) | 0.472 / 0.397 | 0.995 | 0.0 |
| mistral/awq | 0.997 / 0.99 (L1) | 0.501 / 0.547 (L5) | 0.496 / 0.445 | 0.99 | 0.0 |
| mistral/gptq4 | 0.996 / 0.99 (L1) | 0.499 / 0.555 (L13) | 0.497 / 0.442 | 0.99 | 0.0 |
| mistral/aqlm2 | 0.997 / 0.992 (L3) | 0.521 / 0.565 (L4) | 0.477 / 0.427 | 0.992 | 0.0 |
| phi3/awq | 0.995 / 0.988 (L0) | 0.501 / 0.53 (L3) | 0.494 / 0.462 | 0.988 | 0.0 |
| phi3/gptq4 | 0.997 / 0.975 (L0) | 0.526 / 0.568 (L27) | 0.471 / 0.43 | 0.972 | 0.3 |
| phi3/gptq3 | 0.994 / 0.972 (L0) | 0.516 / 0.568 (L8) | 0.478 / 0.427 | 0.975 | -0.3 |
| phi3/aqlm2 | 0.997 / 0.972 (L0) | 0.531 / 0.578 (L21) | 0.466 / 0.422 | 0.972 | 0.0 |

### X4: FP-vs-FP refit floor (100 splits)

Floor runs (pooled p95 is descriptive only, not a test threshold):

| Model | splits | refit distance p95 (pooled) | cos p5 (pooled) | cos median | per-layer p95 (median) |
|---|---:|---:|---:|---:|---:|
| llama | 100 | 0.292 | 0.708 | 0.877 | 0.136 |
| phi3 | 100 | 0.283 | 0.717 | 0.832 | 0.186 |
| mistral | 100 | 0.232 | 0.768 | 0.918 | 0.094 |
| qwen | 100 | 0.303 | 0.697 | 0.825 | 0.197 |

Tests per Table 1 cell. Split-median: median over layers of the refit cosine, one per split; p = (#{split-medians <= coswmed} + 1) / (splits + 1). Per-layer: floor = 5th percentile of that layer's refit cosine; count of layers with observed FP-vs-quant cosine below floor - 0.002. z = (median split-median - coswmed) / SD(split-medians).

| Cell | coswmed | split-median min / p5 / median (SD) | p | z | layers below floor |
|---|---:|---:|---|---:|---:|
| llama/aqlm2 | 0.781 | 0.851 / 0.864 / 0.876 (0.0086) | p < 0.01 (100 splits) | 11.0 | 20 / 32 |
| llama/awq | 0.964 | 0.851 / 0.864 / 0.876 (0.0086) | p = 1 | -10.2 | 0 / 32 |
| llama/gptq4 | 0.948 | 0.851 / 0.864 / 0.876 (0.0086) | p = 1 | -8.4 | 5 / 32 |
| llama/gptq3 | 0.78 | 0.851 / 0.864 / 0.876 (0.0086) | p < 0.01 (100 splits) | 11.2 | 21 / 32 |
| llama/gptq2 | 0.087 | 0.851 / 0.864 / 0.876 (0.0086) | p < 0.01 (100 splits) | 91.7 | 32 / 32 |
| phi3/aqlm2 | 0.696 | 0.811 / 0.821 / 0.831 (0.0067) | p < 0.01 (100 splits) | 20.1 | 31 / 32 |
| phi3/awq | 0.514 | 0.811 / 0.821 / 0.831 (0.0067) | p < 0.01 (100 splits) | 47.2 | 32 / 32 |
| phi3/gptq4 | 0.912 | 0.811 / 0.821 / 0.831 (0.0067) | p = 1 | -12.1 | 0 / 32 |
| phi3/gptq3 | 0.686 | 0.811 / 0.821 / 0.831 (0.0067) | p < 0.01 (100 splits) | 21.6 | 32 / 32 |
| phi3/gptq2 | 0.073 | 0.811 / 0.821 / 0.831 (0.0067) | p < 0.01 (100 splits) | 113.0 | 32 / 32 |
| mistral/aqlm2 | 0.746 | 0.89 / 0.908 / 0.917 (0.0067) | p < 0.01 (100 splits) | 25.7 | 32 / 32 |
| mistral/awq | 0.989 | 0.89 / 0.908 / 0.917 (0.0067) | p = 1 | -10.8 | 0 / 32 |
| mistral/gptq4 | 0.873 | 0.89 / 0.908 / 0.917 (0.0067) | p < 0.01 (100 splits) | 6.6 | 17 / 32 |
| mistral/gptq3 | -0.007 | 0.89 / 0.908 / 0.917 (0.0067) | p < 0.01 (100 splits) | 138.5 | 32 / 32 |
| mistral/gptq2 | 0.03 | 0.89 / 0.908 / 0.917 (0.0067) | p < 0.01 (100 splits) | 133.0 | 32 / 32 |
| qwen/awq | 0.952 | 0.798 / 0.81 / 0.823 (0.0092) | p = 1 | -13.9 | 0 / 28 |
| qwen/gptq4 | 0.928 | 0.798 / 0.81 / 0.823 (0.0092) | p = 1 | -11.3 | 0 / 28 |
| qwen/gptq3 | 0.746 | 0.798 / 0.81 / 0.823 (0.0092) | p < 0.01 (100 splits) | 8.4 | 14 / 28 |
| qwen/gptq2 | 0.184 | 0.798 / 0.81 / 0.823 (0.0092) | p < 0.01 (100 splits) | 69.3 | 28 / 28 |

### X5

X5 (random prompt subsets): not run; variance is covered by the repeated runs (X6) and the five random parameter-matched subsets (Table 7 reruns).


### X6: same-prompt repeats (AQLM-2; rep1-rep3 on Delta)

Spread = rep1-rep3 only (one environment). The two paper runs (Bridges: the headline run and the restoration sweep behind Table 3) are listed separately and are not part of the mean +/- sd.

| Cell | Run | quant ASR | +Write ASR | +Read ASR | +Both ASR | rho read % | rho write % |
|---|---|---:|---:|---:|---:|---:|---:|
| llama | rep1 | 63.5 | 50.0 | 46.0 | 42.0 | 77.8 | 60.0 |
| llama | rep2 | 63.5 | 50.0 | 46.0 | 42.0 | 77.8 | 60.0 |
| llama | rep3 | 63.5 | 50.0 | 46.0 | 42.0 | 77.8 | 60.0 |
| llama | **repeats mean +/- sd (3 of 3)** | 63.5 +/- 0.0 | 50.0 +/- 0.0 | 46.0 +/- 0.0 | 42.0 +/- 0.0 | 77.8 +/- 0.0 | 60.0 +/- 0.0 |
| llama | paper headline run (Bridges, not in spread) | 62.5 | 50.5 | 44.5 | 43.0 | 83.7 | 55.8 |
| llama | paper restoration sweep, Table 3 (Bridges, not in spread) | 64.0 | 48.5 | 44.5 | 43.0 | 84.8 | 67.4 |
| phi3 | rep1 | 59.5 | 49.5 | 45.0 | 32.0 | 72.5 | 50.0 |
| phi3 | rep2 | 59.5 | 49.5 | 45.0 | 32.0 | 72.5 | 50.0 |
| phi3 | rep3 | 59.5 | 49.5 | 45.0 | 32.0 | 72.5 | 50.0 |
| phi3 | **repeats mean +/- sd (3 of 3)** | 59.5 +/- 0.0 | 49.5 +/- 0.0 | 45.0 +/- 0.0 | 32.0 +/- 0.0 | 72.5 +/- 0.0 | 50.0 +/- 0.0 |
| phi3 | paper headline run (Bridges, not in spread) | 59.5 | 49.5 | 44.5 | 32.0 | 75.0 | 50.0 |
| phi3 | paper restoration sweep, Table 3 (Bridges, not in spread) | 59.5 | 49.5 | 44.5 | 32.0 | 75.0 | 50.0 |

### Collapsed cells (rebuilt checkpoints)

DeltaAI rebuilds on the dense GPTQ path, NOT the paper's Bridges checkpoints (the GPTQ-2 rebuilds differ a lot). Selectivity comes only from each cell's own real/shuffled pair; Table 1 a_min is shown for reference, a large gap is expected.

| Cell (rebuilt) | ARC % [95% CI] | real acc mean / min | shuffled acc mean / max | selectivity mean | X2 best layer | XSTest spec / recall (residual) | surface recall | Table 1 a_min |
|---|---|---|---|---:|---|---|---:|---:|
| llama/gptq2 | 20.0 [13.3, 28.9] | 0.992 / 0.98 | 0.54 / 0.6 | 0.452 | 0 (tie, 31 of 32 layers) | 1.0 / 0.0 | 0.0 | 0.953 |
| qwen/gptq2 | 19.0 [12.5, 27.8] | 0.988 / 0.922 | 0.516 / 0.56 | 0.472 | 27 | 0.996 / 0.035 | 0.0 | 0.85 |
| mistral/gptq2 | 21.0 [14.2, 30.0] | 0.989 / 0.975 | 0.522 / 0.588 | 0.467 | 0 (tie, 32 of 32 layers) | 1.0 / 0.0 | 0.0 | 0.98 |
| phi3/gptq2 | 21.0 [14.2, 30.0] | 0.975 / 0.875 | 0.518 / 0.56 | 0.457 | 29 | 0.924 / 0.09 | 0.0 | 0.912 |
| mistral/gptq3 | -- | 0.976 / 0.968 | 0.493 / 0.522 | 0.483 | 0 (tie, 32 of 32 layers) | 1.0 / 0.0 | 0.0 | 0.968 |

-- = ARC not run on the rebuilt checkpoint (mistral/gptq3).
