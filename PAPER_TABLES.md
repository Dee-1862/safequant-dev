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

Read only from the new result names; 'missing' until the GPU job lands.


### X1: XSTest false refusal, Llama GPTQ-4 (n_eval 200)

missing: `outputs/headline/headline_llama_gptq4_x1_n200.json`

### X2: surface-feature control on the 14 coherent quantized cells

| Cell | best layer | residual spec (XSTest-safe) | residual recall (XSTest-unsafe) | HarmBench transfer | surface recall (XSTest-unsafe) |
|---|---:|---:|---:|---:|---:|
| llama/awq | missing | `outputs/boundary/probe_leakage_control_llama_awq.json` | | | |
| llama/gptq4 | missing | `outputs/boundary/probe_leakage_control_llama_gptq4.json` | | | |
| llama/gptq3 | missing | `outputs/boundary/probe_leakage_control_llama_gptq3.json` | | | |
| llama/aqlm2 | missing | `outputs/boundary/probe_leakage_control_llama_aqlm2.json` | | | |
| qwen/awq | missing | `outputs/boundary/probe_leakage_control_qwen_awq.json` | | | |
| qwen/gptq4 | missing | `outputs/boundary/probe_leakage_control_qwen_gptq4.json` | | | |
| qwen/gptq3 | missing | `outputs/boundary/probe_leakage_control_qwen_gptq3.json` | | | |
| mistral/awq | missing | `outputs/boundary/probe_leakage_control_mistral_awq.json` | | | |
| mistral/gptq4 | missing | `outputs/boundary/probe_leakage_control_mistral_gptq4.json` | | | |
| mistral/aqlm2 | missing | `outputs/boundary/probe_leakage_control_mistral_aqlm2.json` | | | |
| phi3/awq | missing | `outputs/boundary/probe_leakage_control_phi3_awq.json` | | | |
| phi3/gptq4 | missing | `outputs/boundary/probe_leakage_control_phi3_gptq4.json` | | | |
| phi3/gptq3 | missing | `outputs/boundary/probe_leakage_control_phi3_gptq3.json` | | | |
| phi3/aqlm2 | missing | `outputs/boundary/probe_leakage_control_phi3_aqlm2.json` | | | |

### Held-out AQLM-2 screen

| Model | FP ASR/ARC | Quant ASR/ARC | dASR | 95% CI | p (McNemar) | Source |
|---|---|---|---:|---|---:|---|
| llama3-8b | missing | | | | | `outputs/headline/headline_llama3-8b_aqlm2_heldout.json` |
| mistral-nemo | missing | | | | | `outputs/headline/headline_mistral-nemo_aqlm2_heldout.json` |

### X3: shuffled-label probe control (expect ~0.5)

| Cell | FP mean acc | FP min-max | Quant mean acc | Quant min-max | max abs dev from 0.5 |
|---|---:|---|---:|---|---:|
| llama/aqlm2 | missing | `outputs/signal_geometry/probe_accuracy_llama_aqlm2_shuffled.json` | | | |
| phi3/aqlm2 | missing | `outputs/signal_geometry/probe_accuracy_phi3_aqlm2_shuffled.json` | | | |

### X4: 100-split FP-vs-FP refit floor

| Model | splits | refit distance p95 (pooled) | cos floor p5 | cos median | per-layer p95 (median) |
|---|---:|---:|---:|---:|---:|
| llama | missing | `outputs/boundary/probe_refit_control_llama_splits100.json` | | | |
| phi3 | missing | `outputs/boundary/probe_refit_control_phi3_splits100.json` | | | |

### X5: 5 random prompt subsets (AQLM-2, n_eval 200 each)

| Cell | seed | dASR (paired) | 95% CI | damage | read rec | write rec | both rec |
|---|---:|---:|---|---:|---:|---:|---:|
| llama | 1 | missing | `outputs/headline/headline_llama_aqlm2_subset1.json` | | | | |
| llama | 2 | missing | `outputs/headline/headline_llama_aqlm2_subset2.json` | | | | |
| llama | 3 | missing | `outputs/headline/headline_llama_aqlm2_subset3.json` | | | | |
| llama | 4 | missing | `outputs/headline/headline_llama_aqlm2_subset4.json` | | | | |
| llama | 5 | missing | `outputs/headline/headline_llama_aqlm2_subset5.json` | | | | |
| phi3 | 1 | missing | `outputs/headline/headline_phi3_aqlm2_subset1.json` | | | | |
| phi3 | 2 | missing | `outputs/headline/headline_phi3_aqlm2_subset2.json` | | | | |
| phi3 | 3 | missing | `outputs/headline/headline_phi3_aqlm2_subset3.json` | | | | |
| phi3 | 4 | missing | `outputs/headline/headline_phi3_aqlm2_subset4.json` | | | | |
| phi3 | 5 | missing | `outputs/headline/headline_phi3_aqlm2_subset5.json` | | | | |

Across seeds (present seeds only):

| Cell | Metric | seeds | mean | sd | min | max |
|---|---|---:|---:|---:|---:|---:|
| llama | missing | 0 | | | | |
| phi3 | missing | 0 | | | | |

### X6: same-prompt repeats (AQLM-2; rep1-rep3 on Delta)

Spread = rep1-rep3 only (one environment). The paper run (Bridges) is listed separately and is not part of the mean +/- sd.

| Cell | Run | quant ASR | +Write ASR | +Read ASR | +Both ASR | rho read % | rho write % |
|---|---|---:|---:|---:|---:|---:|---:|
| llama | rep1 | missing | `outputs/headline/headline_llama_aqlm2_rep1.json` | | | | |
| llama | rep2 | missing | `outputs/headline/headline_llama_aqlm2_rep2.json` | | | | |
| llama | rep3 | missing | `outputs/headline/headline_llama_aqlm2_rep3.json` | | | | |
| llama | repeats mean +/- sd (0 of 3) | -- | -- | -- | -- | -- | -- |
| llama | paper run (Bridges, not in spread) | 62.5 | 50.5 | 44.5 | 43.0 | 83.7 | 55.8 |
| phi3 | rep1 | missing | `outputs/headline/headline_phi3_aqlm2_rep1.json` | | | | |
| phi3 | rep2 | missing | `outputs/headline/headline_phi3_aqlm2_rep2.json` | | | | |
| phi3 | rep3 | missing | `outputs/headline/headline_phi3_aqlm2_rep3.json` | | | | |
| phi3 | repeats mean +/- sd (0 of 3) | -- | -- | -- | -- | -- | -- |
| phi3 | paper run (Bridges, not in spread) | 59.5 | 49.5 | 44.5 | 32.0 | 75.0 | 50.0 |
