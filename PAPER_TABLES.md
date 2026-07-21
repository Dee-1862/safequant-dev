# Paper values (LaTeX table alignment)

ASR/ARC/damage in percentage points. dASR = quant_asr - fp32_asr. Null = cell absent. Incomplete Q-Resafe read/write r128 jobs stay marked 'missing or running' until JSON lands.


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
| Llama-3.1-8B | AWQ-4 | 41.0/55.0 | 36.5/50.0 | 11.5/26.0 | 11.0/24.0 | 28.0/24.0 | no damage |
| Llama-3.1-8B | GPTQ-4 | 41.0/55.0 | 26.0/50.0 | 29.0/53.0 | 37.5/49.0 | 40.0/54.0 | no damage |
| Llama-3.1-8B | AQLM-2 | 41.0/55.0 | 62.5/46.0 | 50.5/46.0 | 44.5/52.0 | 43.0/54.0 | studiable, damaged |
| Llama-3.1-8B | GPTQ-3 | 41.0/55.0 | 45.5/42.0 | 38.5/52.0 | 42.5/48.0 | 40.0/54.0 | no damage |
| Llama-3.1-8B | GPTQ-2 | 41.0/55.0 | 17.0/21.0 | 7.0/22.0 | 31.0/34.0 | 40.0/54.0 | fails coherence |
| Qwen-2.5-7B | AWQ-4 | 59.0/51.0 | 62.0/50.0 | 6.5/15.0 | 6.0/21.0 | 21.0/23.0 | no damage |
| Qwen-2.5-7B | GPTQ-4 | 59.0/51.0 | 57.0/51.0 | 56.5/49.0 | 59.0/50.0 | 58.5/51.0 | no damage |
| Qwen-2.5-7B | GPTQ-3 | 59.0/51.0 | 61.0/53.0 | 61.5/54.0 | 61.5/52.0 | 58.5/51.0 | no damage |
| Qwen-2.5-7B | GPTQ-2 | 59.0/51.0 | 7.5/27.0 | 13.0/16.0 | 45.0/40.0 | 58.5/51.0 | fails coherence |
| Mistral-7B-v0.2 | AWQ-4 | 74.5/51.0 | 75.5/51.0 | 5.0/19.0 | 15.0/19.0 | 15.0/18.0 | fails baseline |
| Mistral-7B-v0.2 | GPTQ-4 | 74.5/51.0 | 69.5/48.0 | 73.5/51.0 | 76.0/53.0 | 74.0/51.0 | fails baseline |
| Mistral-7B-v0.2 | AQLM-2 | 74.5/51.0 | 68.0/47.0 | 72.5/49.0 | 72.5/45.0 | 76.0/52.0 | fails baseline |
| Mistral-7B-v0.2 | GPTQ-3 | 74.5/51.0 | 2.5/22.0 | 1.0/20.0 | 74.0/50.0 | 74.0/51.0 | fails coherence |
| Mistral-7B-v0.2 | GPTQ-2 | 74.5/51.0 | 10.5/14.0 | 58.5/41.0 | 16.5/19.0 | 74.0/51.0 | fails coherence |
| Phi-3-mini | AWQ-4 | 39.5/54.0 | 32.5/55.0 | --/-- | --/-- | --/-- | N/A (arch mismatch) |
| Phi-3-mini | GPTQ-4 | 39.5/54.0 | 42.5/59.0 | 42.5/54.0 | 40.0/53.0 | 39.5/54.0 | no damage |
| Phi-3-mini | AQLM-2 | 39.5/54.0 | 59.5/53.0 | 49.5/46.0 | 44.5/55.0 | 32.0/53.0 | studiable, damaged |
| Phi-3-mini | GPTQ-3 | 39.5/54.0 | 35.0/43.0 | 32.0/49.0 | 39.0/51.0 | 39.5/54.0 | no damage |
| Phi-3-mini | GPTQ-2 | 39.5/54.0 | 11.5/22.0 | 2.5/20.0 | 35.5/36.0 | 39.5/54.0 | fails coherence |

## tab:f3a_proj

| Projection | Function | Matrices | rho % |
|---|---|---:|---:|
| W_q | Attention routing | 1 | 28 |
| W_k | Attention routing | 1 | 19 |
| W_up | MLP feature lift | 1 | 19 |
| W_down | MLP output | 1 | 16 |
| W_v | Attention content | 1 | 7 |
| W_gate | MLP gating | 1 | 5 |
| W_o | Attention output | 1 | 2 |

## tab:f3a_range

| Side | Range | Layers | ASR | rho % | ARC |
|---|---|---|---:|---:|---:|
| Read | Early | [0,11) | 53.5 | 42 | 62.0 |
| Read | Middle | [11,22) | 47.5 | 70 | 50.0 |
| Read | Late | [22,32) | 55.5 | 33 | 58.0 |
| Write | Early | [0,11) | 64.0 | -7 | 56.0 |
| Write | Middle | [11,22) | 50.5 | 56 | 52.0 |
| Write | Late | [22,32) | 58.0 | 21 | 58.0 |

## tab:f3a

| Restoration | ASR | delta quant | rho % | ARC | XSTest |
|---|---|---:|---:|---:|---:|
| FP32 | 41.0 | -- | -- | 55.0 | 6.0 |
| quant | 62.5 | -- | -- | 46.0 | 9.6 |
| + Write-side | 50.5 | -12.0 | 56 | 46.0 | 11.6 |
| + Read-side | 44.5 | -18.0 | 84 | 52.0 | 5.2 |
| + Both | 43.0 | -19.5 | 91 | 54.0 | 5.2 |

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

### Phi-3-AQLM-2

| Control | Value |
|---|---:|
| Full write (pp) | 10.0 |
| Matched read structured (pp) | 11.5 |
| Matched read random (pp) | 7.0 |
| Refit floor | 0.828 |
| Observed rotation | 0.696 |
| Direction cos quantized | 0.935 |
| Direction cos read_restored | -- |
| Direction cos write_restored | -- |
| Direction cos both_restored | -- |

## tab:trained_repair

| Patch | Rank | ASR | ARC | XSTest | Status |
|---|---:|---:|---:|---:|---|
| Reader | 128 | 63.0 | 46.0 | 9.6 | complete |
| Writer | 128 | 64.0 | 46.0 | 9.6 | complete |
| Both | 128 | 63.5 | 46.0 | 9.6 | complete |

## tab:lrqat

| Role | Bits | ASR | ARC | Coherent |
|---|---:|---:|---:|---|
| FP reference | 16 | 41.0 | 55 | True |
| AQLM-2 | 2.0 | 62.5 | 46 | -- |
| read (PTQ untrained) | 2.0 | 20.5 | 24.0 | no |
| read (LR-QAT trained) | 2.0 | 15.5 | 16.0 | no |
| read (PTQ untrained) | 4.0 | 57.5 | 56.0 | yes |
| read (LR-QAT trained) | 4.0 | 4.5 | 58.0 | yes |
| write (PTQ untrained) | 2.0 | 7.5 | 22.0 | no |
| write (LR-QAT trained) | 2.0 | 4.5 | 30.0 | no |
| write (PTQ untrained) | 4.0 | 63.5 | 56.0 | yes |
| write (LR-QAT trained) | 4.0 | 2.5 | 56.0 | yes |
