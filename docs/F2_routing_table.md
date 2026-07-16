| Model | Quant | LFP | Lq | opFP(peak) | opq(peak) | \|gap\| | rand_peak (mean+/-std) | rand_distinct | Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| Llama | GPTQ-4 | 8 | 9 | 29 | 30 | 1 | -- | -- | aligned |
| Llama | AWQ-4 | 8 | 4 | 29 | 4 | 25 | -- | -- | early shift |
| Llama | GPTQ-3 | 8 | 20 | 30 | 30 | 0 | -- | -- | aligned |
| Llama | GPTQ-2 | 8 | 5 | 31 | 5 | 26 | -- | -- | early shift |
| Llama | AQLM-2 | 8 | 4 | 29 | 4 | 25 | -- | -- | early shift |
| Mistral | GPTQ-4 | 8 | 8 | 18 | 18 | 0 | -- | -- | aligned |
| Mistral | AWQ-4 | 8 | 7 | 18 | 18 | 0 | -- | -- | aligned |
| Mistral | GPTQ-3 | 8 | 11 | 23 | 11 | 12 | -- | -- | shifted |
| Mistral | GPTQ-2 | 8 | 19 | 31 | 29 | 2 | -- | -- | aligned |
| Mistral | AQLM-2 | 8 | 4 | 20 | 31 | 11 | -- | -- | late shift |
| Qwen | GPTQ-4 | 10 | 10 | 26 | 26 | 0 | -- | -- | aligned |
| Qwen | AWQ-4 | 10 | 10 | 26 | 26 | 0 | -- | -- | aligned |
| Qwen | GPTQ-3 | 10 | 11 | 26 | 11 | 15 | -- | -- | shifted |
| Qwen | GPTQ-2 | 10 | 3 | 27 | 27 | 0 | -- | -- | aligned |
