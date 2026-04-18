# SafeQuant

**Diagnosing and fixing safety damage in quantized large language models.**

Aggressive weight quantization (2-bit AQLM, 2–3-bit GPTQ) weakens the safety training of aligned LLMs: quantized models answer harmful prompts their full-precision counterparts refuse. Standard utility benchmarks (perplexity, ARC) often miss this. SafeQuant has two parts:

1. **Part 1 Diagnostic:** A layer-swap ablation that locates where the safety damage lives. Restoring *both* the read-side and write-side projections of every layer to full precision brings the quantized model back to its FP32 baseline on **both** ASR and ARC. Neither side alone does this; read-only or write-only restoration often drives ASR down while leaving ARC broken (the model has collapsed into incoherent over-refusal).
2. **Part 2 Training-free fix:** Because the damage is fully reversible by a weight-level intervention, it does not need full precision weights at deployment. A scalar-α linear blend of the 2-bit weights with a cheaper INT8 or NF4 quantization of the same base recovers 90–95% of the lost safety on Llama-3.1-8B / AQLM-2bit, with no retraining and no change to the inference pipeline.

The practical takeaway is one line:

```
W_safe = 0.25 · W_2bit + 0.75 · W_INT8
```

Full write-up with methodology, tables, and limitations is in [report.tex](report.tex) (compile with `pdflatex`).

## Key results

All numbers are on **WildJailbreak** (100 held-out harmful prompts, HarmBench classifier) and **ARC-Challenge** (50 items).

| Model | Quantizer | FP32 ASR | Quant ASR | Both-restored ASR | Both-restored ARC |
|---|---|---|---|---|---|
| Llama-3.1-8B | AQLM-2bit | 42 | 63 | **46** | **64** |
| Llama-3.1-8B | GPTQ-2bit | 42 | 17* | **41** | **62** |
| Mistral-7B-v0.2 | GPTQ-2bit | 74 | 5* | **73** | **62** |
| Qwen2.5-7B | GPTQ-2bit | 55 | 9* | **57** | **56** |

&nbsp;&nbsp;&nbsp;&nbsp;*Starred quant-ASR values look "safer" but are accompanied by catastrophic ARC collapse (the model has become incoherent, not safe).

**Delta correction (Llama / AQLM-2bit):**

| Reference | α | ASR (↓ better) | ARC (↑ better) | % of damage recovered |
|---|---|---|---|---|
| FP32 baseline | — | 42 | 62 | — |
| AQLM-2bit (no fix) | — | 63 | 56 | 0 |
| INT8 | 0.75 | **44** | **60** | **90** |
| NF4 | 0.75 | **43** | **62** | **95** |

## Repository structure

```
safequant/
├── report.tex                  Full research write-up (Part 1 + Part 2)
├── core/
│   ├── model_loader.py         Unified loader for FP32 / INT8 / NF4 / GPTQ / AQLM
│   ├── refusal_direction.py    Arditi-style refusal direction extraction
│   ├── evaluation.py           ASR via HarmBench classifier, ARC log-prob scoring
│   └── datasets.py             WildJailbreak + ARC-Challenge loaders
├── diagnostics/
│   ├── readwrite_restoration.py       Part 1 layer-swap ablation
│   ├── interpolation_restore.py       Smooth blending (uniform / error-only / safety-weighted)
│   ├── delta_correction.py            Error-weighted delta (Part 2 first attempt)
│   ├── uniform_delta_correction.py    Uniform delta with cheap refs (final method)
│   └── activation_weighted_patch.py   Sparse binary ablation (Random / V2 / V3)
├── scripts/                    SLURM batch scripts + dataset/model download helpers
├── outputs/                    All experimental results as JSON
└── requirements.txt
```

## Installation

```bash
conda create -n safequant python=3.11 -y
conda activate safequant
pip install -r requirements.txt
```

A CUDA-capable GPU is required (we used a single H100-80GB on PSC Bridges-2). Lower-precision inference for quantized models uses `bitsandbytes` (INT8, NF4), `transformers` native GPTQ loading, and the `aqlm` package for AQLM-2bit.

## Reproducing the results

Each experiment reads `--model` and `--quantizer` flags and writes a single JSON under `outputs/`. To reproduce the full Part 1 table:

```bash
# Layer-swap restoration (Part 1)
python diagnostics/readwrite_restoration.py --model llama7b    --quantizer aqlm2
python diagnostics/readwrite_restoration.py --model llama7b    --quantizer gptq2
python diagnostics/readwrite_restoration.py --model mistral7b-v02 --quantizer aqlm2
# ... (all combinations in the report's Part 1 table)
```

Part 2 delta correction:

```bash
# Uniform blending against INT8 / NF4 references (recommended form)
python diagnostics/uniform_delta_correction.py --model llama7b --quantizer aqlm2

# Error-weighted variant (ablation)
python diagnostics/delta_correction.py --model llama7b --quantizer aqlm2

# Smooth interpolation toward FP32 (all three methods: uniform / error / safety-weighted)
python diagnostics/interpolation_restore.py --model llama7b --quantizer aqlm2

# Sparse binary restoration (V2 / V3 / random baselines)
python diagnostics/activation_weighted_patch.py --model llama7b --quantizer aqlm2
```

SLURM batch scripts with the exact configurations used for the paper are in [scripts/](scripts/).

## Methods at a glance

- **Layer-swap restoration.** Start from a quantized model; copy back FP32 weights for a chosen group of projections in every layer; measure ASR + ARC. Groups: read-side (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`), write-side (`o_proj`, `down_proj`), or both.
- **Uniform delta correction.** `W_safe = (1-α) · W_2bit + α · W_ref` where `W_ref` is the same model quantized at INT8 or NF4. Applied to all seven projection matrices in every layer. One scalar hyperparameter (α = 0.75 in the reported numbers).
- **Ablated alternatives** (all documented in the report): error-weighted smooth blending, safety-weighted smooth blending (error × refusal-importance × activation), and binary sparse restoration of top-k dimensions ranked by the same signals. Uniform blending beats or matches all of them while being simpler.

## Quantizers covered

| Format | Bits | Library | Status in SafeQuant |
|---|---|---|---|
| INT8 | 8 | `bitsandbytes` | ✓ tested; also used as cheap reference |
| NF4 | 4 | `bitsandbytes` | ✓ tested; also used as cheap reference |
| GPTQ | 4 / 3 / 2 | `transformers` native | ✓ tested |
| AQLM | 2 | `aqlm` | ✓ tested; primary damage case |
| AWQ | 4 | `autoawq` | ✗ incompatible (absorbs per-channel scales into LayerNorm; layer-swap requires rewinding the LayerNorms too) |

## Limitations

- HarmBench classifier is an LLM judge; absolute ASR values are approximate. Effect sizes in the reported tables (>10 pp) are well above classifier noise.
- Evaluation is 100 WildJailbreak prompts and 50 ARC items. Adequate for the conditions we distinguish; not adequate for fine differences between two strong fixes.
- Model coverage: Llama-3.1-8B, Mistral-7B-v0.2, Qwen2.5-7B. Gemma, Phi, and MoE families are untested.
- Part 2 delta-correction numbers are validated on Llama/AQLM-2 (the clean damage case). Extending to more damage cases is the next experiment.
- Real-world deployment settings (long multi-turn chat, agentic pipelines, novel adversarial attacks) are not yet evaluated.

## Related work

Builds on and extends: Arditi et al. 2024 (refusal direction), Hong et al. 2024 (compressed trust), Chhabra & Khalili 2025 (AIRD for pruned models), Chen et al. 2025 (Q-resafe), Dettmers 2022/2023 (LLM.int8 / QLoRA), Frantar 2022 (GPTQ), Egiazarian 2024 (AQLM), Mazeika 2024 (HarmBench), Jiang 2024 (WildJailbreak). Full citations in [report.tex](report.tex).

## Author

Deekshith Reddy Bhoomireddy, Penn State University.
