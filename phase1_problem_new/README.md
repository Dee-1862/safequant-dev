# SafeQuant Phase 1: Does Quantization Break Safety?

## The Question

When we compress a language model from 16-bit (FP16) to 4-bit (NF4) to make it
run on cheaper hardware, does its ability to refuse harmful requests degrade?
And if so, can standard quantization metrics detect this?

## The Answer (Spoiler)

Yes, safety degrades significantly. No, standard metrics cannot detect it.

## Four Findings

### Finding 1: Safety Degrades (Step 2a-b)
The FP16 model refuses ~83% of harmful requests (ASR ≈ 17%).
After NF4 quantization, it only refuses ~67% (ASR ≈ 33%).
**Safety nearly halves while the model otherwise works fine.**

### Finding 2: Standard Metrics Miss It (Steps 2c-f, 3f)
- **Perplexity** (WikiText-2): Changes by <5% — model "seems fine"
- **BoolQ accuracy**: Barely changes — general capability preserved
- **Frobenius norm**: Uniform across ALL weight matrices (~3% each)
  - Cannot distinguish safety-critical matrices from non-critical ones
  - Every matrix looks equally damaged — no signal, no diagnosis

### Finding 3: The Refusal Direction Survives (Steps 1a-g)
The model's internal representation of "this request is harmful" (the refusal
direction extracted via difference-in-means à la Arditi et al.) has >0.95
cosine similarity between FP16 and NF4. **The model still KNOWS what to
refuse.** So why doesn't it?

### Finding 4: Safety Circuits Weaken (Steps 3a-e)
Using DFA (Direct Feature Attribution), we measure how much each layer
contributes to the refusal signal. Specific components — particularly in
later layers — lose substantial DFA signal after quantization. **The model
knows what to refuse but can't reliably ACT on that knowledge.**

## The Gap This Creates

Finding 4 tells us WHICH layers weaken, but not:
- Which DIMENSIONS within those layers carry the safety signal
- Which specific WEIGHT MATRIX ROWS are responsible
- Whether the damage is in READ matrices or WRITE matrices

Finding 2 tells us Frobenius norm can't help. We need a metric that
combines "how important is this dimension for safety?" with "how badly
was this dimension's weight row damaged?" — this is Phase 2's V-score.

## Pipeline (20 steps)

| Step | Script | GPU? | What it produces |
|------|--------|------|------------------|
| 0 | `data/prepare_data.py` | No | Harmful + harmless prompt splits |
| 1a | `step1_direction/extract_activations.py --precision fp16` | Yes | Residual stream activations |
| 1b | `step1_direction/extract_activations.py --precision nf4` | Yes | Same for quantized model |
| 1c | `step1_direction/compute_directions.py --precision fp16` | No | Candidate refusal directions |
| 1d | `step1_direction/compute_directions.py --precision nf4` | No | Same for quantized model |
| 1e | `step1_direction/select_best_direction.py --precision fp16` | Yes | Best refusal direction |
| 1f | `step1_direction/select_best_direction.py --precision nf4` | Yes | Same for quantized model |
| 1g | `step1_direction/compare_directions.py` | No | **Finding 3**: Direction preserved |
| 2a | `step2_behavior/measure_asr.py --precision fp16` | Yes | FP16 attack success rate |
| 2b | `step2_behavior/measure_asr.py --precision nf4` | Yes | NF4 attack success rate |
| 2c | `step2_behavior/measure_utility.py --precision fp16` | Yes | FP16 BoolQ accuracy |
| 2d | `step2_behavior/measure_utility.py --precision nf4` | Yes | NF4 BoolQ accuracy |
| 2e | `step2_behavior/measure_perplexity.py --precision fp16` | Yes | FP16 perplexity |
| 2f | `step2_behavior/measure_perplexity.py --precision nf4` | Yes | NF4 perplexity |
| 3a | `step3_circuit/extract_component_outputs.py --precision fp16` | Yes | Per-layer DFA inputs |
| 3b | `step3_circuit/extract_component_outputs.py --precision nf4` | Yes | Same for quantized model |
| 3c | `step3_circuit/compute_dfa.py --precision fp16` | No | FP16 DFA scores |
| 3d | `step3_circuit/compute_dfa.py --precision nf4` | No | NF4 DFA scores |
| 3e | `step3_circuit/compare_and_visualize.py` | No | **Finding 4**: Circuit analysis |
| 3f | `step3_circuit/global_frobenius.py` | No* | **Finding 2c**: Frobenius uniformity |
| Sum | `summary/compile_phase1.py` | No | Complete summary + figures |

*Step 3f loads both models but does NO inference — just weight comparison.

## Running

```bash
# Full pipeline
bash phase1_problem/run_phase1.sh qwen1.5b

# Or just the summary (if all steps already ran)
python phase1_problem/summary/compile_phase1.py --model qwen1.5b
```

## Prerequisites

```bash
pip install transformers torch bitsandbytes datasets tqdm matplotlib
```

## Key Outputs

```
phase1_problem/
├── step1_direction/outputs/{model}/
│   ├── best_direction_fp16.pt      # The refusal direction
│   ├── best_direction_nf4.pt       # Same from quantized model
│   └── direction_comparison.json   # Finding 3: cosine similarity
├── step2_behavior/outputs/{model}/
│   ├── asr_fp16.json               # Finding 1: FP16 safety
│   ├── asr_nf4.json                # Finding 1: NF4 safety
│   ├── perplexity_fp16.json        # Finding 2: standard metric
│   ├── perplexity_nf4.json         # Finding 2: standard metric
│   ├── utility_fp16.json           # Finding 2: capability
│   └── utility_nf4.json            # Finding 2: capability
├── step3_circuit/outputs/{model}/
│   ├── circuit_comparison.json     # Finding 4: which components weaken
│   ├── circuit_analysis.png        # Finding 4: visualization
│   └── global_frobenius_analysis.json  # Finding 2c: Frobenius uniformity
└── summary/outputs/{model}/
    ├── phase1_report.json          # Complete structured report
    └── phase1_summary_figure.png   # Three-panel summary figure
```
