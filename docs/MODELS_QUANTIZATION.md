# Self-generated GPTQ checkpoints — reproducible recipe

The pre-quantized GPTQ checkpoints referenced in `config.QUANTIZED_CHECKPOINTS`
come from **different third-party providers** (ModelCloud, TheBloke, Qwen) with
different, undocumented calibration sets and settings — so "GPTQ" is not a
controlled variable across models. To fix that, `scripts/quantize_gptq.py`
regenerates every GPTQ cell under **one fixed recipe** via
[GPTQModel](https://github.com/ModelCloud/GPTQModel) (the maintained successor to
the deprecated auto-gptq).

AQLM is left as the public ISTA-DASLab checkpoints (pinned by revision SHA) — it
uses expensive PV-tuning that we do not reproduce. So provenance is:
**AQLM = public pinned checkpoints; GPTQ = self-generated, fixed recipe.**

## Recipe (pin these in the paper)
These are the **community-standard GPTQ settings** — the operating point used by
distributed checkpoints (ModelCloud/TheBloke) — pinned explicitly so the recipe
is fully specified and never inherits a drifting library default.
- **Library:** `gptqmodel==<PIN THE VERSION YOU INSTALL>` (`pip show gptqmodel`).
- **bits:** 4 (also 3/2 if you need the low-bit cells) · **group_size:** 128
- **act-order:** `desc_act=True` · **sym:** `True` · **damp_percent:** `0.05`.
  `desc_act=True` (activation-order) is the dominant quality lever and the standard
  setting; it is **not** tuned toward the refusal direction (it's a generic quality
  knob). Leaving it off (`desc_act=False`) was an artifact that drifted the Llama
  refusal direction to cos **0.51** vs the act-order ModelCloud checkpoint's **0.98**.
- **Calibration:** 128 windows of **2048 tokens** each, concatenated from a fixed
  deterministic C4 slice (`--seqlen 2048 --n_calib 128`; streams the same documents
  every run — no random sampling). Full-length windows give a well-conditioned
  Hessian; the earlier ~500-token snippets under-conditioned it.
- **Seed:** 42 (`--seed`). GPTQ is deterministic given fixed calibration data;
  `--deterministic` adds bit-exact cuBLAS/torch kernels.
- **FP base:** pinned to `config.MODEL_REVISIONS` where set (e.g. `phi3` →
  `ff07dc01…`) so GPTQ weights match the same source revision as the AQLM cells.

## Generate
```bash
pip install gptqmodel          # then record the version below
# one per model, in parallel:
sbatch scripts/quantize_gptq.sbatch llama7b
sbatch scripts/quantize_gptq.sbatch phi3
sbatch scripts/quantize_gptq.sbatch mistral7b-v02
sbatch scripts/quantize_gptq.sbatch qwen7b
```
Each job prints the exact `config.QUANTIZED_CHECKPOINTS` line to paste, pointing at
`<ocean>/quantized_models/<model>-gptq4bit-g128`.

## Installed version
```
gptqmodel==7.1.0        # in the isolated sq_gptq env (generation ONLY)
generated: 2026-07-15
```

## Loading self-generated GPTQ in the eval env (sq_clean)
Checkpoints are produced under gptqmodel/transformers-5 but loaded by the pipeline
under transformers 4.57. Two one-time requirements make them load cleanly:
- **`optimum>=1.24.0`** in `sq_clean` (`pip install "optimum>=1.24.0"`, resolves to
  2.2.0) — else transformers raises "The gptqmodel version should be >= 1.4.3,
  optimum version should >= 1.24.0". Installs with NO core-stack change.
- `quantize_gptq.py` strips the transformers-5 `tokenizer_class` ("TokenizersBackend")
  from `tokenizer_config.json` so tf-4.x loads the checkpoint's own LlamaTokenizerFast.
  Checkpoints made before that fix were patched once with a small json edit.

Loading uses the Triton GPTQ kernel (CUDA/ExLlama extensions aren't built for this
torch) — slower but correct.
