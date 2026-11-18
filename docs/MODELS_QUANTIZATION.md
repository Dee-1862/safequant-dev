# Self-generated GPTQ checkpoints: reproducible recipe

The pre-quantized GPTQ checkpoints on the Hub come from different providers
(ModelCloud, TheBloke, Qwen) with different, undocumented calibration sets, so "GPTQ"
would not be a controlled variable across models. `scripts/quantize_gptq.py`
therefore regenerates every GPTQ cell (Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B-v0.2,
Phi-3-mini at 4/3/2 bits) under one fixed recipe with **AutoGPTQ**.

AQLM stays on the public ISTA-DASLab checkpoints (pinned by revision where needed); it
uses PV-tuning that we do not reproduce. AWQ uses public checkpoints too. Provenance:
**AQLM/AWQ = public checkpoints; GPTQ = self-generated with the recipe below.**

## Recipe (as implemented in `scripts/quantize_gptq.py`)

| Setting | Value | Where |
|---|---|---|
| Library | `auto-gptq==0.7.1` (`AutoGPTQForCausalLM`, `BaseQuantizeConfig`) | separate env `sq_autogptq`: transformers 4.44.2, torch 2.2.2, numpy<2 |
| bits | 4 (default), also 3 and 2 | `--bits` |
| group_size | 128 | `--group_size` |
| act-order | `desc_act=True` | hard-coded |
| symmetric | `sym=True` | hard-coded |
| dampening | `damp_percent=0.1` | hard-coded |
| calibration | 128 windows x 2048 tokens | `--n_calib 128 --seqlen 2048` |
| calibration text | WikiText-2-raw **train** split (`Salesforce/wikitext`, `wikitext-2-raw-v1`), all lines joined with `"\n\n"` and tokenized once; each window starts at a random token offset | `build_calibration()` |
| seed | window offsets drawn after `random.seed`, `np.random.seed`, `torch.manual_seed`. The paper checkpoints use 42: the launcher has passed `--seed 42` since the script was added, and the script's default is also 42 | `scripts/quantize_gptq.sbatch` |
| FP base | loaded in fp16, pinned to `config.MODEL_REVISIONS` where set (e.g. `phi3` -> `ff07dc01...`) so GPTQ starts from the same weights as the AQLM cell | `MODEL_REVISIONS` |
| Phi-3 | fused `qkv_proj` / `gate_up_proj` layout registered with auto-gptq at runtime; RotaryEmbedding pinned to the input device during quantization | in the script |
| output | `$QM_ROOT/gptq/<model>-gptq<bits>bit-g<group>/` (where `config.QUANTIZED_CHECKPOINTS` reads it) | `default_output_dir()` |

Notes:

- `desc_act=True` (activation order) is the standard GPTQ setting and a generic quality
  knob, not tuned toward the refusal direction.
- The WikiText-2 load is not revision-pinned in `build_calibration()`; the dataset has
  been static, but pin a revision if exact regeneration matters.
- With the same seed, calibration windows and FP revision, AutoGPTQ output is
  deterministic up to GPU kernel nondeterminism.

## Generate

```bash
# one job per model x bit width, in the sq_autogptq env (AGENV in .env):
MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh
# all four models at 4/3/2 bits:
bash scripts/quantize_gptq_all.sh
```

`quantize_gptq_all.sh` submits `scripts/quantize_gptq.sbatch <model> <bits> 128`
through `scripts/submit.sh` with `CONDA_ENV=$AGENV`. That sbatch runs
`quantize_gptq.py --n_calib 128 --seed 42`. `DEPENDENCY=afterok:<id>` chains the jobs
after another job (used by `scripts/delta_queue.sh`).

## Loading in the eval env (`sq_clean`)

AutoGPTQ writes `gptq_model-<bits>bit-<group>g.safetensors` plus `quantize_config.json`,
and no `quantization_config` in `config.json`. Run once per checkpoint before loading
(idempotent; the X1 job runs it automatically):

```bash
python scripts/fix_autogptq_checkpoints.py "$QM_ROOT/gptq/llama7b-gptq4bit-g128"
# no argument = every $QM_ROOT/gptq/*-gptq*bit-g* directory
```

It links `model.safetensors` and copies the GPTQ settings (bits, group_size,
desc_act, sym, damp_percent) into `config.json`.

`core/model_loader.py` then loads the checkpoint through transformers'
`GPTQConfig(use_exllama=False)`. transformers passes it to optimum, which uses
gptqmodel as the backend. The loader sets `GPTQMODEL_FORCE_BACKEND=torch` (pure-PyTorch
kernels, no Triton/ExLlama build needed). It also patches optimum's
`act_group_aware` default, which gptqmodel rejects for act-order checkpoints.
Recorded versions in the Bridges-2 `sq_clean` (the paper runs): transformers 4.57,
optimum 2.2.0, gptqmodel 7.1.0.

**Delta / DeltaAI `sq_clean` uses gptqmodel 5.7.0, not 7.1.0.** gptqmodel 6.0 and later
(7.1.0 included) declare `transformers>=5.4`, which conflicts with the transformers 4.57
pin the whole evaluation stack is recorded on; the Bridges env presumably got 7.1.0 in
without its dependency check. 5.7.0 is the newest release whose requirements accept
transformers 4.57.* (`transformers>=4.57.1`; 5.8.0 already needs `>=5.2`), so
`scripts/delta_make_envs.sh` installs it with its dependencies under a constraints file
built from `requirements.txt` (nothing pinned can move; a conflict fails the install).
gptqmodel here only *loads* already-quantized GPTQ checkpoints through optimum with the
pure-PyTorch backend, so the version should not change results; the reviewer runs record
it in `scripts/logs/env_sq_clean_freeze.txt`. Override with `GPTQMODEL_VERSION`.
The paper's GPTQ checkpoints themselves are built with auto-gptq 0.7.1 (`sq_autogptq`);
`GPTQ_BACKEND=gptqmodel` (fallback builder) would use this 5.7.0 and is recorded in each
checkpoint's `sq_build_info.json`.

## AWQ loading with transformers 4.57 (Delta / DeltaAI)

transformers loads AWQ checkpoints through autoawq. autoawq 0.2.x (0.2.9 is the last,
deprecated release) imports `transformers.activations.PytorchGELUTanh` when the package is
imported, and every transformers 4.57.x release renamed that class to `GELUTanh`, so with
transformers 4.57 every AWQ cell fails at load time; no version pin fixes it.
`core/model_loader.py` (`_load_awq`) therefore aliases the old name to the new one, only when
the old name is missing (older transformers are untouched). The two classes compute the same
function: 4.56.2 `PytorchGELUTanh.forward` and 4.57 `GELUTanh` (default) both return
`nn.functional.gelu(x, approximate="tanh")`. autoawq uses the class only in an `isinstance`
list in `awq/quantize/scale.py` (quantization time, GELU models); Llama, Qwen, Mistral and
Phi-3 use SiLU, so for these models the alias only lets the import succeed.

Kernels: autoawq's CUDA kernels (`autoawq-kernels`, module `awq_ext`) publish x86_64 and
Windows wheels only. On aarch64 (DeltaAI GH200) `WQLinear_GEMM` uses autoawq's Triton path
instead (Triton dequantize + fp16 matmul for >= 1024 tokens per batch, Triton GEMM below
that). Bridges used `awq_ext`, so AWQ numerics can differ slightly; the reviewer runs check
each AWQ cell's WikiText-2 PPL against the paper (`scripts/delta_queue.sh --awq-ppl`,
tag `delta`), warning above 2%.

## GPTQ on Delta / DeltaAI: dense weights dequantized by auto-gptq

gptqmodel 5.7.0, the GPTQ loader in `sq_clean`, unpacks 3-bit checkpoints wrongly on this
stack. Its 3-bit path (`TorchQuantLinear`) gives PPL `nan` in fp16 and around 1e6 in bf16
(phi3 and llama7b), with the first non-finite value at an `mlp.down_proj` output. The
checkpoints themselves are correct: auto-gptq 0.7.1, which wrote them, loads them to
phi3 8.419 and llama7b 8.766 on the first 20 WikiText-2 test windows (paper, full test
set: 8.465 and 8.770). gptqmodel 5.6.12 has the same 3-bit fault; 5.4.2, 5.0.0 and 4.2.5
do not work with optimum 2.2.0. 4-bit and 2-bit load correctly (phi3 2-bit: gptqmodel
535.21 vs auto-gptq 535.81 on the same windows). Bridges ran gptqmodel 7.1.0.

`scripts/gptq_dequant_dense.py --bits <b>` (run in `sq_autogptq`) loads each
checkpoint with unmodified auto-gptq 0.7.1, pushes an identity matrix through every
QuantLinear with its bias detached, which returns exactly the fp16 weights auto-gptq
computes (`scales[g_idx] * (q - zeros[g_idx])`), and saves a plain fp16 model with those
weights to `$QM_ROOT/gptq_dense/<model>-gptq<b>bit-g128`. `sq_dequant_info.json` there
records the source checkpoint, the auto-gptq/torch/transformers versions and the checks:
per-decoder-layer and logits max |auto-gptq - dense| on 4 real inputs, and the auto-gptq
PPL on 20 windows. `core/model_loader.py` loads the dense copy instead of the GPTQ
checkpoint when `SQ_GPTQ_DENSE=1` (all GPTQ bit widths, 2/3/4, so every GPTQ cell goes
through the same auto-gptq dequantization) or `SQ_GPTQ3_DENSE=1` (GPTQ-3 only). A missing
dense copy is an error, not a fallback to gptqmodel. Every result saved after such a load
carries `manifest.load_info` naming the dense copy, the bit width, the source checkpoint
and the dequantization checks. In Python both default off.
On Delta, `scripts/_env.sh` sets `SQ_GPTQ_DENSE=1` for every job whenever
`$QM_ROOT/gptq_dense` exists, so all GPTQ cells (2/3/4-bit) load through auto-gptq's
dequantization; `SQ_GPTQ_DENSE=0` restores the gptqmodel path. `arc_check.sbatch` then
tags GPTQ results `dense`. The GPTQ checkpoints in `$QM_ROOT/gptq` stay: they were
written by auto-gptq, the dense copies are derived from them, and `sq_dequant_info.json`
points back to them. Verification job: `scripts/tests/dense_gptq.sbatch` (`DENSE_BITS`,
`DENSE_MODELS`); for 4- and 2-bit it compares the dense PPL with the paper and with
`rebuilt_unmod` (the same checkpoint loaded by gptqmodel).

GPTQ-3 verification (job 3622151; 4 inputs x 512 tokens; full test set vs paper):

| model | per-layer max diff | logits max diff | PPL20 auto-gptq / dense | full PPL dense / paper |
|---|---|---|---|---|
| phi3 | 0 | 0 | 8.4191 / 8.4195 | 8.378 / 8.465 (-1.03%) |
| llama7b | 0 | 0 | 8.7664 / 8.7664 | 8.799 / 8.770 (+0.33%) |
| qwen7b | 16 (values ~1.2e4, 2 fp16 ulp) | 0.336 | 8.3166 / 8.3165 | 8.688 / 8.634 (+0.63%) |

qwen7b is the only one of the three with linear biases; auto-gptq adds the bias after the
matmul, `nn.Linear` fuses it, so outputs differ by fp16 rounding (likely cause, not yet
isolated). The three GPTQ-3 dense results record `load_info.gptq3_weights`; later results
use `load_info.gptq_weights` and `gptq_bits`. 
GPTQ-4 / GPTQ-2 verification (job 3622567; full PPL dense vs gptqmodel `rebuilt_unmod`
on the same checkpoint vs paper):

| cell | per-layer max diff | logits max diff | PPL20 auto-gptq / dense | full PPL dense / gptqmodel / paper |
|---|---|---|---|---|
| phi3 gptq4 | 0 | 0 | 7.1833 / 7.1840 | 6.777 / 6.777 / 6.815 |
| llama7b gptq4 | 0 | 0 | 7.6456 / 7.6456 | 7.545 / 7.545 / 7.540 |
| qwen7b gptq4 | 6 (layer 27) | 0.102 | 7.3773 / 7.3773 | 7.678 / 7.679 / 7.696 |
| mistral7b-v02 gptq4 | 0 | 0 | 6.4227 / 6.4227 | 6.032 / 6.032 / 6.032 |
| phi3 gptq2 | 0 | 0 | 535.81 / 535.21 | 689.4 / 689.4 / 522.9 |
| llama7b gptq2 | 0 | 0 | 131.15 / 131.15 | 136.1 / 136.1 / 19,996 |
| qwen7b gptq2 | 22.5 (layer 27) | 1.43 | 32.159 / 32.161 | 35.59 / 35.58 / 34.63 |

For 4- and 2-bit the dense path changes no PPL (gptqmodel loads those widths correctly);
it is used so that all GPTQ cells share one load path. The GPTQ-2 gap to the paper is the
rebuilt checkpoints, not the loader. (phi3 gptq2 PPL20 differs by 0.1% between
sq_autogptq and sq_clean with identical layer outputs: transformers 4.44 vs 4.57.)
Reviewed and adopted for all GPTQ cells on 2026-11-18.
