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
Recorded versions in `sq_clean`: transformers 4.57, optimum 2.2.0, gptqmodel 7.1.0.
`scripts/delta_make_envs.sh` installs exactly these.

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
