# Environment variables

SafeQuant reads `HF_HOME` and `QM_ROOT` from the environment. There is no auto-detection: you set paths once in `scripts/env.local.sh` and source it before any run.

## Quick start

```bash
cp scripts/env.example.sh scripts/env.local.sh
# edit scripts/env.local.sh (see below)
source scripts/env.local.sh
python -c "import config; print(config.HF_HOME, config.QM_ROOT)"
```

`env.local.sh` is gitignored. Never commit allocation ids or personal paths.

## Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_HOME` | yes (cluster) | HuggingFace hub cache |
| `QM_ROOT` | yes (cluster) | Checkpoints: `base/`, `awq/`, `aqlm/`, `gptq/` |
| `REPO_ROOT` | optional | Git checkout (sbatch sets this automatically) |
| `SLURM_ACCOUNT` | Bridges only | Slurm account for `sbatch --account=` |
| `CONDA_ENV` | Bridges only | Path to your conda env |
| `AQLM_REPO` | Q-Resafe fold only | Clone of [ISTA-DASLab/AQLM](https://github.com/ISTA-DASLab/AQLM) |

Laptop defaults if unset: `~/.cache/huggingface` and `~/quantized_models`.

Reviewer-run variables (GH200 nodes; all optional):

| Variable | Purpose |
|----------|---------|
| `SQ_GPTQ_DENSE` | `1`: load every GPTQ cell from its dense dequantized copy in `$QM_ROOT/gptq_dense` (docs/MODELS_QUANTIZATION.md); `scripts/_env.sh` sets it when that folder exists |
| `SQ_GPTQ3_DENSE` | `1`: dense copy for GPTQ-3 only |
| `SQ_FIT_JOBS` | Parallel probe fits (X2 diagnostic, X4); default `OMP_NUM_THREADS` or all CPUs |
| `SQ_LOAD_LOCK` | Lock file: processes sharing it load models one at a time (`probe_pool.sbatch` sets it) |
| `SQ_GPUS_PER_NODE` | GPUs per node requested by `scripts/submit.sh` (default 1; 0 = CPU job) |
| `GPTQ_BACKEND` | `autogptq` (default) or `gptqmodel` for `scripts/quantize_gptq.py` |
| `POOL`, `MAX_PAR`, `GPU_BUDGET_GB`, `PROBE_CLAIMS` | `scripts/probe_pool.sbatch`: pieces to run, pieces at once, GPU memory budget, claim folder |
| `EVAL_BATCH_SIZE` | Generation batch size (default 16) |

## Software versions

Original analyses: H100 nodes (loader versions noted in docs/MODELS_QUANTIZATION.md,
e.g. gptqmodel 7.1.0 for GPTQ).

Reviewer analyses (GH200 120GB, aarch64), recorded in every merged result's `piece_env`:

| Package | Version |
|---|---|
| Python | 3.10.22 |
| torch | 2.9.1+cu128 (CUDA 12.8, cuDNN 9.10.2) |
| transformers | 4.57.6 |
| accelerate | 1.10.1 |
| optimum | 2.2.0 |
| aqlm | 1.1.7 |
| bitsandbytes | 0.49.2 |
| tokenizers | 0.22.2 |
| safetensors | 0.6.2 |
| numpy | 2.2.6 |
| triton | 3.5.1 |

GPTQ dense copies were written in a separate env with auto-gptq 0.7.1
(`scripts/delta_make_envs.sh`); gptqmodel 5.7.0 is the GPTQ loader in the main env.

## Local workstation

Edit `scripts/env.local.sh`:

```bash
export REPO_ROOT=/path/to/safequant
export HF_HOME=/data/huggingface
export QM_ROOT=/data/quantized_models
```

## Bridges-2: how to find your values

SSH to a **login node**, then:

```bash
accounts          # Slurm account + allocation info
echo $USER        # your username (used in Ocean paths)
ls /ocean/projects/$USER   # often shows your Ocean directory name
```

From `accounts`, note:

- **SLURM account** (first column) → `SLURM_ACCOUNT`
- **Ocean allocation** (project id in `/ocean/projects/<id>/$USER`) → replace `YOUR_OCEAN_ALLOCATION` below

Edit `scripts/env.local.sh`:

```bash
export SLURM_ACCOUNT=YOUR_SLURM_ACCOUNT
export REPO_ROOT=/path/to/projects/safequant
export HF_HOME=/ocean/projects/YOUR_OCEAN_ALLOCATION/$USER/.cache/huggingface
export QM_ROOT=/ocean/projects/YOUR_OCEAN_ALLOCATION/$USER/quantized_models
export CONDA_ENV=/ocean/projects/YOUR_OCEAN_ALLOCATION/$USER/conda_envs/sq_clean
```

Then every session:

```bash
cd "$REPO_ROOT"
source scripts/env.local.sh
bash scripts/setup_env.sh
bash scripts/fetch_checkpoints.sh
```

Submit jobs:

```bash
sbatch --account="$SLURM_ACCOUNT" scripts/headline/headline_eval_llama.sbatch
```

## NCSA Delta (Slurm)

The `.sbatch` files carry Bridges-2 headers (`--partition=GPU-shared`, `--gpus=h100-80:1`).
On Delta, submit through `scripts/submit.sh`, which supplies account, partition and GPU
request from `.env` (command-line flags override the `#SBATCH` lines; the Bridges-only
`--gpus=h100-80:1` line is stripped from the stream, files on disk are untouched).

All values below are **placeholders** - confirm on Delta with `accounts`, `module avail`
and the Delta user guide:

```bash
# .env
SLURM_ACCOUNT=<from `accounts`; GPU accounts usually end in -delta-gpu>
SLURM_PARTITION=gpuA100x4        # gpuA40x4: cheaper, fine for 8B fp16. Max time 48 h
SLURM_CPU_PARTITION=cpu          # optional: setup job (downloads) runs here
SLURM_CPU_ACCOUNT=<from `accounts`; CPU accounts usually end in -delta-cpu>  # required with SLURM_CPU_PARTITION
HF_HOME=/path/on/project/or/work/fs/hf_cache      # large filesystem, not $HOME
QM_ROOT=/path/on/project/or/work/fs/quantized_models
CONDA_ENV=/path/to/conda_envs/sq_clean           # created by scripts/delta_make_envs.sh
AGENV=/path/to/conda_envs/sq_autogptq            # GPTQ generation env (same script)
CONDA_SH=                        # empty: conda comes from `module load`
MODULES=                         # space-separated, e.g. an anaconda module
CUDA_HOME=                       # only exported if set
```

`scripts/_env.sh` reads `CONDA_SH`, `MODULES` and `CUDA_HOME` from `.env`. When they are
empty and the Bridges-2 install is present, the old Bridges defaults are used.

First jobs:

```bash
bash scripts/submit.sh scripts/tests/00_env_check.sbatch
bash scripts/submit.sh scripts/tests/01_smoke_llama_aqlm2.sbatch
```

If Delta rejects the job for GPU type even with the stripping, delete the `--gpus=` lines
from the sbatch files and let `submit.sh` supply `--gpus-per-node`.

## Checkpoint layout under `QM_ROOT`

```
quantized_models/
  base/llama7b/
  awq/llama7b/
  aqlm/llama7b/
  gptq/llama7b-gptq4bit-g128/
```
