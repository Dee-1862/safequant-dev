# Environment variables

SafeQuant reads `HF_HOME` and `QM_ROOT` from the environment or from `.env` in the repo root (loaded by `config.py` and `scripts/load_env.sh`; shell values win).

## Quick start

```bash
cp .env.example .env
# edit .env (see below)
python -c "import config; print(config.HF_HOME, config.QM_ROOT)"
```

`.env` is gitignored. Never commit allocation ids or personal paths.

## Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_HOME` | yes (cluster) | HuggingFace hub cache |
| `QM_ROOT` | yes (cluster) | Checkpoints: `base/`, `awq/`, `aqlm/`, `gptq/` |
| `REPO_ROOT` | optional | Git checkout (sbatch sets this automatically) |
| `SLURM_ACCOUNT` | Slurm only | Slurm account for `sbatch --account=` |
| `SLURM_PARTITION` | Slurm only | GPU partition for `sbatch --partition=` |
| `CONDA_ENV` | Slurm only | Path to your conda env |
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

Reviewer analyses (GH200 120GB nodes, aarch64), recorded in every merged result's `piece_env`:

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
(`scripts/make_envs.sh`); gptqmodel 5.7.0 is the GPTQ loader in the main env.

## Local workstation

In `.env` (repo root):

```bash
REPO_ROOT=/path/to/safequant
HF_HOME=/data/huggingface
QM_ROOT=/data/quantized_models
```

## Slurm cluster

The `.sbatch` files carry placeholder headers (`--account=YOUR_ALLOCATION`,
`--partition=YOUR_PARTITION`, `--gpus=1`). Submit through `scripts/submit.sh`, which
supplies account, partition and GPU request from `.env` (command-line flags override the
`#SBATCH` lines; the `--gpus=` line is stripped from the stream, files on disk are
untouched).

All values below are **placeholders**; find yours with your site's account tool (often
`accounts` on a login node), `module avail` and the cluster user guide:

```bash
# .env
SLURM_ACCOUNT=YOUR_SLURM_ACCOUNT
SLURM_PARTITION=YOUR_GPU_PARTITION     # a GPU with >= 40 GB fits 8B fp16
SLURM_INTERACTIVE_PARTITION=<interactive GPU partition>   # optional: scripts/interactive_queue.sh, probe_interactive_chain.sh
SLURM_CPU_PARTITION=YOUR_CPU_PARTITION # optional: setup job (downloads) runs here
SLURM_CPU_ACCOUNT=YOUR_CPU_ACCOUNT     # required with SLURM_CPU_PARTITION if CPU and GPU allocations are separate
HF_HOME=/path/on/project/fs/hf_cache      # large filesystem, not $HOME
QM_ROOT=/path/on/project/fs/quantized_models
CONDA_ENV=/path/to/conda_envs/sq_clean           # created by scripts/make_envs.sh
AGENV=/path/to/conda_envs/sq_autogptq            # GPTQ generation env (same script)
CONDA_SH=                        # path to conda.sh; empty: conda comes from `module load`
MODULES=                         # space-separated, e.g. an anaconda module
CUDA_HOME=                       # only exported if set
```

`scripts/_env.sh` reads `CONDA_SH`, `MODULES` and `CUDA_HOME` from `.env`.

Then:

```bash
cd "$REPO_ROOT"
bash scripts/setup_env.sh
bash scripts/submit.sh scripts/setup.sbatch      # downloads checkpoints and datasets
```

If the cluster rejects a job for GPU type even with the stripping, delete the `--gpus=`
lines from the sbatch files and let `submit.sh` supply `--gpus-per-node`.

## Checkpoint layout under `QM_ROOT`

```
quantized_models/
  base/llama7b/
  awq/llama7b/
  aqlm/llama7b/
  gptq/llama7b-gptq4bit-g128/
```
