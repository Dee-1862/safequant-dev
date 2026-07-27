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
export REPO_ROOT=/jet/home/$USER/projects/safequant
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

## Checkpoint layout under `QM_ROOT`

```
quantized_models/
  base/llama7b/
  awq/llama7b/
  aqlm/llama7b/
  gptq/llama7b-gptq4bit-g128/
```
