# Scripts (optional cluster + utilities)

**Primary entry point:** [`run_pipeline.py`](../run_pipeline.py) at the repo root.

- `python run_pipeline.py --results --md` - rebuild tables from committed `outputs/` (no GPU)
- `python run_pipeline.py --run table1` (or `full`) - re-run experiments (GPU + checkpoints)

This folder holds result aggregators, checkpoint helpers, and optional Bridges-2 Slurm launchers.

| Resource | Override via |
|----------|----------------|
| Repo checkout | `REPO_ROOT` |
| HF cache | `HF_HOME` |
| Checkpoints | `QM_ROOT` |
| Conda env | `CONDA_ENV` |

Copy `.env.example` to `.env` in the repo root and edit paths before running or submitting jobs.

## Layout

```
scripts/
  load_env.sh             # loads .env (used by fetch + sbatch jobs)
  _env.sh                 # shared Slurm env (sourced by every job)
  fetch_checkpoints.sh    # download AWQ/AQLM into QM_ROOT
  logs/                   # Slurm stdout/stderr
  tests/                  # smoke jobs: run these before the full grid
  signal_geometry/        # tab:geometry
  behavioral_direction/   # tab:behavioral
  regime_sweep/           # tab:regime, tab:ppl, tab:dequant
  localization/           # tab:f3a_proj, tab:f3a_range
  headline/               # tab:f3a, tab:stats
  trained_repair/         # tab:trained_repair, tab:lrqat
  boundary/               # tab:controls (refit)
```

## First-time setup (login node)

```bash
cp .env.example .env   # edit HF_HOME, QM_ROOT
pip install -r requirements.txt
huggingface-cli login
bash scripts/fetch_checkpoints.sh
```

## Python CLI

Same two commands as the repo root:

```bash
python run_pipeline.py --run table1
python run_pipeline.py --run full
python run_pipeline.py --results --md
```

## Optional: Bridges-2 paths

Claim-aligned sbatch jobs mirror `run_pipeline.py`. Paths:

- Jobs `cd` into `REPO_ROOT` and run `python -m experiments.<claim>.<module>` so imports resolve without `sys.path` hacks.
- Result JSONs go under `outputs/<claim>/` via `core.manifest.save_result` (except Q-Resafe pairs/DPO/LR-QAT which may still take `--output`).
- Do **not** put large HF downloads under `$HOME`; `HF_HOME` points at Ocean.
- Model aliases / HuggingFace IDs live in `config.py` (`MODELS`, `QUANTIZED_CHECKPOINTS`), not in these sbatch files.
