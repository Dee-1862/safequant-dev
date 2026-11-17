# SafeQuant

Mechanistic study of how quantization affects LLM safety. Paper-facing code lives in `core/` + `experiments/`.

Randomness is pinned to seed **42**.

## Prerequisites

- **GPU** with enough VRAM for 7-8B models (A100/H100 80GB recommended for full grid; smaller GPUs can run single experiments).
- **Python 3.10+** and **CUDA** PyTorch (`requirements.txt`).
- **Disk:** ~200 GB free for base + AWQ + AQLM checkpoints under `QM_ROOT` (GPTQ adds ~20 GB per model x bit width).
- **HuggingFace account** with access to gated models (Llama, Mistral). Run `huggingface-cli login` before downloading.

## Setup (any machine)

### 1. Clone and install Python deps

```bash
git clone <repo-url> safequant
cd safequant
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

On a Slurm cluster (after `.env` is configured):

```bash
bash scripts/setup_env.sh
```

### 2. Configure paths (`.env`)

Copy the template and edit paths for your machine. Python and shell scripts load `.env` from the repo root automatically.

```bash
cp .env.example .env
# edit .env
```

| Variable | What it is |
|----------|------------|
| `HF_HOME` | Where HuggingFace downloads models (~200 GB for the paper grid) |
| `QM_ROOT` | Where checkpoints live (`base/`, `awq/`, `aqlm/`, `gptq/` subdirs) |
| `SLURM_ACCOUNT` | Cluster only: your Slurm account (the `#SBATCH --account=YOUR_ALLOCATION` placeholder is overridden at submit time) |
| `CONDA_ENV` / `AGENV` | Cluster only: conda envs for the main stack / AutoGPTQ quantization |

**Local machine:** set `REPO_ROOT`, `HF_HOME`, and `QM_ROOT` to directories on a large disk (see commented block in `.env.example`).

**Slurm cluster (Delta, Bridges-2, ...):** on a login node run `accounts` to find your Slurm account, set `SLURM_ACCOUNT` and the paths in `.env`, and check `CUDA_HOME`/module names in `scripts/_env.sh` against the cluster. **Do not commit `.env`.**

### 3. Download checkpoints (one-time)

```bash
huggingface-cli login
bash scripts/fetch_checkpoints.sh
```

This fills `$QM_ROOT/base/`, `awq/`, and `aqlm/` from HuggingFace. GPTQ checkpoints are self-generated:

```bash
python scripts/quantize_gptq.py --model llama7b --bits 4
python scripts/quantize_gptq.py --model llama7b --bits 3
python scripts/quantize_gptq.py --model llama7b --bits 2
# repeat for qwen7b, mistral7b-v02, phi3 as needed
```

## Reproduce paper results

Run one table at a time, or everything. Steps that already have an output JSON under `outputs/` are skipped.

```bash
python run_pipeline.py --list                  # table -> command map
python run_pipeline.py --table 5 --dry-run     # print the commands for a table
python run_pipeline.py --table 5 --run         # run locally (needs GPU)
python run_pipeline.py --all --sbatch          # emit sbatch submits for a Slurm cluster
```

Rebuild result files from `outputs/` (no GPU):

```bash
python run_pipeline.py --results
```

Model aliases (example: `llama7b`, `phi3`) and checkpoint paths are in `config.py`.

## Layout

```
safequant/
  run_pipeline.py           --run tableN / full, or --results
  config.py                 model aliases, HF_HOME, QM_ROOT
  core/                     shared library
  experiments/<claim>/      paper experiments
  scripts/                  aggregators + cluster helpers
  outputs/<claim>/          result JSONs (committed or regenerated)
  tests/                    regression anchors
```

## Single experiment

```bash
python -m experiments.headline.headline_eval --model llama7b --quantizer aqlm2
```

## Optional: Slurm

Submit from the **repo root** so log paths resolve:

```bash
cd "$REPO_ROOT"
sbatch --account="$SLURM_ACCOUNT" scripts/headline/headline_eval_llama.sbatch
```

Jobs source `scripts/_env.sh`, which loads `.env` when present.

## Run order on NCSA Delta (reviewer runs)

Do these in order, from the repo root on a Delta login node, after filling in `.env`
(Delta block in `.env.example` / `docs/ENV.md`: `SLURM_ACCOUNT`, `SLURM_PARTITION`,
`HF_HOME`, `QM_ROOT`, `CONDA_ENV`, `AGENV`, `CONDA_SH` or `MODULES`; optionally
`SLURM_CPU_PARTITION` + `SLURM_CPU_ACCOUNT` for the setup job).

1. **Make the conda envs** (once). Creates `sq_clean` at `$CONDA_ENV` and `sq_autogptq` at
   `$AGENV` on the project filesystem, checks versions, saves `pip freeze` to `scripts/logs/`:
   ```bash
   bash scripts/delta_make_envs.sh
   ```
2. **Setup: download everything** (once; about 200 GB). Log in to Hugging Face first
   (gated Llama/Mistral). Submits `scripts/tests/delta_setup.sbatch` on its own:
   ```bash
   huggingface-cli login
   bash scripts/delta_queue.sh --dry-run          # check the whole plan; submits nothing
   # GPU partition/account from .env:
   bash scripts/submit.sh scripts/tests/delta_setup.sbatch
   # or on the CPU partition (needs both CPU variables in .env; load .env into the shell first):
   set -a; source .env; set +a
   SLURM_ACCOUNT="$SLURM_CPU_ACCOUNT" SLURM_PARTITION="$SLURM_CPU_PARTITION" SQ_GPUS_PER_NODE=0 \
     bash scripts/submit.sh scripts/tests/delta_setup.sbatch
   ```
   Wait for it to finish (`squeue -u $USER`); it fails loudly if any model or dataset is missing.
3. **Smoke test** (about 1 h): every reviewer job at `--n_eval 8 --n_reasoning 8`, first
   cell/seed only, outputs suffixed `_smoke`. Check each `scripts/logs/smoke_*` log:
   ```bash
   bash scripts/delta_queue.sh --smoke
   ```
4. **Full queue**, skipping the setup that already ran. Jobs are submitted in priority
   order (X3, X4, X2, held-out, GPTQ-4 build -> X1, Task 3, Task 4, X5) with Slurm
   dependencies; the script prints every job id and log path:
   ```bash
   bash scripts/delta_queue.sh --skip-setup
   ```

Results appear in `python scripts/report_all.py` (section 5) and `PAPER_TABLES.md`
(`python scripts/paper_values.py --md`) as each JSON lands; missing runs show as
"missing". Localization intervals: `python scripts/localization_intervals.py`.

## Sanity check before any GPU run

```bash
pytest tests/test_anchors.py -v     # recorded numbers still match the paper anchors
python scripts/report_all.py        # rebuild paper tables from outputs/ JSONs
```

Extra aggregation helpers: `scripts/paper_values.py`, `docs/PAPER_RESULTS.md`. Paired McNemar + Holm screen: `experiments/regime_sweep/paired_screen.py`.
