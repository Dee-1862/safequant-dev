# Bridges-2 Slurm launchers

Claim-aligned sbatch jobs for SafeQuant. Paths match the Bridges-2 layout used previously:

| Resource | Path |
|----------|------|
| Repo checkout | `/jet/home/bhoomire/projects/safequant_v2` |
| Conda env | `/ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean` |
| HF cache (models) | `/ocean/projects/cis240116p/bhoomire/.cache/huggingface` |
| CUDA | `/opt/packages/cuda/v12.4.0` |
| Account / partition | `cis250287p` / `GPU-shared` / `h100-80:1` |

## Layout

```
scripts/
  _env.sh                 # shared Bridges-2 env (sourced by every job)
  setup_env.sh            # one-time login-node bootstrap
  logs/                   # Slurm stdout/stderr
  signal_geometry/        # Table 1
  behavioral_direction/   # Sec. 5 Arditi
  regime_sweep/           # Table 2
  localization/           # Tables 3–4
  headline/               # Table 5
  trained_repair/         # Tables 6–7
  boundary/               # Appendix controls
```

## First-time setup (login node)

```bash
bash scripts/setup_env.sh
```

## Submit examples

```bash
# From the Jet checkout
cd /jet/home/bhoomire/projects/safequant_v2

sbatch scripts/headline/headline_eval_llama.sbatch
sbatch scripts/trained_repair/qresafe_pairs.sbatch
sbatch scripts/trained_repair/qresafe_dpo.sbatch read
sbatch scripts/trained_repair/lrqat_llama.sbatch
sbatch scripts/localization/depth_bands_read.sbatch
```

Override the checkout if the repo was cloned under a different name:

```bash
REPO_ROOT=/jet/home/bhoomire/projects/safequant sbatch scripts/headline/headline_eval_llama.sbatch
```

## Notes

- Jobs `cd` into `REPO_ROOT` and run `python -m experiments.<claim>.<module>` so imports resolve without `sys.path` hacks.
- Result JSONs go under `outputs/<claim>/` via `core.manifest.save_result` (except Q-Resafe pairs/DPO/LR-QAT which may still take `--output`).
- Do **not** put large HF downloads under `$HOME` — `HF_HOME` points at Ocean.
- Model aliases / HuggingFace IDs live in `config.py` (`MODELS`, `QUANTIZED_CHECKPOINTS`), not in these sbatch files.
