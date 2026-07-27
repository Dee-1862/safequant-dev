# Bridges-2 Slurm launchers

Claim-aligned sbatch jobs for SafeQuant. Paths are resolved from `scripts/_env.sh` using environment variables (no usernames in the repo).

| Resource | Typical setup |
|----------|----------------|
| Repo checkout | `$REPO_ROOT` (defaults to directory above `scripts/`) |
| Conda env | `$CONDA_ENV` (default `${OCEAN_BASE}/conda_envs/sq_clean`) |
| HF cache | `$HF_HOME` (default `${OCEAN_BASE}/.cache/huggingface`) |
| Quantized checkpoints | `$QUANTIZED_MODELS` |

Copy `env.example` to `.env` for local overrides. Never commit `.env`.

## Layout

```
scripts/
  _env.sh                 # shared env (sourced by every job)
  setup_env.sh            # one-time login-node bootstrap
  logs/                   # Slurm stdout/stderr (relative paths in #SBATCH)
  tests/                  # smoke jobs
  signal_geometry/        # Table 1
  behavioral_direction/   # Sec. 5 Arditi
  regime_sweep/           # Table 2, PPL, dequant
  localization/           # Tables 3-4
  headline/               # Table 5
  trained_repair/         # Tables 6-7
  boundary/               # Appendix controls
```

## Paper values

```bash
python scripts/paper_values.py > paper_values.json
python scripts/paper_values.py --md > PAPER_TABLES.md
python run_pipeline.py --results
```

Tables: `tab:geometry`, `tab:regime`, `tab:f3a_proj`, `tab:f3a_range`, `tab:f3a`, `tab:controls`, `tab:trained_repair`, `tab:lrqat`, `tab:ppl`, `tab:dequant`, `tab:leakage`, `table:stats`.

## Submit examples

```bash
cd "${REPO_ROOT:-$PWD}"
sbatch scripts/headline/headline_eval_llama.sbatch
sbatch scripts/trained_repair/qresafe_pairs.sbatch
sbatch scripts/regime_sweep/wikitext_ppl.sbatch
```

Override paths at submit time:

```bash
REPO_ROOT=/path/to/safequant OCEAN_BASE=/ocean/projects/.../you sbatch scripts/headline/headline_eval_llama.sbatch
```

## Notes

- Jobs run `python -m experiments.<claim>.<module>` from `$REPO_ROOT`.
- Result JSONs go under `outputs/<claim>/` via `core.manifest.save_result`.
- Q-Resafe pairs default to **anonymized** output (hashes only). The pairs sbatch passes `--store_text` for local DPO training.
- Do not commit `outputs/` or displacement-pair files with full harmful prompt text.
