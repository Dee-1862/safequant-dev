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
  tests/                  # smoke jobs — run these before the full grid
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

## Paper values (LaTeX tables)

Pull numbers keyed to each paper table label:

```bash
python scripts/paper_values.py > paper_values.json
python scripts/paper_values.py --md > PAPER_TABLES.md
```

Tables: `tab:geometry`, `tab:regime`, `tab:f3a_proj`, `tab:f3a_range`, `tab:f3a`, `tab:controls`, `tab:trained_repair`, `tab:lrqat`.

Incomplete Q-Resafe read/write r128 jobs show as `missing or running` until JSON lands.

## Smoke before scaling

Do **not** mass-submit multi-model jobs until the smoke chain is green. Details: [`scripts/tests/README.md`](tests/README.md).

```bash
cd /jet/home/bhoomire/projects/safequant_v2
pytest tests/ -v                                          # login node; or: python -m pytest tests/ -v
# if missing: pip install pytest
bash scripts/tests/submit_smoke.sh                        # env → llama+phi → pairs
# or one at a time:
sbatch scripts/tests/00_env_check.sbatch
sbatch scripts/tests/01_smoke_llama_aqlm2.sbatch
sbatch scripts/tests/03_smoke_phi3_aqlm2.sbatch
```

## Submit examples (full paper runs)

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
