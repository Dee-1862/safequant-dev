# Bridges-2 smoke tests

Tiny GPU jobs to prove the checkout, env, seed, and model cells work **before** you submit the full multi-model grid.

Use these only for plumbing. Numbers from smoke runs are **not** paper results.

## Order of operations

```text
1. Login node   →  git pull + pytest (CPU)
2. Env check    →  sbatch scripts/tests/00_env_check.sbatch
3. GPU smoke    →  llama/aqlm2 + phi3/aqlm2 (can run in parallel)
4. Optional     →  qresafe pairs smokes (llama, phi3)
5. Full grid    →  only after 1–3 look clean
```

Or submit the full chain with Slurm dependencies:

```bash
bash scripts/tests/submit_smoke.sh
SKIP_PAIRS=1 bash scripts/tests/submit_smoke.sh   # env + both cells only
SKIP_LLAMA=1 bash scripts/tests/submit_smoke.sh   # phi only
SKIP_PHI=1 bash scripts/tests/submit_smoke.sh     # llama only
```

## 1. Login node (no GPU)

```bash
cd /jet/home/bhoomire/projects/safequant_v2   # or your REPO_ROOT
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean

# Unit tests (seed + anchors). Must be 14 passed.
pytest tests/ -v

# Seed fingerprint from docs/SEEDS.md
python -c "
from core.seeds import set_global_seed
import numpy as np
set_global_seed()
print(np.random.default_rng(42).integers(0, 100, size=5).tolist())
"
# Expected: [8, 77, 65, 43, 43]
```

## 2. Env check job (~minutes)

```bash
sbatch scripts/tests/00_env_check.sbatch
tail -f scripts/logs/smoke_env_*.out
```

Checks: conda, CUDA visible, HF cache writable, `pytest`, seed fingerprint, core imports.

## 3. GPU cell smokes (tiny `n_*`)

```bash
sbatch scripts/tests/01_smoke_llama_aqlm2.sbatch
sbatch scripts/tests/03_smoke_phi3_aqlm2.sbatch
tail -f scripts/logs/smoke_llama_aqlm2_*.out
tail -f scripts/logs/smoke_phi3_aqlm2_*.out
```

Each runs, with small sample sizes:
- `direction_drift`
- `probe_accuracy`
- `headline_eval`

Phi-3 specifically exercises the fused `qkv_proj` / `gate_up_proj` layout (and the pinned FP revision in `config.py`).

Success looks like: job exits 0, JSONs under `outputs/`. Cos-sim / ASR at these `n_*` are noisy — ignore magnitude.

## 4. Optional Q-Resafe pairs smokes

```bash
sbatch scripts/tests/02_smoke_qresafe_pairs.sbatch          # llama
sbatch scripts/tests/04_smoke_qresafe_pairs_phi3.sbatch     # phi3
```

Skip until cell smokes are green; still expensive vs step 3.

## After smoke is green

Submit full claim jobs from `scripts/<claim>/` (see `scripts/README.md`). Do **not** use smoke `n_*` for paper tables.

```bash
sbatch scripts/headline/headline_eval_llama.sbatch
sbatch scripts/headline/headline_eval_phi3.sbatch
# ...
```

## Watching jobs

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls -lt outputs/signal_geometry/ outputs/headline/ | head
```

## Overrides

```bash
REPO_ROOT=/jet/home/bhoomire/projects/safequant sbatch scripts/tests/03_smoke_phi3_aqlm2.sbatch
```
