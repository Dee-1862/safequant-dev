#!/bin/bash
#SBATCH --job-name=sq_smoke_env
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/smoke_env_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/smoke_env_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Fast plumbing check: env + CUDA + pytest + seed. No full model experiments.

set -euo pipefail

echo "=== smoke env check | $(date) ==="

python - <<'PY'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not visible on this allocation")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

python - <<'PY'
from pathlib import Path
from core.seeds import set_global_seed, DEFAULT_SEED
import numpy as np

assert DEFAULT_SEED == 42
set_global_seed()
got = np.random.default_rng(42).integers(0, 100, size=5).tolist()
print(f"seed fingerprint={got}")
assert got == [8, 77, 65, 43, 43], got

import core.datasets
import core.model_loader
import core.manifest
print("core imports ok")
print(f"HF_HOME writable check -> {Path('.').resolve()}")
PY

echo "--- pytest ---"
python -m pytest tests/ -v
echo "=== smoke env OK | $(date) ==="
