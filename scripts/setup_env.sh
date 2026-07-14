#!/bin/bash
# One-time Bridges-2 env bootstrap (login node). Installs missing packages into sq_clean.
# Usage: bash scripts/setup_env.sh

set -euo pipefail

CONDA_BASE=/opt/packages/anaconda3-2024.10-1
OCEAN_BASE=/ocean/projects/cis240116p/bhoomire
CONDA_ENV="${OCEAN_BASE}/conda_envs/sq_clean"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "Activated: ${CONDA_ENV}"
echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"

mkdir -p "${OCEAN_BASE}/.cache/huggingface"
mkdir -p "${OCEAN_BASE}/.cache/huggingface/datasets"

pip install -q "bitsandbytes>=0.43.0" "accelerate>=0.26.0" "scipy>=1.11.0" "pytest>=8.0"

python -c "
import torch, transformers, bitsandbytes as bnb, accelerate
print(f'torch={torch.__version__} transformers={transformers.__version__}')
print(f'bitsandbytes={bnb.__version__} accelerate={accelerate.__version__}')
print(f'CUDA (torch): {torch.version.cuda}')
print('Note: torch.cuda.is_available() is False on login nodes; True on GPU allocations.')
"

echo "Setup complete. Example: sbatch scripts/headline/headline_eval_llama.sbatch"
