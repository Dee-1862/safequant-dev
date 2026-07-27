#!/bin/bash
# One-time HPC env bootstrap (login node). Installs missing packages into sq_clean.
# Usage: bash scripts/setup_env.sh
# Set OCEAN_BASE and CONDA_ENV before running if not using defaults from _env.sh.

set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "Activated: ${CONDA_ENV}"
echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${QUANTIZED_MODELS}"

pip install -q "bitsandbytes>=0.43.0" "accelerate>=0.26.0" "scipy>=1.11.0" "pytest>=8.0"

python -c "
import torch, transformers, bitsandbytes as bnb, accelerate
print(f'torch={torch.__version__} transformers={transformers.__version__}')
print(f'bitsandbytes={bnb.__version__} accelerate={accelerate.__version__}')
print(f'CUDA (torch): {torch.version.cuda}')
print('Note: torch.cuda.is_available() is False on login nodes; True on GPU allocations.')
"

echo "Setup complete. Example: sbatch scripts/headline/headline_eval_llama.sbatch"
