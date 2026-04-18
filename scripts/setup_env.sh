#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh
#
# Install all missing packages into the existing my_env conda environment on
# Bridges-2 (PSC).  Run this ONCE from the login node before submitting jobs.
#
# Usage:
#   bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

CONDA_BASE=/opt/packages/anaconda3-2024.10-1
ENV_NAME=my_env
OCEAN_DIR=/ocean/projects/cis240116p/bhoomire

# ── Activate conda ────────────────────────────────────────────────────────────
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "✓ Activated conda env: ${ENV_NAME}"
echo "  Python: $(python --version)"
echo "  Torch:  $(python -c 'import torch; print(torch.__version__)')"

# ── Create Ocean cache dirs ───────────────────────────────────────────────────
# HuggingFace models can be 3-10GB each. HOME has a 25GB quota.
# Always cache to Ocean (no quota issues for $PROJECT).
mkdir -p "${OCEAN_DIR}/.cache/huggingface"
mkdir -p "${OCEAN_DIR}/outputs"
echo "✓ Ocean cache dir: ${OCEAN_DIR}/.cache/huggingface"

# ── Install missing packages ──────────────────────────────────────────────────
echo ""
echo "Installing missing packages..."

# bitsandbytes: NF4 quantization (requires CUDA, must match torch cu118)
pip install bitsandbytes>=0.43.0

# accelerate: required by transformers for device_map="auto"
pip install accelerate>=0.26.0

# matplotlib: for any visualisation scripts
pip install matplotlib>=3.7.0

# scipy: used by some diagnostic scripts
pip install scipy>=1.11.0

echo ""
echo "✓ All packages installed."

# ── Verify GPU-critical imports ───────────────────────────────────────────────
echo ""
echo "Verifying imports..."
python -c "
import torch
import transformers
import bitsandbytes as bnb
import accelerate
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  bitsandbytes {bnb.__version__}')
print(f'  accelerate   {accelerate.__version__}')
print(f'  CUDA in torch: {torch.version.cuda}')
print()
print('NOTE: torch.cuda.is_available() will be False on login nodes.')
print('      It will be True on GPU nodes (v100/l40s/h100) during jobs.')
"

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "Setup complete. Submit jobs with:"
echo "  sbatch scripts/run_pipeline.sbatch"
echo "────────────────────────────────────────────────────────────────────────"
