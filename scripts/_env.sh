#!/bin/bash
# Shared Bridges-2 environment for SafeQuant sbatch jobs.
# Source from any job script after the #SBATCH headers:
#   source "$(dirname "$0")/../_env.sh"   # from scripts/<claim>/
#   source "$(dirname "$0")/_env.sh"      # from scripts/
#
# Override on the submit line if needed:
#   REPO_ROOT=/jet/home/$USER/projects/safequant sbatch scripts/headline/headline_eval_llama.sbatch

set -euo pipefail

# Account / paths on Bridges-2 (PSC)
# - Code lives on $HOME Jet: small quota, use for git checkout only
# - HF models + big caches live on Ocean project storage
export REPO_ROOT="${REPO_ROOT:-/jet/home/bhoomire/projects/safequant_v2}"
export OCEAN_BASE="${OCEAN_BASE:-/ocean/projects/cis240116p/bhoomire}"
export CONDA_ENV="${CONDA_ENV:-${OCEAN_BASE}/conda_envs/sq_clean}"
export HF_HOME="${HF_HOME:-${OCEAN_BASE}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/opt/packages/cuda/v12.4.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Logs for this job tree
export SQ_LOG_DIR="${SQ_LOG_DIR:-${REPO_ROOT}/scripts/logs}"
mkdir -p "${SQ_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Conda + CUDA toolchain
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
module load gcc/10.2.0 2>/dev/null || true

cd "${REPO_ROOT}"

echo "=== SafeQuant Bridges-2 | job=${SLURM_JOB_ID:-local} | $(date) ==="
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  CONDA_ENV=${CONDA_ENV}"
echo "  HF_HOME=${HF_HOME}"
echo "  CUDA_HOME=${CUDA_HOME}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
