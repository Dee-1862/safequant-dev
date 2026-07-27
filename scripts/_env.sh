#!/bin/bash
# Shared HPC environment for SafeQuant sbatch jobs.
# Source from any job script after the #SBATCH headers:
#   source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
#
# Override on the submit line when paths differ:
#   REPO_ROOT=/path/to/safequant OCEAN_BASE=/ocean/.../you sbatch scripts/...

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
export OCEAN_BASE="${OCEAN_BASE:-${PROJECT:-${HOME}/safequant_ocean}}"
export CONDA_ENV="${CONDA_ENV:-${OCEAN_BASE}/conda_envs/sq_clean}"
export HF_HOME="${HF_HOME:-${OCEAN_BASE}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export QUANTIZED_MODELS="${QUANTIZED_MODELS:-${OCEAN_BASE}/quantized_models}"
export AQLM_REPO="${AQLM_REPO:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/opt/packages/cuda/v12.4.0}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export SQ_LOG_DIR="${SQ_LOG_DIR:-${REPO_ROOT}/scripts/logs}"
mkdir -p "${SQ_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ -f /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh ]]; then
  source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
  module load gcc/10.2.0 2>/dev/null || true
fi

cd "${REPO_ROOT}"

echo "=== SafeQuant | job=${SLURM_JOB_ID:-local} | $(date) ==="
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  CONDA_ENV=${CONDA_ENV}"
echo "  HF_HOME=${HF_HOME}"
echo "  QUANTIZED_MODELS=${QUANTIZED_MODELS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
