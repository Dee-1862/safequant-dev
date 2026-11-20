#!/bin/bash
# Shared environment for SafeQuant sbatch jobs. Loads .env from repo root.
#   source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"   (sbatch; run from repo root)
#
# Cluster-specific bits come from .env (see .env.example):
#   CONDA_SH   path to conda.sh. Empty -> assume conda is already on PATH
#              (e.g. via MODULES).
#   CUDA_HOME  only exported when set.
#   MODULES    space-separated list passed to `module load` before conda activate.
#   CONDA_ENV  env to activate (skipped when empty).

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/load_env.sh"

export REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export QM_ROOT="${QM_ROOT:-${QUANTIZED_MODELS:-${HOME}/quantized_models}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
# Size CPU thread pools to the Slurm allocation. Unset, torch/OpenMP/BLAS start a pool
# for the whole node (GPTQ build test: 298 threads on 1 allowed CPU, ~6x slower).
# Speed only; a job's own OMP_NUM_THREADS wins.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export SQ_LOG_DIR="${SQ_LOG_DIR:-${REPO_ROOT}/scripts/logs}"
# GPTQ via dense copies (core/model_loader.py): where $QM_ROOT/gptq_dense exists,
# every GPTQ cell (2/3/4-bit) loads the fp16 weights auto-gptq 0.7.1 computes, so all
# GPTQ results share one load path. A missing dense copy is an error, never a fallback.
# SQ_GPTQ_DENSE=0 (in .env or --export) restores the gptqmodel path.
if [[ -z "${SQ_GPTQ_DENSE+x}" && -d "${QM_ROOT}/gptq_dense" ]]; then
  export SQ_GPTQ_DENSE=1
fi

mkdir -p "${SQ_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${QM_ROOT}"

# --- modules, CUDA, conda ----------------------------------------------------
if [[ -n "${MODULES:-}" ]]; then
  if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load ${MODULES} || echo "WARN: 'module load ${MODULES}' failed" >&2
  else
    echo "WARN: MODULES set but 'module' command not found" >&2
  fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}"
  else
    echo "WARN: CONDA_ENV set but conda not found (set CONDA_SH or MODULES)" >&2
  fi
fi

cd "${REPO_ROOT}"

echo "=== SafeQuant | job=${SLURM_JOB_ID:-local} | $(date) ==="
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  HF_HOME=${HF_HOME}"
echo "  QM_ROOT=${QM_ROOT}"
echo "  SQ_GPTQ_DENSE=${SQ_GPTQ_DENSE:-<unset>}"
echo "  CONDA_ENV=${CONDA_ENV:-<none>}  MODULES=${MODULES:-<none>}  CUDA_HOME=${CUDA_HOME:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
