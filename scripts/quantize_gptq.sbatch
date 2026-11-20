#!/bin/bash
#SBATCH --job-name=quantize_gptq
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/quantize_gptq_%j.out
#SBATCH --error=scripts/logs/quantize_gptq_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Reproducible GPTQ quantization of one base model via GPTQModel.
# Submit one per model to run them in parallel:
#   sbatch scripts/quantize_gptq.sbatch llama7b
#   sbatch scripts/quantize_gptq.sbatch phi3
#   sbatch scripts/quantize_gptq.sbatch mistral7b-v02
#   sbatch scripts/quantize_gptq.sbatch qwen7b
# Optional 2nd/3rd args override bits/group_size:  sbatch ... phi3 4 128
# GPTQ_BACKEND=autogptq (default; run in the sq_autogptq env, CONDA_ENV=$AGENV) or
# gptqmodel (run in sq_clean). scripts/quantize_gptq_all.sh picks the env to match.

set -euo pipefail

MODEL="${1:-llama7b}"
BITS="${2:-4}"
GROUP="${3:-128}"
export GPTQ_BACKEND="${GPTQ_BACKEND:-autogptq}"
# Size CPU thread pools to the allocation. Unset, torch/OpenMP start a pool for the
# whole node (phi3 test: 298 threads on 1 allowed CPU), and auto-gptq's CPU packing
# slowed from ~2 min to over an hour. Affects speed only.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
echo "  threads: OMP_NUM_THREADS=${OMP_NUM_THREADS} (cpus-per-task=${SLURM_CPUS_PER_TASK:-unset})"

echo "=== GPTQ quantize ${MODEL} (bits=${BITS}, group=${GROUP}, backend=${GPTQ_BACKEND}) | $(date) ==="
# GPTQ_OUTPUT_DIR: build into another folder (e.g. an unmodified-packing reference);
# default is the pipeline path $QM_ROOT/gptq/<model>-gptq<bits>bit-g<group>.
OUT_ARGS=(); [[ -n "${GPTQ_OUTPUT_DIR:-}" ]] && OUT_ARGS=(--output_dir "${GPTQ_OUTPUT_DIR}")
python -u scripts/quantize_gptq.py \
  --model "${MODEL}" --bits "${BITS}" --group_size "${GROUP}" \
  --n_calib 128 --seqlen 2048 --seed 42 --backend "${GPTQ_BACKEND}" "${OUT_ARGS[@]}"
echo "=== done ${MODEL} | $(date) ==="
