#!/bin/bash
#SBATCH --job-name=sq_arc_check
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=0:45:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/arc_check_%j.out
#SBATCH --error=scripts/logs/arc_check_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# ARC-Challenge (first 100 items, the paper's set) on ONE quantized cell, printed next
# to the paper's quantized/FP32 ARC and chance (25%). Example:
#   bash scripts/submit.sh scripts/regime_sweep/arc_check.sbatch \
#     --export=ALL,EXP_MODEL=llama7b,EXP_QUANT=gptq2,ARC_TAG=rebuilt_unmod
# Output: outputs/regime_sweep/arc_check_<m>_<q>_<tag>.json (never overwrites).
set -euo pipefail
MODEL="${EXP_MODEL:?set EXP_MODEL}"
QUANT="${EXP_QUANT:?set EXP_QUANT}"
TAG="${ARC_TAG:-rebuilt_unmod}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [[ "$QUANT" == gptq* ]]; then
  CKPT="${QM_ROOT}/gptq/${MODEL}-gptq${QUANT#gptq}bit-g128"
  [[ -f "${CKPT}/config.json" ]] || { echo "ERROR: checkpoint not found at ${CKPT}" >&2; exit 1; }
  python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"
fi
echo "=== ARC check | ${MODEL}/${QUANT} | tag=${TAG} | $(date) ==="
python -u scripts/regime_sweep/arc_check.py --model "${MODEL}" --quantizer "${QUANT}" \
  --n_reasoning 100 --tag "${TAG}"
echo "=== done | $(date) ==="
