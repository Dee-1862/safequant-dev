#!/bin/bash
#SBATCH --job-name=sq_arc_check
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=0:45:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/arc_check_%j.out
#SBATCH --error=scripts/logs/arc_check_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# ARC-Challenge (first 100 items, the paper's set) on ONE quantized cell, printed next
# to the paper's quantized/FP32 ARC and chance (25%). Example:
#   bash scripts/submit.sh scripts/regime_sweep/arc_check.sbatch \
#     --export=ALL,EXP_MODEL=llama7b,EXP_QUANT=gptq2,ARC_TAG=rebuilt_gptqmodel
# Output: outputs/regime_sweep/arc_check_<m>_<q>_<tag>.json (never overwrites).
set -euo pipefail
MODEL="${EXP_MODEL:?set EXP_MODEL}"
QUANT="${EXP_QUANT:?set EXP_QUANT}"
# Default tag follows the load path: GPTQ rebuilt = dense copy (SQ_GPTQ_DENSE=1),
# rebuilt_gptqmodel = gptqmodel; other quantizers rerun
if [[ "$QUANT" != gptq* ]]; then DEFTAG=rerun
elif [[ "${SQ_GPTQ_DENSE:-0}" == 1 ]]; then DEFTAG=rebuilt
else DEFTAG=rebuilt_gptqmodel; fi
TAG="${ARC_TAG:-$DEFTAG}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [[ "$QUANT" == gptq* ]]; then
  CKPT="${QM_ROOT}/gptq/${MODEL}-gptq${QUANT#gptq}bit-g128"
  [[ -f "${CKPT}/config.json" ]] || { echo "ERROR: checkpoint not found at ${CKPT}" >&2; exit 1; }
  python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"
fi
echo "=== ARC check | ${MODEL}/${QUANT} | tag=${TAG} | SQ_GPTQ_DENSE=${SQ_GPTQ_DENSE:-0} | $(date) ==="
python -u scripts/regime_sweep/arc_check.py --model "${MODEL}" --quantizer "${QUANT}" \
  --n_reasoning 100 --tag "${TAG}"
echo "=== done | $(date) ==="
