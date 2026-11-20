#!/bin/bash
#SBATCH --job-name=sq_loc_flags
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/loc_flags_%j.out
#SBATCH --error=scripts/logs/loc_flags_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Task 3 (R2 W3, R3 W5): rerun the localization tables with per-prompt labels saved
# (flags_by_cond) so scripts/localization_intervals.py can give paired-bootstrap rho
# intervals. Same arguments as the existing launchers, plus --tag flags:
#   per_projection  middle band L11-22   (= per_projection_middle_llama.sbatch)
#   depth_bands     read + write          (= depth_bands_{read,write}.sbatch)
#   param_matched   --seeds 5             (param_matched_restore.sbatch used 3)
#
# Submit everything:  bash scripts/submit.sh scripts/localization/localization_with_flags.sbatch
# Or split by step (what scripts/queue.sh does, so the steps run in parallel):
#   bash scripts/submit.sh scripts/localization/localization_with_flags.sbatch --export=ALL,STEPS=perproj
# Outputs (new names; any that exist are refused and the loop moves on):
#   outputs/localization/per_projection_<m>_aqlm2_L11-22_flags.json
#   outputs/localization/depth_bands_<m>_aqlm2_{read,write}_flags.json
#   outputs/regime_sweep/param_matched_restore_<m>_aqlm2_flags.json

STEPS="${STEPS:-perproj depth pm}"
MODELS="${MODELS:-llama7b phi3}"
TAG="${TAG:-flags}"

echo "=== localization with flags | steps=${STEPS} models=${MODELS} tag=${TAG} | $(date) ==="
for STEP in ${STEPS}; do
  for MODEL in ${MODELS}; do
    case "${STEP}" in
      perproj)
        echo "---- per_projection ${MODEL} L11-22 | $(date) ----"
        python -u -m experiments.localization.per_projection \
          --model "${MODEL}" --quantizer aqlm2 --n_eval 200 --n_reasoning 50 \
          --layer_lo 11 --layer_hi 22 --tag "${TAG}" \
          || echo "FAILED per_projection ${MODEL}"
        ;;
      depth)
        for SIDE in ${SIDES:-read write}; do   # SIDES=read|write runs one side
          echo "---- depth_bands ${MODEL} ${SIDE} | $(date) ----"
          python -u -m experiments.localization.depth_bands \
            --model "${MODEL}" --quantizer aqlm2 --projections "${SIDE}" \
            --n_eval 200 --n_reasoning 100 --tag "${TAG}" \
            || echo "FAILED depth_bands ${MODEL} ${SIDE}"
        done
        ;;
      pm)
        echo "---- param_matched_restore ${MODEL} seeds=5 | $(date) ----"
        python -u -m experiments.regime_sweep.param_matched_restore \
          --model "${MODEL}" --quantizer aqlm2 --n_eval 200 --n_reasoning 50 \
          --seeds 5 --tag "${TAG}" \
          || echo "FAILED param_matched_restore ${MODEL}"
        ;;
      *) echo "unknown step: ${STEP} (use perproj, depth, pm)" >&2 ;;
    esac
  done
done
echo "=== done | $(date) ==="
