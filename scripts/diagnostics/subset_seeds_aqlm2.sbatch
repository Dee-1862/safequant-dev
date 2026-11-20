#!/bin/bash
#SBATCH --job-name=sq_x5_subsets
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x5_subsets_%j.out
#SBATCH --error=scripts/logs/x5_subsets_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X5: 5 random prompt subsets (subset_seed 1..5), n_eval=200 each, to put intervals
# around the headline numbers. Sampling is from the full WildJailbreak eval pool;
# global seed stays 42. Existing default runs are untouched.
#
# Cells: the two significant AQLM-2 cells in paired_screen: llama7b + phi3 (default).
# Override with e.g.
#   CELLS="llama7b" bash scripts/submit.sh scripts/regime_sweep/subset_seeds_aqlm2.sbatch
# Set ALSO_RESTORATION=1 to additionally run restoration_sweep per seed (slower).
#
# Output (new names, never overwritten; the job aborts a run if its file exists):
#   outputs/headline/headline_<model>_aqlm2_subset<k>.json
#   outputs/regime_sweep/restoration_sweep_<model>_aqlm2_subset<k>.json   (if ALSO_RESTORATION=1)
# 24 h is the cap for a full 2 cells x 5 seeds run; split it by CELLS/SEEDS if needed
# (Delta's limit is 48 h).

CELLS="${CELLS:-llama7b phi3}"
SEEDS="${SEEDS:-1 2 3 4 5}"

echo "=== X5 prompt-subset runs | cells=${CELLS} seeds=${SEEDS} | $(date) ==="
for MODEL in ${CELLS}; do
  for K in ${SEEDS}; do
    echo "---- ${MODEL}/aqlm2 subset_seed=${K} ----"
    python -u -m experiments.headline.headline_eval \
      --model ${MODEL} --quantizer aqlm2 \
      --n_eval 200 --n_reasoning 100 --n_boot 2000 \
      --subset_seed ${K} \
      || echo "FAILED headline ${MODEL} subset${K}"
    if [[ "${ALSO_RESTORATION:-0}" == "1" ]]; then
      python -u -m experiments.regime_sweep.restoration_sweep \
        --model ${MODEL} --quantizer aqlm2 --n_eval 200 --n_reasoning 100 \
        --subset_seed ${K} \
        || echo "FAILED restoration ${MODEL} subset${K}"
    fi
  done
done
echo "=== done | $(date) ==="
