#!/bin/bash
#SBATCH --job-name=sq_x4_refit100
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x4_refit100_llama_%j.out
#SBATCH --error=scripts/logs/x4_refit100_llama_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X4: 100-split FP-vs-FP probe refit floor. Reports the 95th percentile of the refit
# distance (1 - cos) in summary_splits.refit_distance_p95_pooled (and per layer).
# Split 0 reproduces the original single-split control exactly.
# Submit:  bash scripts/submit.sh scripts/boundary/probe_refit_control_splits100_llama.sbatch
# Output:  outputs/boundary/probe_refit_control_{llama,phi3}_splits100.json  (new names)
# CPU-heavy (100 splits x layers x 2 logistic fits on 4096-d); the GPU is only used for
# the one-off FP residual extraction. Default runs llama7b and phi3; set MODELS="llama7b" for one.

echo "=== X4 probe refit control, 100 splits | $(date) ==="
for MODEL in ${MODELS:-llama7b phi3}; do
  python -u -m experiments.boundary.probe_refit_control \
    --model ${MODEL} --n_calib 200 --n_splits 100 \
    || echo "FAILED ${MODEL}"
done
echo "=== done | $(date) ==="
