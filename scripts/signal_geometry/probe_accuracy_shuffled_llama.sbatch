#!/bin/bash
#SBATCH --job-name=sq_x3_shuffle
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x3_shuffle_llama_%j.out
#SBATCH --error=scripts/logs/x3_shuffle_llama_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X3: shuffled-label probe control. Same pipeline as probe_accuracy, but harmful/harmless
# labels are permuted (seed 42) before fitting. Expect ~50% accuracy at every layer.
# Submit:  bash scripts/submit.sh scripts/signal_geometry/probe_accuracy_shuffled_llama.sbatch
# Output:  outputs/signal_geometry/probe_accuracy_<model>_<quant>_shuffled.json  (separate files)

echo "=== X3 shuffled-label probe | $(date) ==="
for MODEL in ${SWEEP_MODELS:-llama7b}; do
  for QUANTIZER in ${SWEEP_QUANTIZERS:-aqlm2}; do
    echo "---- ${MODEL} / ${QUANTIZER} (shuffled) ----"
    python -u -m experiments.signal_geometry.probe_accuracy \
      --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200 --probe_type linear \
      --shuffle_labels \
      || echo "FAILED ${MODEL}/${QUANTIZER}"
  done
done
echo "=== done | $(date) ==="
