#!/bin/bash
#SBATCH --job-name=sq_x6_repeats
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=32:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x6_repeats_%j.out
#SBATCH --error=scripts/logs/x6_repeats_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X6 (R2 W4): same-prompt repeats for run-to-run variance. Re-runs the headline eval
# on the DEFAULT prompts (same 200 WildJailbreak prompts as the paper run) with
# --tag rep2 and --tag rep3. AQLM's 2-bit kernels are not bit-deterministic, so
# repeats measure kernel/run noise. reviewer_runs.py combines them with the original
# headline_<model>_aqlm2.json (read only) into mean +/- sd.
#
# Submit:  bash scripts/submit.sh scripts/headline/headline_repeats_aqlm2.sbatch
# Outputs: outputs/headline/headline_<model>_aqlm2_rep{2,3}.json  (new names)
# ~7-8 h per run (same as headline_eval_*.sbatch); 4 runs. Split with CELLS / REPS
# if needed, e.g. CELLS=phi3 REPS=rep3.

CELLS="${CELLS:-llama7b phi3}"
REPS="${REPS:-rep2 rep3}"

echo "=== X6 same-prompt repeats | cells=${CELLS} reps=${REPS} | $(date) ==="
for MODEL in ${CELLS}; do
  for REP in ${REPS}; do
    echo "---- ${MODEL}/aqlm2 ${REP} | $(date) ----"
    python -u -m experiments.headline.headline_eval \
      --model "${MODEL}" --quantizer aqlm2 \
      --n_eval 200 --n_reasoning 100 --n_boot 2000 \
      --tag "${REP}" \
      || echo "FAILED ${MODEL} ${REP}"
  done
done
echo "=== done | $(date) ==="
