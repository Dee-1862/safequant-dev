#!/bin/bash
#SBATCH --job-name=sq_x6_repeats
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=32:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x6_repeats_%j.out
#SBATCH --error=scripts/logs/x6_repeats_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X6 (R2 W4): same-prompt repeats for run-to-run variance. Re-runs the headline eval
# on the DEFAULT prompts (same 200 WildJailbreak prompts as the paper run) three times
# (--tag rep1, rep2, rep3). AQLM's 2-bit kernels are not bit-deterministic, so
# repeats measure kernel/run noise. All three repeats run in the SAME env, so
# environment differences cannot leak into the spread. reviewer_runs.py reports
# mean +/- sd over rep1-rep3 only and lists the original headline run
# separately as "paper run" (read only, not in the spread).
#
# Submit:  bash scripts/submit.sh scripts/headline/headline_repeats_aqlm2.sbatch
# Outputs: outputs/headline/headline_<model>_aqlm2_rep{1,2,3}.json  (new names)
# ~7-8 h per run (same as headline_eval_*.sbatch); 6 runs. scripts/queue.sh
# submits one job per cell x rep; by hand: CELLS=phi3 REPS=rep3.

CELLS="${CELLS:-llama7b phi3}"
REPS="${REPS:-rep1 rep2 rep3}"

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
