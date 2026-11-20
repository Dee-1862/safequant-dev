#!/bin/bash
#SBATCH --job-name=sq_hpiece
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=1:55:00
#SBATCH --output=scripts/logs/hpiece_%j.out
#SBATCH --error=scripts/logs/hpiece_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Headline-eval pieces (experiments/headline/headline_piece.py), run back to back in one job.
# PIECES = ';'-separated argument lists, one per piece (use '+' between --parts, e.g. wj+arc).
#   PIECES="--group heldout --model llama3-8b --quantizer aqlm2 --condition fp32 --parts wj+arc" \
#   SLURM_PARTITION="$SLURM_INTERACTIVE_PARTITION" bash scripts/submit.sh scripts/headline/headline_piece.sbatch --time=1:55:00
# Merge afterwards on CPU: python -m experiments.headline.merge_pieces --group ... (see that file).
set -uo pipefail
FAIL=0
echo "=== headline pieces | job ${SLURM_JOB_ID} | $(hostname) | $(git rev-parse --short HEAD) | $(date) ==="
IFS=';' read -ra P <<< "${PIECES:?set PIECES}"
for a in "${P[@]}"; do
  [[ -z "${a// }" ]] && continue
  echo "---- piece: ${a} | $(date) ----"
  # shellcheck disable=SC2086
  python -u -m experiments.headline.headline_piece ${a} || { echo "FAILED piece: ${a}"; FAIL=1; }
done
echo "=== done | $(date) | exit ${FAIL} ==="
exit "${FAIL}"
