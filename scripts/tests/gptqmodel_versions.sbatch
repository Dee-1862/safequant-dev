#!/bin/bash
#SBATCH --job-name=sq_gq_versions
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:30:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/gq_versions_%j.out
#SBATCH --error=scripts/logs/gq_versions_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Which gptqmodel version loads rebuilt GPTQ-3 / GPTQ-2 checkpoints correctly with
# sq_clean's transformers 4.57? For each candidate installed under
# $GQ_TEST_DIR/<version> (pip --target, --no-deps; sq_clean itself untouched) plus the
# env's own 5.7.0, load phi3 and llama7b at 3 and 2 bits and compute PPL on 20
# WikiText-2 windows; auto-gptq 0.7.1's own loader is the reference (it wrote the
# checkpoints). Prints RESULT lines only. Submit (interactive partition = fast start):
#   SLURM_PARTITION="$SLURM_INTERACTIVE_PARTITION" bash scripts/submit.sh \
#     scripts/tests/gptqmodel_versions.sbatch --cpus-per-task=8
set -uo pipefail   # not -e: every combination runs even if one fails
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
GQ_TEST_DIR="${GQ_TEST_DIR:-$(dirname "${CONDA_ENV}")/gptqmodel_test}"
VERSIONS="${GQ_VERSIONS:-7.1.0 5.6.12 5.4.2 5.0.0 4.2.5}"
MODELS="${GQ_MODELS:-phi3 llama7b}"
BITS="${GQ_BITS:-3 2}"
D=scripts/tests/diag_gptq3.py

for M in $MODELS; do
  for b in $BITS; do
    python -u scripts/fix_autogptq_checkpoints.py "${QM_ROOT}/gptq/${M}-gptq${b}bit-g128" >/dev/null
    echo "################ ${M} gptq${b}"
    "${AGENV}/bin/python" -u "$D" --model "$M" --bits "$b" --method autogptq       # reference
    python -u "$D" --model "$M" --bits "$b" --method gptqmodel --dtype fp16        # env (5.7.0)
    for V in $VERSIONS; do
      if [[ -d "${GQ_TEST_DIR}/${V}/gptqmodel" ]]; then
        PYTHONPATH="${GQ_TEST_DIR}/${V}${PYTHONPATH:+:${PYTHONPATH}}" \
          python -u "$D" --model "$M" --bits "$b" --method gptqmodel --dtype fp16
      else
        echo "RESULT ${M} gptq${b} | gptqmodel ${V}: SKIPPED (not installed in ${GQ_TEST_DIR})"
      fi
    done
  done
done
echo "=== summary"; grep -hE '^RESULT|^####' "scripts/logs/gq_versions_${SLURM_JOB_ID}.out" || true
echo "=== done | $(date) ==="
