#!/bin/bash
#SBATCH --job-name=sq_diag_gptq3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/diag_gptq3_%j.out
#SBATCH --error=scripts/logs/diag_gptq3_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Why do rebuilt GPTQ-3 checkpoints give PPL = nan? For each model (DIAG_MODELS,
# default "phi3 llama7b"), on the same 20 WikiText-2 windows:
#   1) gptqmodel fp16  (the pipeline path, sq_clean)
#   2) gptqmodel bf16  (overflow test)
#   3) auto-gptq 0.7.1 from_quantized (sq_autogptq; checks the checkpoint itself)
# Prints RESULT lines and the first module with a non-finite output. Writes nothing
# to outputs/. Submit:  bash scripts/submit.sh scripts/tests/diag_gptq3.sbatch --cpus-per-task=8
set -uo pipefail   # not -e: run every method even if one fails
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
for MODEL in ${DIAG_MODELS:-phi3 llama7b}; do
  python -u scripts/fix_autogptq_checkpoints.py "${QM_ROOT}/gptq/${MODEL}-gptq3bit-g128"
  python -u scripts/tests/diag_gptq3.py --model "$MODEL" --method gptqmodel --dtype fp16
  python -u scripts/tests/diag_gptq3.py --model "$MODEL" --method gptqmodel --dtype bf16
  "${AGENV}/bin/python" -u scripts/tests/diag_gptq3.py --model "$MODEL" --method autogptq
done
echo "=== summary"; grep -h '^RESULT\|first non-finite' "scripts/logs/diag_gptq3_${SLURM_JOB_ID}.out" || true
echo "=== done | $(date) ==="
