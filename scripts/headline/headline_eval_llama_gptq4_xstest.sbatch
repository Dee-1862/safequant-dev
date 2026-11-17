#!/bin/bash
#SBATCH --job-name=sq_x1_xstest
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x1_xstest_llama_gptq4_%j.out
#SBATCH --error=scripts/logs/x1_xstest_llama_gptq4_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X1 (reviewer item, top priority): XSTest false-refusal on Llama GPTQ-4.
# headline_eval already runs XSTest; this just runs the GPTQ-4 cell.
# Submit:  bash scripts/submit.sh scripts/headline/headline_eval_llama_gptq4_xstest.sbatch
# Output:  outputs/headline/headline_llama_gptq4_x1_n200.json   (new name; never overwrites)
#
# Needs the self-generated checkpoint  ${QM_ROOT}/gptq/llama7b-gptq4bit-g128 .
# If it is missing, build it first (sq_autogptq env, separate job):
#   MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh
# then re-submit this job once that job has finished.

CKPT="${QM_ROOT}/gptq/llama7b-gptq4bit-g128"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "ERROR: GPTQ-4 checkpoint not found at ${CKPT}" >&2
  echo "Build it:  MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh   (then re-submit)" >&2
  exit 1
fi

echo "=== X1 headline eval Llama GPTQ-4 (XSTest) | $(date) ==="
python -u -m experiments.headline.headline_eval \
  --model llama7b --quantizer gptq4 \
  --n_eval 200 --n_reasoning 100 --n_boot 2000 \
  --tag x1_n200
echo "=== done | $(date) ==="
