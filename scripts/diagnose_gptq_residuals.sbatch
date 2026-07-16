#!/bin/bash
#SBATCH --job-name=sq_gptq_diag
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=0:40:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/gptq_diag_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/gptq_diag_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

MODEL="${DIAG_MODEL:-llama7b}"
QUANTIZER="${DIAG_QUANTIZER:-gptq4}"

echo "=== gptq residual diagnostic | ${MODEL} / ${QUANTIZER} | $(date) ==="
python -u -m scripts.diagnose_gptq_residuals --model "${MODEL}" --quantizer "${QUANTIZER}" --n 24
echo "=== done | $(date) ==="
