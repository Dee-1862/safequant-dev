#!/bin/bash
#SBATCH --job-name=sq_gptq_qc
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=0:30:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/gptq_qc_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/gptq_qc_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# Coherence check: does our GPTQ model generate sensible text?
# COHERENT -> measurement bug in direction_drift.  GARBAGE -> kernel/toolchain.
MODEL="${QC_MODEL:-llama7b}"
QUANTIZER="${QC_QUANTIZER:-gptq4}"

echo "=== gptq quality check | ${MODEL} / ${QUANTIZER} | $(date) ==="
python -u -m scripts.check_gptq_quality --model "${MODEL}" --quantizer "${QUANTIZER}"
echo "=== done | $(date) ==="
