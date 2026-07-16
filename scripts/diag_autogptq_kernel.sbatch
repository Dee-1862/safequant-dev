#!/bin/bash
#SBATCH --job-name=sq_ag_kernel
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=0:40:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/ag_kernel_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/ag_kernel_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

echo "=== auto-gptq exLlama kernel test | $(date) ==="
python -u -m scripts.diag_autogptq_kernel --model "${AG_MODEL:-llama7b}" --n 24
echo "=== done | $(date) ==="
