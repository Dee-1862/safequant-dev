#!/bin/bash
#SBATCH --job-name=sq_qresafe
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/qresafe_dpo_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/qresafe_dpo_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# Submit: sbatch scripts/trained_repair/qresafe_dpo.sbatch read|write|full


MASK="${1:-read}"
echo "=== Q-Resafe DPO | mask=${MASK} | $(date) ==="
python -u -m experiments.trained_repair.qresafe_dpo \
  --mask "${MASK}" --model llama7b --quantizer aqlm2 --n_eval 400
echo "=== done | $(date) ==="
