#!/bin/bash
#SBATCH --job-name=sq_cosmed_restore
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/cosmed_restore_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/cosmed_restore_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# RQ1 mediation control: cosmed under quant / +read / +write / +both restore.
echo "=== cosmed after restoration | ${EXP_MODEL:-llama7b}/aqlm2 | $(date) ==="
python -u -m experiments.signal_geometry.cosmed_after_restore \
  --model ${EXP_MODEL:-llama7b} --quantizer aqlm2 --n_calib 200
echo "=== done | $(date) ==="
