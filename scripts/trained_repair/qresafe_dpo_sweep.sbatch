#!/bin/bash
#SBATCH --job-name=sq_qresafe_sw
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/qresafe_sweep_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/qresafe_sweep_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# Replaces dozens of scripts_old/qresafe_*_r*.sbatch one-offs.


echo "=== Q-Resafe DPO rank/mask sweep | $(date) ==="
# Compact sweep covering paper Table 6–7 cells; edit Ranks/Masks as needed.
for MASK in read write full; do
  for R in 128 256 512 1024 2048; do
    ALPHA=$((R * 2))
    echo "---- mask=${MASK} r=${R} alpha=${ALPHA} ----"
    python -u -m experiments.trained_repair.qresafe_dpo \
      --mask ${MASK} --r ${R} --alpha ${ALPHA} \
      --beta 0.01 --epochs 1 --lr 5e-7 \
      --n_eval 200 --opt8bit --seed 42 \
      || echo "FAILED ${MASK}/r${R}"
  done
done
echo "=== done | $(date) ==="
