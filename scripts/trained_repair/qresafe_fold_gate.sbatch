#!/bin/bash
#SBATCH --job-name=sq_fold_gate
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/fold_gate_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/fold_gate_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Zero-delta bit-identity + delta fold smoke before DPO.


echo "=== Q-Resafe fold round-trip gate | $(date) ==="
python -u -m experiments.trained_repair.qresafe_fold
echo "=== delta fold smoke | $(date) ==="
python -u -m experiments.trained_repair.qresafe_fold delta
echo "=== done | $(date) ==="
