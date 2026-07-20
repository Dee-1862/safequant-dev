#!/bin/bash
#SBATCH --job-name=sq_leak
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/leak_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/leak_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# RQ1 leakage control: FP32 refusal probe vs TF-IDF surface baseline on XSTest
# (safe = surface-harmful-but-benign) + HarmBench transfer. FP32 only.
echo "=== probe leakage control | ${EXP_MODEL:-llama7b} | $(date) ==="
python -u -m experiments.boundary.probe_leakage_control \
  --model ${EXP_MODEL:-llama7b} --n_calib 200 --n_transfer 200
echo "=== done | $(date) ==="
