#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Linear refusal probe — Mistral-7B-v0.2 AQLM-2.
#
# Cross-architecture test of the "detection preserved" claim. Mistral-AQLM-2
# is the strongest F2 self-dominance condition (asymΔ=+29pp). If detection
# is preserved here, the two-stage / signal+consumption picture survives
# the cross-arch self-dominance case (i.e. detection doesn't depend on
# whether F2 is cross-dom or self-dom).
#
# Per condition: ~25 min. Walltime 2h with margin.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_probe_mistral_lin
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_mistral_lin_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_mistral_lin_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Linear probe Mistral AQLM-2 | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

python -u experiments/signal_geometry/probe_accuracy.py \
    --model mistral7b-v02 --quantizer aqlm2 \
    --n_calib 200 --n_folds 5 --probe_type linear

echo ""
echo "=== Linear probe Mistral AQLM-2 complete | $(date) ==="
