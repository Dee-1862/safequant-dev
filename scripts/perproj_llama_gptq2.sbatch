#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F3a per-matrix decomposition — Llama-3.1-8B / GPTQ-2.
#
# Calls readwrite_per_projection.py (NOT per_projection_middle.py): restores
# ONE projection type at a time (q, k, v, o, gate, up, down) across ALL
# layers, not the middle-layer subset. This is the "per-matrix" axis
# orthogonal to the read-vs-write group split done by readwrite_restoration.
#
# Three-criteria framework predicts this cell is degenerate (F3a damage
# −21.5pp + ARC collapse −12pp). Per-matrix restoration here is a "the
# methodology breaks in destroyed regime" control.
#
# RUNTIME: auto_gptq Triton kernel ≈ 46s/sample. At n_eval=200 × 9 conds ×
# 46s ≈ 23h — won't fit. Using n_eval=100 → ~11.5h, fits 14h walltime.
#
# Output: outputs/readwrite_per_projection_llama7b_gptq2.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_perproj_llgptq2
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=14:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_llgptq2_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_llgptq2_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a per-matrix Llama/GPTQ-2 | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

python -u experiments/localization/per_projection.py \
    --model llama7b --quantizer gptq2 \
    --n_eval 100 --n_reasoning 50

echo "=== F3a per-matrix Llama/GPTQ-2 complete | $(date) ==="
