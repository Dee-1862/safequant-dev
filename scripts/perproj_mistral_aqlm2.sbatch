#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F3a per-matrix decomposition — Mistral-7B-v0.2 / AQLM-2.
#
# Calls readwrite_per_projection.py (NOT per_projection_middle.py): restores
# ONE projection type at a time (q, k, v, o, gate, up, down) across ALL
# layers, not the middle-layer subset. This is the "per-matrix" axis
# orthogonal to the read-vs-write group split done by readwrite_restoration.
#
# Three-criteria framework predicts this cell is N/A (F3a damage −7.5pp —
# quantized is slightly safer than FP32; no damage to recover). Per-matrix
# restoration will show approximately zero recoveries across all 7 projections,
# serving as the "no-damage control" for F3a section.
#
# Mistral-AQLM-2 uses the fast AQLM kernel (~5-7s/sample), so n_eval=200
# is comfortable: 200 × 9 × 6s ≈ 3h. 6h walltime gives buffer.
#
# Output: outputs/readwrite_per_projection_mistral7b-v02_aqlm2.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_perproj_misaqlm2
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_misaqlm2_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_misaqlm2_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a per-matrix Mistral/AQLM-2 | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

python -u diagnostics/readwrite_per_projection.py \
    --model mistral7b-v02 --quantizer aqlm2 \
    --n_eval 200 --n_reasoning 50

echo "=== F3a per-matrix Mistral/AQLM-2 complete | $(date) ==="
