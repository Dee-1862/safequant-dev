#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F3a Decomposition Option 3 — Per-layer-range restoration on Llama-AQLM-2.
#
# Runs 5 conditions in one job (FP32 + quant + early/middle/late):
#   • early   = layers [0, 11)   — F2 says op_q=4 lives here
#   • middle  = layers [11, 22)
#   • late    = layers [22, 32)  — F2 says op_FP=29 lives here
#
# Tests F2 routing interpretation: restoring layers where the operative
# direction lives should recover more damage. If early dominates →
# AQLM-2's safety mechanism truly migrated to early layers (matches LR
# finding op_q=4). If late dominates → r̂_FP at layer 29 was the active
# safety direction the quant model lost.
#
# Per condition: ~1h. Total: ~5h. Walltime 8h with margin.
#
# Output: outputs/readwrite_per_layer_range_llama7b_aqlm2.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f3a_perlayer
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_perlayer_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_perlayer_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a per-layer-range decomposition (Llama AQLM-2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

# Llama-3.1-8B has 32 layers. Splits: early=[0,11), middle=[11,22), late=[22,32).
python -u diagnostics/readwrite_per_layer_range.py \
    --model llama7b \
    --quantizer aqlm2 \
    --n_eval 200 \
    --n_reasoning 50 \
    --early_end 11 \
    --mid_end 22

echo ""
echo "=== F3a per-layer-range decomposition complete | $(date) ==="
