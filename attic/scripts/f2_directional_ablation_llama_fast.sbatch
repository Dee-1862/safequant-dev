#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 2 — Directional ablation, Llama-3.1-8B FAST quantizers.
# Quantizers: GPTQ-4, AWQ-4, AQLM-2 (fast inference kernels)
#
# Each invocation produces 4 conditions per quantizer:
#   baseline, self-ablation, cross-ablation, random
#
# Output: outputs/directional_ablation_llama7b_{gptq4,awq,aqlm2}.json
#
# n_calib=200 (direction estimation), n_eval=200 (ASR), n_arc=50 (utility check)
# Per-quantizer: ~2 h → ~6 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f2_llama_fast
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f2_llama_fast_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f2_llama_fast_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F2 Llama FAST — Directional Ablation | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""
    echo "########################################################################"
    echo "# $desc"
    echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

MODEL=llama7b
for QUANTIZER in gptq4 awq aqlm2; do
    robust_run "F2 — directional_ablation on ${MODEL}/${QUANTIZER}" \
        python -u diagnostics/directional_ablation.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_calib 200 \
            --n_eval 200 \
            --n_arc 50
done

echo ""
echo "=== F2 Llama FAST complete | $(date) ==="
