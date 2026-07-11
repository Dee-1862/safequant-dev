#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 3a — Static read/write restoration, Qwen-2.5-7B-Instruct.
# Quantizers: GPTQ-4 (preserved regime).
#
# Cross-architecture replication of F3a read-side asymmetry on the third
# architecture in the matrix. Qwen has no AQLM-2 condition (the upstream
# AQLM-2 quantization run is disabled — see scripts/quantize_aqlm_qwen.sbatch),
# so the F3a Qwen coverage spans GPTQ-{4,3,2} only.
#
# AWQ is intentionally excluded from F3a to match the Llama/Mistral matrix
# (AWQ restoration requires scale-space transformation that is implemented in
# experiments/regime_sweep/restoration_sweep.py but kept out of the F3a table).
#
# Output: outputs/readwrite_qwen7b_gptq4.json
#
# n_eval=200, n_reasoning=50 → ~3 h
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f3a_qwen
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_qwen_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_qwen_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a Qwen — Static Restoration | $(date) ==="
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

MODEL=qwen7b
for QUANTIZER in gptq4; do
    robust_run "F3a — readwrite_restoration on ${MODEL}/${QUANTIZER}" \
        python -u experiments/regime_sweep/restoration_sweep.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_eval 200 \
            --n_reasoning 50
done

echo ""
echo "=== F3a Qwen complete | $(date) ==="
