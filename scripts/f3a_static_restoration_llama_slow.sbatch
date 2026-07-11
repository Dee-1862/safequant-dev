#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 3a SLOW — Static read/write restoration, Llama-3.1-8B, GPTQ-3 + GPTQ-2.
# Quantizers: GPTQ-3, GPTQ-2 (Triton kernel, ~30 s/prompt)
#
# Per-decision override: includes destroyed-regime quantizers despite F1+F2
# already establishing null. Reviewer-defensibility: shows explicit null
# rather than written disclosure.
#
# Each invocation produces 5 sub-conditions: baseline-FP32, quantized,
# +write restored, +read restored, +both restored. Each sub-condition runs
# ASR (n=200) + ARC (n=50).
#
# Expected runtime per slow quantizer: ~9 h
#   - 4 slow sub-conditions × (200 prompts × 30 sec + 50 ARC × ~30 sec)
#   - + FP32 baseline ~30 min
# Total: ~18 h
#
# Output: outputs/readwrite_llama7b_{gptq3,gptq2}.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f3a_llama_slow
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_llama_slow_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_llama_slow_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a Llama SLOW — Static Restoration | $(date) ==="
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
for QUANTIZER in gptq3 gptq2; do
    robust_run "F3a — readwrite_restoration on ${MODEL}/${QUANTIZER} (slow Triton kernel)" \
        python -u experiments/regime_sweep/restoration_sweep.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_eval 200 \
            --n_reasoning 50
done

echo ""
echo "=== F3a Llama SLOW complete | $(date) ==="
