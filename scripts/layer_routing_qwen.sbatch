#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Layer-routing analysis, Qwen 2.5 7B.
# Quantizers: GPTQ-4, AWQ-4 only (no AQLM-2 checkpoint for Qwen 2.5)
#
# Output: outputs/layer_routing_qwen7b_{gptq4,awq}.json
# Per condition ~30 min → ~1.5 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_lr_qwen
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lr_qwen_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lr_qwen_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Layer routing Qwen | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "########################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

MODEL=qwen7b
for QUANTIZER in gptq4 awq; do
    robust_run "Layer routing on ${MODEL}/${QUANTIZER}" \
        python -u experiments/boundary/routing_null.py \
            --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200
done

echo ""
echo "=== Layer routing Qwen complete | $(date) ==="
