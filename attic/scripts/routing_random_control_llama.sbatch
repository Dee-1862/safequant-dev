#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Random-direction routing control, Llama-3.1-8B.
# For each (model, quantizer), samples K random unit vectors and computes
# the per-prompt operative-layer distribution. Validates whether the
# late-layer peak we see for r̂_FP / r̂_q is direction-specific or merely
# a property of activation magnitudes at late layers.
#
# Output: outputs/routing_random_control_llama7b_{gptq4,awq,gptq3,gptq2,aqlm2}.json
# Per condition ≈ same time as layer_routing (residual extraction dominates).
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_rrc_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rrc_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rrc_llama_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Random-direction routing control Llama | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "########################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

MODEL=llama7b
for QUANTIZER in gptq4 awq gptq3 gptq2 aqlm2; do
    robust_run "Random-direction routing control on ${MODEL}/${QUANTIZER}" \
        python -u diagnostics/routing_random_control.py \
            --model ${MODEL} --quantizer ${QUANTIZER} \
            --n_calib 200 --n_random 20
done

echo ""
echo "=== Random-direction routing control Llama complete | $(date) ==="
