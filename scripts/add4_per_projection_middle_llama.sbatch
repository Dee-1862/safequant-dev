#!/bin/bash
# Addition 4 (Llama) — per-projection × middle-layer restoration.
#
# Scales the original Llama-AQLM-2 run to the remaining 4 quantizers on the
# same model (awq, gptq4, gptq3, gptq2). aqlm2 is already done — its result
# lives at outputs/per_projection_middle_llama7b_aqlm2_L11-22.json.
#
# Layer range [11, 22) matches the per-layer-range decomposition that
# identified middle layers as carrying 72% of the recoverable damage on
# Llama-AQLM-2. Same range used here so cells are comparable.
#
# Note: AWQ per-projection restoration is untested in this pipeline. F3a
# llama/mistral sbatches excluded AWQ for LayerNorm-fold reasons, but
# readwrite_restoration._transform_fp32_to_awq_space() handles the AWQ
# scale-space transform. Verify the awq cell looks sane before trusting
# downstream comparisons.
#SBATCH --job-name=sq2_add4_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=16:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add4_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add4_llama_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Addition 4 per-projection × middle (Llama-3.1-8B) | $(date) ==="
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
for QUANTIZER in awq gptq4 gptq3 gptq2; do
    robust_run "Add4 — per_projection_middle on ${MODEL}/${QUANTIZER}" \
        python -u experiments/localization/per_projection.py \
            --model ${MODEL} --quantizer ${QUANTIZER} \
            --layer_lo 11 --layer_hi 22 \
            --n_eval 200 --n_reasoning 50
done

echo ""
echo "=== Addition 4 Llama complete | $(date) ==="
