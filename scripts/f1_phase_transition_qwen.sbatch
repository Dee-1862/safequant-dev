#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 1 — Phase transition cos-sim, Qwen2.5-7B-Instruct.
#
# Third-architecture replication (Llama, Mistral, now Qwen). Excludes AQLM-2
# until the quantize sbatch produces a checkpoint; the other 4 quantizers
# are already cached.
#
# Quantizers: GPTQ-4, AWQ-4, GPTQ-3, GPTQ-2
# n_calib = 200 harmful + 200 harmless
#
# Output: outputs/direction_drift_qwen7b_{gptq4,awq,gptq3,gptq2}.json
# Per-condition: ~30 min fast / ~60 min slow → ~3 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f1_qwen
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_qwen_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_qwen_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F1 Qwen — Phase Transition | $(date) ==="
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
for QUANTIZER in gptq4 awq gptq3 gptq2; do
    robust_run "F1 — direction_drift on ${MODEL}/${QUANTIZER}" \
        python -u experiments/signal_geometry/direction_drift.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_calib 200 \
            --output outputs/direction_drift_${MODEL}_${QUANTIZER}.json
done

echo ""
echo "=== F1 Qwen complete | $(date) ==="
