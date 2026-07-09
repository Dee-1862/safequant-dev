#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 1 — Phase transition in refusal-direction cos-sim, Mistral-7B-v0.2.
#
# Cross-architecture replication of F1 phase-transition curve. Same 5
# quantizers as Llama: GPTQ-4, AWQ-4, AQLM-2, GPTQ-3, GPTQ-2.
#
# Output: outputs/direction_drift_mistral7b-v02_{gptq4,awq,aqlm2,gptq3,gptq2}.json
#
# n_calib = 200 harmful + 200 harmless
# Per-condition: ~30 min fast / ~60 min slow → ~3.5 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f1_mistral
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_mistral_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_mistral_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F1 Mistral — Phase Transition | $(date) ==="
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

MODEL=mistral7b-v02
for QUANTIZER in gptq4 awq aqlm2 gptq3 gptq2; do
    robust_run "F1 — direction_drift on ${MODEL}/${QUANTIZER}" \
        python -u diagnostics/direction_drift.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_calib 200 \
            --output outputs/direction_drift_${MODEL}_${QUANTIZER}.json
done

echo ""
echo "=== F1 Mistral complete | $(date) ==="
