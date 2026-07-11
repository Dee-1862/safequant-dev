#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 1 — Phase transition in refusal-direction cos-sim, Llama-3.1-8B.
#
# Runs experiments/signal_geometry/direction_drift.py on Llama-3.1-8B-Instruct at 5 quantizers
# (GPTQ-4, AWQ-4, AQLM-2, GPTQ-3, GPTQ-2). Each produces per-layer cos-sim
# against the FP32 reference and reports the headline regime number.
#
# Output: outputs/direction_drift_llama7b_{gptq4,awq,aqlm2,gptq3,gptq2}.json
#
# n_calib = 200 harmful + 200 harmless (uniform per finalized plan)
# Per-condition: ~30 min fast / ~60 min slow → ~3.5 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f1_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_llama_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F1 Llama — Phase Transition | $(date) ==="
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
for QUANTIZER in gptq4 awq aqlm2 gptq3 gptq2; do
    robust_run "F1 — direction_drift on ${MODEL}/${QUANTIZER}" \
        python -u experiments/signal_geometry/direction_drift.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_calib 200 \
            --output outputs/direction_drift_${MODEL}_${QUANTIZER}.json
done

echo ""
echo "=== F1 Llama complete | $(date) ==="
