#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 3a SLOW — Static read/write restoration, Mistral-7B-v0.2, GPTQ-3 + GPTQ-2.
# Quantizers: GPTQ-3, GPTQ-2 (Triton kernel)
#
# Per-decision override: includes destroyed-regime quantizers. Note F1 showed
# Mistral-GPTQ-3 is DRIFTED (cos_median 0.90) rather than DESTROYED, so it
# may have recoverable damage worth measuring.
#
# Output: outputs/readwrite_mistral7b-v02_{gptq3,gptq2}.json
# Expected runtime: ~18 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f3a_mistral_slow
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_mistral_slow_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_mistral_slow_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a Mistral SLOW — Static Restoration | $(date) ==="
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
for QUANTIZER in gptq3 gptq2; do
    robust_run "F3a — readwrite_restoration on ${MODEL}/${QUANTIZER} (slow Triton kernel)" \
        python -u experiments/regime_sweep/restoration_sweep.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_eval 200 \
            --n_reasoning 50
done

echo ""
echo "=== F3a Mistral SLOW complete | $(date) ==="
