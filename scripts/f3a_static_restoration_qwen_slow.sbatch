#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 3a SLOW — Static read/write restoration, Qwen-2.5-7B-Instruct,
# GPTQ-3 + GPTQ-2 (Triton kernel, slow).
#
# Completes the Qwen F3a coverage for the destroyed/drifted regime.
# Probe-geometry context:
#   GPTQ-3 median cos 0.747  PARTIALLY-ROTATED
#   GPTQ-2 median cos 0.085  PRESERVED-BUT-MOVED (large rotation, low cos)
# These rows are expected to fail at least one of the three-criteria gates
# (degenerate recovery or ARC collapse) — running them documents the failure
# for the F3a matrix rather than leaving the row blank.
#
# Output: outputs/readwrite_qwen7b_{gptq3,gptq2}.json
# Expected runtime: ~18 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f3a_qwen_slow
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_qwen_slow_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f3a_qwen_slow_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F3a Qwen SLOW — Static Restoration | $(date) ==="
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
for QUANTIZER in gptq3 gptq2; do
    robust_run "F3a — readwrite_restoration on ${MODEL}/${QUANTIZER} (slow Triton kernel)" \
        python -u experiments/regime_sweep/restoration_sweep.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_eval 200 \
            --n_reasoning 50
done

echo ""
echo "=== F3a Qwen SLOW complete | $(date) ==="
