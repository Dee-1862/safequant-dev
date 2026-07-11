#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Finding 2 — Directional ablation, Llama-3.1-8B SLOW quantizers.
# Quantizers: GPTQ-3, GPTQ-2 (Triton kernel, ~30 s/prompt)
#
# Destroyed-regime quantizers — the cross-vs-self asymmetry should collapse
# here. Reduced n_eval to n=200 to stay tractable on Triton kernels.
#
# Output: outputs/directional_ablation_llama7b_{gptq3,gptq2}.json
#
# n_calib=200, n_eval=200, n_arc=50
# Per-quantizer: ~7.5 h → ~15 h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_f2_llama_slow
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f2_llama_slow_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f2_llama_slow_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F2 Llama SLOW — Directional Ablation | $(date) ==="
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
    robust_run "F2 — directional_ablation on ${MODEL}/${QUANTIZER} (slow Triton kernel)" \
        python -u diagnostics/directional_ablation.py \
            --model ${MODEL} \
            --quantizer ${QUANTIZER} \
            --n_calib 200 \
            --n_eval 200 \
            --n_arc 50
done

echo ""
echo "=== F2 Llama SLOW complete | $(date) ==="
