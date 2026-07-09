#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Layer-routing analysis SLOW, Llama-3.1-8B.
# Quantizers: GPTQ-3, GPTQ-2 (Triton kernel)
#
# Llama GPTQ-3 and GPTQ-2 are in DESTROYED regime per F1 (cos ≈ 0). Routing
# analysis here will likely look uniform / noisy, which is itself informative
# as a control: it confirms that the routing-structure signal seen in fast
# conditions disappears once the refusal direction is destroyed.
#
# Output: outputs/layer_routing_llama7b_{gptq3,gptq2}.json
# Per condition: ~1 h fwd + ~30min load = ~1.5h → ~3h total
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_lr_llama_slow
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lr_llama_slow_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lr_llama_slow_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Layer routing Llama SLOW | $(date) ==="
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
for QUANTIZER in gptq3 gptq2; do
    robust_run "Layer routing on ${MODEL}/${QUANTIZER} (slow Triton)" \
        python -u diagnostics/layer_routing.py \
            --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200
done

echo ""
echo "=== Layer routing Llama SLOW complete | $(date) ==="
