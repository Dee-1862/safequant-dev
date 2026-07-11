#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Linear refusal probes — Llama-3.1-8B at GPTQ-4, GPTQ-3, GPTQ-2.
#
# Extends the pilot (Llama-AQLM-2, "preserved" interpretation) to test if
# detection preservation generalizes across quantizers within Llama.
# Hypothesis: detection should remain >95% acc at every layer for fast
# quantizers (gptq4). For destroyed-regime (gptq3/gptq2), probes may
# finally collapse — that's a real test of where the "signal is preserved"
# claim breaks down.
#
# Per condition: ~25 min fast / ~50 min slow. Total: ~2 h.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_probe_llama_lin
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_llama_lin_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_llama_lin_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Linear probes Llama (gptq4, gptq3, gptq2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "########################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

for QUANTIZER in gptq4 gptq3 gptq2; do
    robust_run "Linear probe llama7b/${QUANTIZER}" \
        python -u experiments/signal_geometry/probe_accuracy.py \
            --model llama7b --quantizer ${QUANTIZER} \
            --n_calib 200 --n_folds 5 --probe_type linear
done

echo ""
echo "=== Linear probes Llama complete | $(date) ==="
