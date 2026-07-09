#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Nonlinear (MLP) refusal probes — Mistral-7B-v0.2 at AQLM-2, GPTQ-4, GPTQ-3, GPTQ-2.
#
# Cross-architecture nonlinear-probe extension. Tests whether the
# detection-preserved claim survives nonlinear probing on Mistral as well.
#
# Per condition: ~30 min fast / ~60 min slow. Total: ~3 h.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_probe_mistral_mlp
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_mistral_mlp_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_mistral_mlp_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | MLP probes Mistral (aqlm2, gptq4, gptq3, gptq2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "########################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

for QUANTIZER in aqlm2 gptq4 gptq3 gptq2; do
    robust_run "MLP probe mistral7b-v02/${QUANTIZER}" \
        python -u diagnostics/refusal_probe.py \
            --model mistral7b-v02 --quantizer ${QUANTIZER} \
            --n_calib 200 --n_folds 5 --probe_type mlp
done

echo ""
echo "=== MLP probes Mistral complete | $(date) ==="
