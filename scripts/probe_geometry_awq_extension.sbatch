#!/bin/bash
# Probe-geometry — fill the two missing AWQ-4 cells:
#   • Llama-AWQ-4 (cross-dominant F2 condition, asymΔ=-17.5pp; key data point)
#   • Mistral-AWQ-4 (PRESERVED null-asymmetry condition)
# Qwen-AWQ-4 already in outputs/. After this completes, the geometry matrix
# is full: 14/14 conditions across 3 architectures × all relevant quantizers.
#SBATCH --job-name=sq2_geo_awq_ext
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_awq_ext_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_awq_ext_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Probe-geometry AWQ extension (Llama + Mistral) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "###################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "###################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

robust_run "Probe-geometry llama7b/awq" \
    python -u experiments/signal_geometry/probe_rotation.py --model llama7b --quantizer awq --n_calib 200

robust_run "Probe-geometry mistral7b-v02/awq" \
    python -u experiments/signal_geometry/probe_rotation.py --model mistral7b-v02 --quantizer awq --n_calib 200

echo "=== Probe-geometry AWQ extension complete | $(date) ==="
