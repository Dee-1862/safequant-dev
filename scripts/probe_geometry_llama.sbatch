#!/bin/bash
# Probe-geometry extension: Llama at GPTQ-4, GPTQ-3, GPTQ-2.
# AQLM-2 already done by add1. Tests whether the ~38° rotation observed
# on Llama-AQLM-2 is regime-specific (~0° for preserved, ~high for drifted,
# noise/0 for destroyed) or a generic property of all quantization.
#SBATCH --job-name=sq2_geo_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_llama_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Probe-geometry Llama (gptq4, gptq3, gptq2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "###################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "###################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

for Q in gptq4 gptq3 gptq2; do
    robust_run "Probe-geometry llama7b/${Q}" \
        python -u diagnostics/probe_geometry.py --model llama7b --quantizer ${Q} --n_calib 200
done

echo "=== Probe-geometry Llama complete | $(date) ==="
