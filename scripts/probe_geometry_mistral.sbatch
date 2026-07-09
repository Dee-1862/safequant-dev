#!/bin/bash
# Probe-geometry: Mistral × 4 quantizers (AQLM-2, GPTQ-4, GPTQ-3, GPTQ-2).
# Cross-architecture replication of the rotation finding.
#SBATCH --job-name=sq2_geo_mistral
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_mistral_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/geo_mistral_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Probe-geometry Mistral (aqlm2, gptq4, gptq3, gptq2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""; echo "###################################################################"
    echo "# $desc"; echo "# started: $(date)"
    echo "###################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

for Q in aqlm2 gptq4 gptq3 gptq2; do
    robust_run "Probe-geometry mistral7b-v02/${Q}" \
        python -u diagnostics/probe_geometry.py --model mistral7b-v02 --quantizer ${Q} --n_calib 200
done

echo "=== Probe-geometry Mistral complete | $(date) ==="
