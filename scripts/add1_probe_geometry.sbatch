#!/bin/bash
# Addition 1: probe-geometry comparison (cos_sim of probe weights FP32 vs quant).
# Tests whether signal is preserved IN-PLACE or MOVED.
#SBATCH --job-name=sq2_add1_geo
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add1_geo_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add1_geo_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Addition 1 probe geometry (Llama-AQLM-2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

python -u experiments/signal_geometry/probe_rotation.py --model llama7b --quantizer aqlm2 --n_calib 200

echo "=== Addition 1 complete | $(date) ==="
