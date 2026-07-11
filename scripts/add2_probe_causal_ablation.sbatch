#!/bin/bash
# Addition 2: ablate the probe direction at each layer in quant model; measure ASR.
# Tests whether probe direction IS the operative refusal feature.
#SBATCH --job-name=sq2_add2_causal
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add2_causal_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add2_causal_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Addition 2 probe causal ablation (Llama-AQLM-2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

# Ablate every 4th layer (0,4,8,12,16,20,24,28) — 8 conditions × 100 prompts
python -u experiments/boundary/probe_causal_ablation.py \
    --model llama7b --quantizer aqlm2 \
    --n_calib 200 --n_eval 100 \
    --layers_to_ablate 0,4,8,12,16,20,24,28

echo "=== Addition 2 complete | $(date) ==="
