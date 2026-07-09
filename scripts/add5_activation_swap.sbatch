#!/bin/bash
# Addition 5: activation-swap test — replace FP32 residual at layer L with
# the quant model's residual, then generate. Tests whether quant activation
# carries usable safety info (consumption-damage) or has lost it (activation
# damage).
#SBATCH --job-name=sq2_add5_swap
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add5_swap_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/add5_swap_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Addition 5 activation swap (Llama-AQLM-2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /jet/home/bhoomire/projects/safequant_v2

# Swap at 5 representative layers: F2's qL=4, middle 11/16/22, F2's fpL=29
python -u diagnostics/activation_swap.py \
    --model llama7b --quantizer aqlm2 \
    --swap_layers 4,11,16,22,29 \
    --n_eval 100

echo "=== Addition 5 complete | $(date) ==="
