#!/bin/bash
#SBATCH --job-name=ap_aqlm2
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --output=/jet/home/bhoomire/projects/safequant/scripts/logs/activation_patch_aqlm2_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant/scripts/logs/activation_patch_aqlm2_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant

# Llama-3.1-8B AQLM-2bit (damage: +21pp — large signal)
echo ""
echo "======================================================================"
echo "  Starting: llama7b / aqlm2 | $(date)"
echo "======================================================================"
python diagnostics/activation_weighted_patch.py \
    --model llama7b \
    --quantizer aqlm2 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ llama7b/aqlm2 done | $(date)"

# Mistral-7B-v0.2 AQLM-2bit (damage: -10pp — moderate signal)
echo ""
echo "======================================================================"
echo "  Starting: mistral7b-v02 / aqlm2 | $(date)"
echo "======================================================================"
python diagnostics/activation_weighted_patch.py \
    --model mistral7b-v02 \
    --quantizer aqlm2 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ mistral7b-v02/aqlm2 done | $(date)"

echo ""
echo "All activation patch AQLM runs complete: $(date)"
