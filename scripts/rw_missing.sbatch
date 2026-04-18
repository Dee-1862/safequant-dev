#!/bin/bash
#SBATCH --job-name=rw_missing
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --output=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_missing_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_missing_%j.err

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

# 1. Mistral-7B-v0.2 NF4
echo ""
echo "======================================================================"
echo "  Starting: mistral7b-v02 / nf4 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model mistral7b-v02 \
    --quantizer nf4 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ mistral7b-v02/nf4 done | $(date)"

# 2. Qwen2.5-7B NF4
echo ""
echo "======================================================================"
echo "  Starting: qwen7b / nf4 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model qwen7b \
    --quantizer nf4 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ qwen7b/nf4 done | $(date)"

# 3. Qwen2.5-7B GPTQ-4
echo ""
echo "======================================================================"
echo "  Starting: qwen7b / gptq4 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model qwen7b \
    --quantizer gptq4 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ qwen7b/gptq4 done | $(date)"

echo ""
echo "All missing runs complete: $(date)"
