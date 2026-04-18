#!/bin/bash
#SBATCH --job-name=rw_int8
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --output=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_int8_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_int8_%j.err

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

# Llama-3.1-8B INT8
echo ""
echo "======================================================================"
echo "  Starting: llama7b / int8 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model llama7b \
    --quantizer int8 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ llama7b/int8 done | $(date)"

# Mistral-7B-v0.2 INT8
echo ""
echo "======================================================================"
echo "  Starting: mistral7b-v02 / int8 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model mistral7b-v02 \
    --quantizer int8 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ mistral7b-v02/int8 done | $(date)"

# Qwen2.5-7B INT8
echo ""
echo "======================================================================"
echo "  Starting: qwen7b / int8 | $(date)"
echo "======================================================================"
python diagnostics/readwrite_restoration.py \
    --model qwen7b \
    --quantizer int8 \
    --n_eval 100 \
    --n_reasoning 50
echo "  ✓ qwen7b/int8 done | $(date)"

echo ""
echo "All INT8 runs complete: $(date)"
