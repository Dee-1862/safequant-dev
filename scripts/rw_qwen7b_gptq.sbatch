#!/bin/bash
#SBATCH --job-name=rw_q7_gptq
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --output=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_qwen7b_gptq_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant/scripts/logs/rw_qwen7b_gptq_%j.err

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

for QTZ in gptq4 gptq3 gptq2; do
    echo ""
    echo "======================================================================"
    echo "  Starting: qwen7b / ${QTZ} | $(date)"
    echo "======================================================================"
    python diagnostics/readwrite_restoration.py \
        --model qwen7b \
        --quantizer ${QTZ} \
        --n_eval 100 \
        --n_reasoning 50
    echo "  ✓ qwen7b/${QTZ} done | $(date)"
done

echo ""
echo "All Qwen7b GPTQ runs complete: $(date)"
