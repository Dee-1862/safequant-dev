#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F2a — Random-direction ablation distribution, Llama-3.1-8B / AWQ-4.
# K=20 random unit vectors, n_eval=100. Bounds the null-distribution that
# the descriptive self/cross asymmetry observation must clear.
#
# Output: outputs/random_ablation_dist_llama7b_awq.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_rad_llama_awq
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rad_llama_awq_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rad_llama_awq_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F2a random ablation Llama/AWQ-4 | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

python -u diagnostics/random_ablation_distribution.py \
    --model llama7b --quantizer awq \
    --n_eval 100 --n_random 20

echo "=== F2a random ablation Llama/AWQ-4 complete | $(date) ==="
