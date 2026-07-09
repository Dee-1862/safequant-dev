#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=160G
#SBATCH -o /jet/home/bhoomire/projects/safequant_v2/scripts/logs/qresafe_write_r128_s42_%j.log
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2
echo "=== Q-Resafe write r=128 seed=42 | $(date) ==="
python -u -m diagnostics.qresafe_dpo --mask write --r 128 --alpha 256 --beta 0.01 --epochs 3 --lr 5e-6   --n_eval 400 --opt8bit --seed 42 --output outputs/qresafe_write_r128_s42_llama7b_aqlm2.json
