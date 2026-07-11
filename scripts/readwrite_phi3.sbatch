#!/bin/bash
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/readwrite_phi3_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/readwrite_phi3_%j.err
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd /jet/home/bhoomire/projects/safequant_v2
echo "=== read/write restoration | phi3/aqlm2 | $(date) ==="
python -u experiments/regime_sweep/restoration_sweep.py --model phi3 --quantizer aqlm2 --n_eval 200 --n_reasoning 50
