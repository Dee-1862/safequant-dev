#!/bin/bash
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/phi3_diag_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/phi3_diag_%j.err
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
cd /jet/home/bhoomire/projects/safequant_v2
python -u -m diagnostics.phi3_restore_diag
