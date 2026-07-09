#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --mem=128G
#SBATCH -o /jet/home/bhoomire/projects/safequant_v2/scripts/logs/refit_phi3_%j.log
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2
echo "=== probe refit noise-floor phi3 | $(date) ==="
python -u -m diagnostics.probe_refit_control --model phi3 --n_calib 200   --output outputs/probe_refit_control_phi3.json
