#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -o headline_full_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2

# Tier 1: headline-cell ASR(>=400)+ARC+XSTest over the 5 restoration conditions,
# with McNemar(paired flips) + bootstrap CIs on recovery.
python -u -m diagnostics.headline_eval_full \
  --model llama7b --quantizer aqlm2 \
  --n_eval 400 --n_reasoning 100 --n_boot 2000 \
  --output outputs/headline_full_llama7b_aqlm2.json
