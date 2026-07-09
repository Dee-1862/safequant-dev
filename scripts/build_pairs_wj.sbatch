#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o pairs_wj_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2

# Displacement pairs on the ADVERSARIAL distribution (WildJailbreak eval) -- the
# distribution the 41->62.5 headline ASR is measured on, where FP refuses but the
# AQLM-2 cell complies. max_new_tokens=128 keeps 1000x2 batch-1 generations tractable;
# 128 tokens is ample to judge compliance and to supply the yl loser text.
python -u -m diagnostics.build_displacement_pairs \
  --model llama7b --quantizer aqlm2 \
  --prompt_source wildjailbreak --wj_config eval \
  --n_prompts 1000 --max_new_tokens 128 \
  --output outputs/displacement_pairs_llama7b_wj.json
