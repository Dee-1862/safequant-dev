#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o snip_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2

# Contrastive SNIP: saliency of CE(refuse|x) - CE(comply|x), isolating the
# refuse-vs-comply decision weights and cancelling generic-text / output-magnitude
# components that may inflate the write side under plain refusal-CE.
python -u -m diagnostics.snip_safety_localization \
  --model llama7b --n_prompts 260 --mode contrastive \
  --taus 0.001 0.01 0.1 \
  --output outputs/snip_llama7b_contrastive.json
