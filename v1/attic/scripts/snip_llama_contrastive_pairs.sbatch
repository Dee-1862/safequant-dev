#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o snip_pairs_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2

PAIRS=outputs/displacement_pairs_llama7b.json

# Phase 1: build genuine displacement pairs (FP refused & AQLM-2 complied).
python -u -m diagnostics.build_displacement_pairs \
  --model llama7b --quantizer aqlm2 --n_prompts 260 \
  --output $PAIRS || { echo "PHASE1_FAILED"; exit 1; }

# Phase 2: contrastive SNIP on the FP model over the real pairs.
python -u -m diagnostics.snip_safety_localization \
  --model llama7b --pairs_file $PAIRS \
  --taus 0.001 0.01 0.1 \
  --output outputs/snip_llama7b_contrastive_pairs.json
