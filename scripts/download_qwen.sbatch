#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU
#SBATCH --gpus=h100-80:8
#SBATCH -N 1
#SBATCH -t 04:00:00                 # 4 hours walltime
#SBATCH -o download_model_%j.log

export HF_HOME=/ocean/projects/cis240116p/bhoomire/hf_cache

# Enable faster downloads via huggingface hub if installed
export HF_HUB_ENABLE_HF_TRANSFER=1

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
m='Qwen/Qwen2.5-7B-Instruct'
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m, dtype='auto')
print('cached', m)
"
