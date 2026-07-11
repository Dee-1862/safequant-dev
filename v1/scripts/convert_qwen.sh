#!/bin/bash
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
cd /jet/home/bhoomire/projects/AQLM
export HF_HOME=/ocean/projects/cis240116p/bhoomire/hf_cache

python convert_to_hf.py \
  Qwen/Qwen2.5-7B-Instruct \
  /ocean/projects/cis240116p/bhoomire/aqlm_out/qwen25_7b_pv \
  /ocean/projects/cis240116p/bhoomire/aqlm_out/Qwen2.5-7B-Instruct-AQLM-PV-2Bit-1x16-hf \
  --save_safetensors
