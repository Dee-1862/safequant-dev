#!/bin/bash
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
cd /jet/home/bhoomire/projects/AQLM

python lmeval.py --model hf \
  --model_args pretrained=/ocean/projects/cis240116p/bhoomire/aqlm_out/Qwen2.5-7B-Instruct-AQLM-PV-2Bit-1x16-hf,dtype=float16,trust_remote_code=True \
  --tasks arc_challenge --batch_size 8
