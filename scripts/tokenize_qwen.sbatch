#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o tokenize_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
cd /jet/home/bhoomire/projects/AQLM
export HF_HOME=/ocean/projects/cis240116p/bhoomire/hf_cache

TARGET_MODEL=Qwen/Qwen2.5-7B-Instruct
SEQLEN=4096
DATASET=togethercomputer/RedPajama-Data-1T-Sample
OUTPUT_PATH=/ocean/projects/cis240116p/bhoomire/aqlm_data/redpajama_qwen25

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 finetune.py \
  --base_model $TARGET_MODEL --quantized_model ./doesnt_matter --dtype bfloat16 \
  --block_type Qwen2DecoderLayer \
  --dataset_name=$DATASET --split train --dataset_config_name plain_text \
  --cache_dir=/ocean/projects/cis240116p/bhoomire/aqlm_data/cache_dir \
  --trust_remote_code --model_seqlen=$SEQLEN \
  --preprocessing_num_workers=64 --preprocessing_chunk_length 100000 \
  --save_dataset_and_exit $OUTPUT_PATH
