#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o pv_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
cd /jet/home/bhoomire/projects/AQLM
export HF_HOME=/ocean/projects/cis240116p/bhoomire/hf_cache

export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export QUANTIZED_WEIGHTS_PATH=/ocean/projects/cis240116p/bhoomire/aqlm_out/qwen25_7b_1x16g8
export TOKENIZED_DATASET_PATH=/ocean/projects/cis240116p/bhoomire/aqlm_data/redpajama_qwen25
export SAVE_PATH=/ocean/projects/cis240116p/bhoomire/aqlm_out/qwen25_7b_pv
export SEQLEN=4096

torchrun --nproc-per-node=4 finetune.py \
  --base_model $MODEL_PATH \
  --quantized_model $QUANTIZED_WEIGHTS_PATH \
  --model_seqlen=$SEQLEN \
  --block_type Qwen2DecoderLayer \
  --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype uint16 \
  --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 42 \
  --preprocessing_chunk_length 100000 \
  --cache_dir=/ocean/projects/cis240116p/bhoomire/aqlm_data/cache_dir \
  --trust_remote_code \
  --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
  --lamb --debias --lr 3e-4 --adam_beta1 0.90 --adam_beta2 0.95 \
  --max_code_change_per_step 1e-2 --code_lr 1e-2 --code_beta1 0.0 --code_beta2 0.95 \
  --beam_size 5 --delta_decay 0 \
  --batch_size=128 --microbatch_size=1 --max_epochs 1 \
  --gradient_checkpointing --print_every_steps=1 \
  --eval_every_steps=10 --keep_best_model \
  --save $SAVE_PATH --save_every_steps 100 \
  --attn_implementation eager
