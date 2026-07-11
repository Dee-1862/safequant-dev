#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared              # H100 nodes, partial allocation allowed
#SBATCH --gpus=h100-80:2            # match your cluster's gres string; 2 GPUs ~halves time
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -o calib_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
cd /jet/home/bhoomire/projects/AQLM
export HF_HOME=/ocean/projects/cis240116p/bhoomire/hf_cache
export CUDA_VISIBLE_DEVICES=0,1

export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export DATASET_PATH=c4          # calibration tokenizes c4 with the Qwen tokenizer on the fly
export SAVE_PATH=/ocean/projects/cis240116p/bhoomire/aqlm_out/qwen25_7b_1x16g8

python main.py $MODEL_PATH $DATASET_PATH \
  --nsamples=1024 --val_size=128 \
  --num_codebooks=1 --nbits_per_codebook=16 --in_group_size=8 \
  --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=10 --finetune_early_stop=3 \
  --finetune_keep_best --local_batch_size=1 \
  --offload_activations \
  --attn_implementation eager \
  --resume \
  --save $SAVE_PATH
