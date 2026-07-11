#!/bin/bash
#SBATCH --job-name=arditi_off
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/%j_arditi_off_unq.out

source /jet/home/bhoomire/projects/safequant_v2/scripts/setup_env.sh
cd /jet/home/bhoomire/projects/safequant_v2

echo "Running Official Arditi Selection (FP32)"
python diagnostics/run_arditi_official.py --model llama7b --quantizer fp32
