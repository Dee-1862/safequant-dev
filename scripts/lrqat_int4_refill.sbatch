#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --mem=128G
#SBATCH -o /jet/home/bhoomire/projects/safequant_v2/scripts/logs/lrqat_int4_refill_%j.log
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
cd /jet/home/bhoomire/projects/safequant_v2
echo "=== LR-QAT INT-4 REFILL (read, n_eval=400, n_xstest=250) | $(date) ==="
python -u diagnostics/lrqat_repair.py \
    --model llama7b --quantizer aqlm2 --role read --n_bits 4 \
    --steps 3000 --lr 1e-4 --rank 32 --alpha 64 \
    --seq_len 512 --grad_accum 8 --n_train_docs 4000 \
    --group_size 128 --n_eval 400 --n_reasoning 50 --n_xstest 250 \
    --output outputs/lrqat_llama7b_aqlm2_read_int4_n400.json
