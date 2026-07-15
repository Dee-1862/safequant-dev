#!/bin/bash
#SBATCH --job-name=sq_lrqat
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lrqat_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lrqat_llama_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# Tables 6–7 — matched-budget read/write INT-2 and INT-4.


echo "=== LR-QAT repair Llama/AQLM-2 | $(date) ==="
BUDGET_ARGS="--steps 3000 --lr 1e-4 --rank 32 --alpha 64 \
             --seq_len 512 --grad_accum 8 --n_train_docs 4000 \
             --group_size 128 --n_eval 200 --n_reasoning 50 --n_xstest 50"

python -u -m experiments.trained_repair.lrqat \
  --model llama7b --quantizer aqlm2 --role read --n_bits 2 ${BUDGET_ARGS}

python -u -m experiments.trained_repair.lrqat \
  --model llama7b --quantizer aqlm2 --role write --n_bits 2 ${BUDGET_ARGS}

python -u -m experiments.trained_repair.lrqat \
  --model llama7b --quantizer aqlm2 --role read --n_bits 4 ${BUDGET_ARGS}

python -u -m experiments.trained_repair.lrqat \
  --model llama7b --quantizer aqlm2 --role write --n_bits 4 ${BUDGET_ARGS}

echo "=== done | $(date) ==="
