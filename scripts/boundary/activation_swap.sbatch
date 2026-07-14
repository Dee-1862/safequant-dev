#!/bin/bash
#SBATCH --job-name=sq_actswap
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/act_swap_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/act_swap_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Appendix — swap quantized activations into FP32 at select layers.


echo "=== activation swap | $(date) ==="
python -u -m experiments.boundary.activation_swap \
  --model llama7b --quantizer aqlm2 \
  --swap_layers 4,11,16,22,29 --n_eval 200 --max_new_tokens 256
echo "=== done | $(date) ==="
