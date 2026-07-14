#!/bin/bash
#SBATCH --job-name=sq_perproj_mid
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_mid_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/perproj_mid_llama_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Middle-layer-only projection ablation (scripts_old/add4).


echo "=== per-projection middle band Llama | $(date) ==="
python -u -m experiments.localization.per_projection \
  --model llama7b --quantizer aqlm2 --n_eval 200 --n_reasoning 50 \
  --layer_lo 11 --layer_hi 22
echo "=== done | $(date) ==="
