#!/bin/bash
#SBATCH --job-name=sq_probe_acc
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_acc_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_acc_llama_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"


echo "=== probe accuracy Llama | $(date) ==="
python -u -m experiments.signal_geometry.probe_accuracy \
  --model llama7b --quantizer aqlm2 --n_calib 200 --probe_type linear
echo "=== done | $(date) ==="
