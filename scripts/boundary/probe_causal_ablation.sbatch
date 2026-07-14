#!/bin/bash
#SBATCH --job-name=sq_ablate
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_ablate_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/probe_ablate_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Appendix — ablate best probe direction, measure ASR.


echo "=== probe causal ablation | $(date) ==="
python -u -m experiments.boundary.probe_causal_ablation \
  --model llama7b --quantizer aqlm2 \
  --n_calib 200 --n_eval 200
echo "=== done | $(date) ==="
