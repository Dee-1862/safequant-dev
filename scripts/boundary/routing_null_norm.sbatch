#!/bin/bash
#SBATCH --job-name=sq_route
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/route_norm_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/route_norm_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Appendix — de-confounded routing null with cosine projection.


echo "=== routing null (normalized) | $(date) ==="
python -u -m experiments.boundary.routing_null \
  --model llama7b --quantizer aqlm2 --n_calib 200 --normalize
echo "=== done | $(date) ==="
