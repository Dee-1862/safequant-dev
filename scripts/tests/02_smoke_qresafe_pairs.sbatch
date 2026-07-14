#!/bin/bash
#SBATCH --job-name=sq_smoke_pairs
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/smoke_qresafe_pairs_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/smoke_qresafe_pairs_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Optional: tiny Q-Resafe pairs path (generate + HarmBench + save). Skip until 01 is green.

set -euo pipefail

MODEL="${SMOKE_MODEL:-llama7b}"
QUANT="${SMOKE_QUANT:-aqlm2}"
OUT="outputs/trained_repair/qresafe_pairs_${MODEL}_${QUANT}_smoke.json"

echo "=== smoke qresafe_pairs ${MODEL}/${QUANT} | $(date) ==="
python -u -m experiments.trained_repair.qresafe_pairs \
  --model "${MODEL}" --quantizer "${QUANT}" \
  --prompt_source wildjailbreak --wj_config eval \
  --n_prompts 32 --max_new_tokens 64 \
  --output "${OUT}"

ls -lh "${OUT}"
echo "=== smoke qresafe_pairs OK | $(date) ==="
