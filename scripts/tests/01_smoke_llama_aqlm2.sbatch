#!/bin/bash
#SBATCH --job-name=sq_smoke_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/smoke_llama_aqlm2_%j.out
#SBATCH --error=scripts/logs/smoke_llama_aqlm2_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# One-cell GPU smoke: Llama-7B + AQLM2 with tiny n_*. Not for paper numbers.

set -euo pipefail

MODEL="${SMOKE_MODEL:-llama7b}"
QUANT="${SMOKE_QUANT:-aqlm2}"

echo "=== smoke ${MODEL}/${QUANT} | $(date) ==="

echo "---- direction_drift ----"
python -u -m experiments.signal_geometry.direction_drift \
  --model "${MODEL}" --quantizer "${QUANT}" --n_calib 16

echo "---- probe_accuracy ----"
python -u -m experiments.signal_geometry.probe_accuracy \
  --model "${MODEL}" --quantizer "${QUANT}" --n_calib 32 --n_folds 2

echo "---- headline_eval ----"
python -u -m experiments.headline.headline_eval \
  --model "${MODEL}" --quantizer "${QUANT}" \
  --n_eval 32 --n_reasoning 16 --n_boot 100

echo "=== smoke outputs ==="
ls -lt outputs/signal_geometry/direction_drift_"${MODEL}"* 2>/dev/null | head -3 || true
ls -lt outputs/signal_geometry/probe_accuracy_"${MODEL}"* 2>/dev/null | head -3 || true
ls -lt outputs/headline/headline_eval_"${MODEL}"* 2>/dev/null | head -3 || true

echo "=== smoke ${MODEL}/${QUANT} OK | $(date) ==="
