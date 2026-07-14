#!/bin/bash
#SBATCH --job-name=sq_f1_drift
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_drift_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/f1_drift_llama_%j.err
source "$(cd "$(dirname "$0")/.." && pwd)/_env.sh"

# Table 1 — refusal-direction cos-sim drift across quantizers (RESULTS.md).


echo "=== direction drift Llama multi-quant | $(date) ==="
MODEL=llama7b
for QUANTIZER in gptq4 awq aqlm2 gptq3 gptq2; do
  echo "---- ${MODEL} / ${QUANTIZER} ----"
  python -u -m experiments.signal_geometry.direction_drift \
    --model ${MODEL} --quantizer ${QUANTIZER} --n_calib 200 || echo "FAILED ${QUANTIZER}"
done
echo "=== done | $(date) ==="
