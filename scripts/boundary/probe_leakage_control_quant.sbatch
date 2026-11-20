#!/bin/bash
#SBATCH --job-name=sq_x2_leakage_q
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x2_leakage_quant_%j.out
#SBATCH --error=scripts/logs/x2_leakage_quant_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X2 (R2 W5, R3 W3): surface-feature control on every coherent quantized cell.
# Same residual probe + XSTest + HarmBench transfer as the FP32 control, but the
# residuals come from the quantized model (--quantizer). The TF-IDF surface baseline
# is recomputed in every file so each JSON is self-contained.
#
# Submit:  bash scripts/submit.sh scripts/boundary/probe_leakage_control_quant.sbatch
# Output:  outputs/boundary/probe_leakage_control_<model>_<quant>.json  (new names;
#          a cell whose file already exists is refused and the loop moves on)
#
# Default CELLS = the 14 coherent Table 2 cells (quantized ARC >= 40 in
# restoration_sweep_*.json; excluded: the four GPTQ-2 cells and Mistral GPTQ-3).
# Override:  CELLS="llama7b:gptq4 phi3:aqlm2" bash scripts/submit.sh ...
# ~1-1.5 h per cell (one 8B load + residual extraction); 24 h covers all 14.

CELLS="${CELLS:-llama7b:awq llama7b:gptq4 llama7b:gptq3 llama7b:aqlm2 \
qwen7b:awq qwen7b:gptq4 qwen7b:gptq3 \
mistral7b-v02:awq mistral7b-v02:gptq4 mistral7b-v02:aqlm2 \
phi3:awq phi3:gptq4 phi3:gptq3 phi3:aqlm2}"

echo "=== X2 probe leakage control on quantized cells | $(date) ==="
echo "cells ($(echo ${CELLS} | wc -w)):"
for CELL in ${CELLS}; do echo "  ${CELL}"; done

for CELL in ${CELLS}; do
  MODEL="${CELL%%:*}"; QUANT="${CELL##*:}"
  echo "---- ${MODEL} / ${QUANT} | $(date) ----"
  python -u -m experiments.boundary.probe_leakage_control \
    --model "${MODEL}" --quantizer "${QUANT}" --n_calib 200 --n_transfer 200 \
    || echo "FAILED ${MODEL}/${QUANT}"
done
echo "=== done | $(date) ==="
