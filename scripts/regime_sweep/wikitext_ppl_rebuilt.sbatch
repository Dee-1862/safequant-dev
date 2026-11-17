#!/bin/bash
#SBATCH --job-name=sq_wtppl_rebuilt
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/wtppl_rebuilt_%j.out
#SBATCH --error=scripts/logs/wtppl_rebuilt_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# WikiText-2 PPL check of ONE rebuilt GPTQ cell against the paper value.
# Submitted by scripts/delta_queue.sh --gptq-all after each build (afterok), e.g.
#   bash scripts/submit.sh scripts/regime_sweep/wikitext_ppl_rebuilt.sbatch \
#     --export=ALL,EXP_MODEL=phi3,EXP_QUANT=gptq4,WTPPL_FP32=1
# EXP_MODEL  model alias (llama7b, qwen7b, mistral7b-v02, phi3)
# EXP_QUANT  gptq4 | gptq3
# PPL_TAG    output tag (default rebuilt): outputs/regime_sweep/
#            wikitext_ppl_<m>_<q>_wikitext2_ppl_<tag>.json. Never overwrites.
# WTPPL_FP32 1 = also measure fp32 under the same tag (environment control; one job per
#            model does this), 0 = take the fp32 reference from the paper JSON.
# Prints rebuilt vs paper PPL and warns if they differ by more than 2%.
set -euo pipefail

MODEL="${EXP_MODEL:?set EXP_MODEL}"
QUANT="${EXP_QUANT:?set EXP_QUANT (gptq4|gptq3)}"
TAG="${PPL_TAG:-rebuilt}"
FP32="${WTPPL_FP32:-0}"
CKPT="${QM_ROOT}/gptq/${MODEL}-gptq${QUANT#gptq}bit-g128"

if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "ERROR: checkpoint not found at ${CKPT}" >&2; exit 1
fi
[[ -f "${CKPT}/sq_build_info.json" ]] && { echo "--- ${CKPT}/sq_build_info.json"; cat "${CKPT}/sq_build_info.json"; }

# AutoGPTQ checkpoints need model.safetensors + quantization_config for transformers
# loading (idempotent; a gptqmodel checkpoint already has both and is left as is).
echo "=== patch checkpoint for transformers loading | $(date) ==="
python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"

NOFP32=(); [[ "$FP32" == "1" ]] || NOFP32=(--no_fp32)
echo "=== wikitext2 ppl | ${MODEL}/${QUANT} | tag=${TAG} fp32=${FP32} | $(date) ==="
python -u -m experiments.regime_sweep.wikitext_ppl \
  --model "${MODEL}" --quantizers "${QUANT}" --seq_len 2048 --tag "${TAG}" "${NOFP32[@]}"

echo "=== rebuilt vs paper | $(date) ==="
python -u scripts/regime_sweep/compare_ppl_rebuilt.py --model "${MODEL}" --quant "${QUANT}" --tag "${TAG}"
echo "=== done | $(date) ==="
