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

# WikiText-2 PPL check of ONE cell against the paper value: a rebuilt GPTQ cell
# (delta_queue.sh --gptq-all, after its build) or a public AWQ checkpoint on Delta's stack
# (delta_queue.sh --awq-ppl, EXP_QUANT=awq PPL_TAG=delta: aarch64 uses autoawq's Triton
# path, Bridges used awq_ext). Example:
#   bash scripts/submit.sh scripts/regime_sweep/wikitext_ppl_rebuilt.sbatch \
#     --export=ALL,EXP_MODEL=phi3,EXP_QUANT=gptq4,WTPPL_FP32=1
# EXP_MODEL  model alias (llama7b, qwen7b, mistral7b-v02, phi3)
# EXP_QUANT  gptq4 | gptq3 | awq
# PPL_TAG    output tag (default rebuilt): outputs/regime_sweep/
#            wikitext_ppl_<m>_<q>_wikitext2_ppl_<tag>.json. Never overwrites.
# WTPPL_FP32 1 = also measure fp32 under the same tag (environment control; one job per
#            model does this), 0 = take the fp32 reference from the paper JSON.
# Prints rebuilt vs paper PPL and warns if they differ by more than 2%.
set -euo pipefail

MODEL="${EXP_MODEL:?set EXP_MODEL}"
QUANT="${EXP_QUANT:?set EXP_QUANT (gptq4|gptq3|awq)}"
TAG="${PPL_TAG:-rebuilt}"
FP32="${WTPPL_FP32:-0}"
case "$QUANT" in
  gptq*) CKPT="${QM_ROOT}/gptq/${MODEL}-gptq${QUANT#gptq}bit-g128" ;;
  awq)   CKPT="${QM_ROOT}/awq/${MODEL}" ;;   # public checkpoint (delta_setup); config.py path
  *) echo "ERROR: EXP_QUANT must be gptq4, gptq3 or awq (got ${QUANT})" >&2; exit 2 ;;
esac

if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "ERROR: checkpoint not found at ${CKPT}" >&2; exit 1
fi
[[ -f "${CKPT}/sq_build_info.json" ]] && { echo "--- ${CKPT}/sq_build_info.json"; cat "${CKPT}/sq_build_info.json"; }

if [[ "$QUANT" == gptq* ]]; then
  # AutoGPTQ checkpoints need model.safetensors + quantization_config for transformers
  # loading (idempotent; a gptqmodel checkpoint already has both and is left as is).
  echo "=== patch checkpoint for transformers loading | $(date) ==="
  python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"
else
  # AWQ on aarch64 runs autoawq's Triton path (no awq_ext wheels); record which one.
  python - <<'PY' || true
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh
from awq.modules.linear.gemm import awq_ext, TRITON_AVAILABLE
print(f"  AWQ kernels: awq_ext={awq_ext is not None}  triton={TRITON_AVAILABLE}")
PY
fi

# Tagged outputs are never overwritten (wikitext_ppl refuses). On a resubmit, skip what
# already exists instead of failing: the cell itself, or just its fp32 control.
_out() { python -c "from core.manifest import RunConfig, result_path; print(result_path('experiments/regime_sweep/wikitext_ppl.py', RunConfig(model='${MODEL}', quantizer='$1', variant='wikitext2_ppl_${TAG}')))"; }
if [[ -f "$(_out "${QUANT}")" ]]; then
  echo "=== $(_out "${QUANT}") already exists; not re-measuring | $(date) ==="
else
  if [[ "$FP32" == "1" && -f "$(_out fp32)" ]]; then
    echo "  fp32 control already measured under tag ${TAG} ($(_out fp32)); skipping fp32"
    FP32=0
  fi
  NOFP32=(); [[ "$FP32" == "1" ]] || NOFP32=(--no_fp32)
  echo "=== wikitext2 ppl | ${MODEL}/${QUANT} | tag=${TAG} fp32=${FP32} | $(date) ==="
  python -u -m experiments.regime_sweep.wikitext_ppl \
    --model "${MODEL}" --quantizers "${QUANT}" --seq_len 2048 --tag "${TAG}" "${NOFP32[@]}"
fi

echo "=== rebuilt vs paper | $(date) ==="
python -u scripts/regime_sweep/compare_ppl_rebuilt.py --model "${MODEL}" --quant "${QUANT}" --tag "${TAG}"
echo "=== done | $(date) ==="
