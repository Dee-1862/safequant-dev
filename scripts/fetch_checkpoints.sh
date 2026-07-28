#!/bin/bash
# Download pre-quantized AWQ/AQLM checkpoints into QM_ROOT (paths in config.py).
# GPTQ is self-generated: python scripts/quantize_gptq.py --model llama7b --bits 4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/load_env.sh"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export QM_ROOT="${QM_ROOT:-${HOME}/quantized_models}"
mkdir -p "${HF_HOME}" "${HF_HOME}/datasets"
mkdir -p "${QM_ROOT}/awq" "${QM_ROOT}/aqlm" "${QM_ROOT}/gptq"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Run: pip install -U huggingface_hub"
  exit 1
fi

fetch() {
  local repo="$1"
  local dest="$2"
  if [[ -d "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null)" ]]; then
    echo "skip (exists): ${dest}"
    return 0
  fi
  echo "download: ${repo} -> ${dest}"
  huggingface-cli download "${repo}" --local-dir "${dest}"
}

echo "HF_HOME=${HF_HOME}"
echo "QM_ROOT=${QM_ROOT}"
echo "Run huggingface-cli login first if any model is gated."

# AWQ + AQLM (destinations match config.py QUANTIZED_CHECKPOINTS)
fetch hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 "${QM_ROOT}/awq/llama7b"
fetch ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf "${QM_ROOT}/aqlm/llama7b"
fetch TheBloke/Mistral-7B-Instruct-v0.2-AWQ "${QM_ROOT}/awq/mistral7b-v02"
fetch ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8 "${QM_ROOT}/aqlm/mistral7b-v02"
fetch Qwen/Qwen2.5-7B-Instruct-AWQ "${QM_ROOT}/awq/qwen7b"
fetch Sreenington/Phi-3-mini-4k-instruct-AWQ "${QM_ROOT}/awq/phi3"
fetch ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf "${QM_ROOT}/aqlm/phi3"

echo "Done. FP32 bases cache in HF_HOME on first load. Generate GPTQ with scripts/quantize_gptq.py."
