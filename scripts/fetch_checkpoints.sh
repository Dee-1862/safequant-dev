#!/bin/bash
# scripts/fetch_checkpoints.sh
#
# Download every reputable base + AWQ + AQLM checkpoint into ONE consolidated
# tree on Ocean, with per-method subfolders -- so no cell depends on a scattered
# HF hub cache. GPTQ cells are self-generated separately by quantize_gptq.py into
# <root>/<alias>-gptq<bits>bit-g128/.
#
# Layout created under $QM_ROOT:
#   base/<alias>/   unquantized
#   awq/<alias>/    AWQ-4bit
#   aqlm/<alias>/   AQLM-2bit
#
# Run on a LOGIN node (needs internet + a HF token for gated repos: `huggingface-cli login`).
#   bash scripts/fetch_checkpoints.sh
# Skips anything already fully downloaded (huggingface-cli is resumable/idempotent).
set -uo pipefail   # NOT -e: one failed (e.g. gated) repo must not abort the rest

QM_ROOT="${QM_ROOT:-/ocean/projects/cis240116p/bhoomire/quantized_models}"
FAILED=""

dl() {  # dl <subdir/alias> <repo> [revision]
  local dest="$QM_ROOT/$1" repo="$2" rev="${3:-main}"
  echo "=== $repo @ ${rev} -> $dest ==="
  if huggingface-cli download "$repo" --revision "$rev" \
       --local-dir "$dest" \
       --exclude "original/*" "*.gguf" "*.pth" "consolidated*"; then
    echo "  ok: $1"
  else
    echo "  !! FAILED: $1 ($repo) -- check 'huggingface-cli login' / gated access"
    FAILED="$FAILED $1"
  fi
}

echo "consolidating checkpoints under: $QM_ROOT"

# ---------- base (unquantized) ----------
dl base/llama7b        meta-llama/Llama-3.1-8B-Instruct
dl base/qwen7b         Qwen/Qwen2.5-7B-Instruct
dl base/mistral7b-v02  mistralai/Mistral-7B-Instruct-v0.2
dl base/phi3           microsoft/Phi-3-mini-4k-instruct                                    ff07dc01615f8113924aed013115ab2abd32115b

# ---------- AWQ (4-bit) ----------
dl awq/llama7b         hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
dl awq/qwen7b          Qwen/Qwen2.5-7B-Instruct-AWQ
dl awq/mistral7b-v02   TheBloke/Mistral-7B-Instruct-v0.2-AWQ
dl awq/phi3            Sreenington/Phi-3-mini-4k-instruct-AWQ                              7479fcd4153c520f59601d2f9dd08c86daa35f20

# ---------- AQLM (2-bit) ----------  (no qwen by design)
dl aqlm/llama7b        ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf
dl aqlm/mistral7b-v02  ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8
dl aqlm/phi3           ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf            90cc18c5d0985a9105b1d551515d38e716dfc274

echo ""
if [ -n "$FAILED" ]; then
  echo "!! SOME DOWNLOADS FAILED:$FAILED"
  echo "   Run 'huggingface-cli login' (needs Llama/Mistral gated access) and re-run"
  echo "   this script -- it resumes and only re-fetches what's missing."
else
  echo "DONE. Consolidated layout under $QM_ROOT:"
  echo "  base/{llama7b,qwen7b,mistral7b-v02,phi3}"
  echo "  awq/{llama7b,qwen7b,mistral7b-v02,phi3}"
  echo "  aqlm/{llama7b,mistral7b-v02,phi3}"
  echo "  <alias>-gptq{2,3,4}bit-g128  (self-generated separately)"
fi
