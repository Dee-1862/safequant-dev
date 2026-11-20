#!/bin/bash
#SBATCH --job-name=sq_dense_gptq
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/dense_gptq_%j.out
#SBATCH --error=scripts/logs/dense_gptq_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# GPTQ via dense weights: build and verify, for every bit width (one load path for all GPTQ).
# Cells = DENSE_BITS x DENSE_MODELS (defaults "4 2" x "phi3 llama7b qwen7b mistral7b-v02";
# a cell whose GPTQ checkpoint is missing is skipped). Per cell:
#   1) sq_autogptq: scripts/gptq_dequant_dense.py -> $QM_ROOT/gptq_dense/<m>-gptq<b>bit-g128
#      (per-layer max |auto-gptq - dense| on 4 real inputs; auto-gptq PPL on 20 windows)
#   2) sq_clean: PPL of the dense copy on the same 20 windows (diag_gptq3.py --method dense)
#   3) sq_clean: full WikiText-2 test PPL with SQ_GPTQ_DENSE=1, tag rebuilt
#   4) compare with the paper, and with rebuilt_gptqmodel (same checkpoint loaded by gptqmodel)
# SQ_GPTQ_DENSE=1 is set only for step 3 here.
# Submit:  SLURM_PARTITION="$SLURM_INTERACTIVE_PARTITION" DENSE_BITS="4 2" bash scripts/submit.sh \
#            scripts/tests/dense_gptq.sbatch --cpus-per-task=8
set -uo pipefail   # not -e: report every cell even if one step fails
FAIL=0             # exit 1 at the end if any requested cell has no dense copy (afterok chains)
for B in ${DENSE_BITS:-4 2}; do
for M in ${DENSE_MODELS:-phi3 llama7b qwen7b mistral7b-v02}; do
  CK="${QM_ROOT}/gptq/${M}-gptq${B}bit-g128"
  echo "################ ${M} gptq${B} | $(date)"
  if [[ ! -f "${CK}/config.json" ]]; then echo "RESULT ${M} gptq${B}: no checkpoint at ${CK}; skipped"; FAIL=1; continue; fi
  python -u scripts/fix_autogptq_checkpoints.py "${CK}" >/dev/null
  if [[ -f "${QM_ROOT}/gptq_dense/${M}-gptq${B}bit-g128/config.json" ]]; then
    echo "  dense copy exists; skipping dequantization"
  else
    "${AGENV}/bin/python" -u scripts/gptq_dequant_dense.py --model "$M" --bits "$B" || { echo "RESULT ${M} gptq${B}: dequantization FAILED"; FAIL=1; continue; }
  fi
  python -u scripts/tests/diag_gptq3.py --model "$M" --bits "$B" --method dense
  SQ_GPTQ_DENSE=1 python -u -m experiments.regime_sweep.wikitext_ppl \
    --model "$M" --quantizers "gptq${B}" --seq_len 2048 --tag rebuilt --no_fp32
  python -u scripts/regime_sweep/compare_ppl_rebuilt.py --model "$M" --quant "gptq${B}" --tag rebuilt || true
  python -u scripts/regime_sweep/compare_ppl_rebuilt.py --model "$M" --quant "gptq${B}" --tag rebuilt --ref_tag rebuilt_gptqmodel || true
done
done
echo "=== summary"; cat "scripts/logs/dense_gptq_${SLURM_JOB_ID}.out" | grep -hE '^RESULT|^  [a-z0-9-]+ +gptq[0-9] +[0-9.]+|^####' || true   # via cat: grep refuses its own output file
echo "=== done | $(date) | exit ${FAIL} ==="
exit "${FAIL}"
