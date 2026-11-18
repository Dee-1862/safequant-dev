#!/bin/bash
#SBATCH --job-name=sq_dense_gptq3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:45:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/dense_gptq3_%j.out
#SBATCH --error=scripts/logs/dense_gptq3_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# GPTQ-3 via dense weights: build and verify (option A). Per model (DENSE_MODELS, default
# "phi3 llama7b qwen7b"):
#   1) sq_autogptq: scripts/gptq_dequant_dense.py -> $QM_ROOT/gptq_dense/<m>-gptq3bit-g128
#      (per-layer max |auto-gptq - dense| on 4 real inputs; auto-gptq PPL on 20 windows)
#   2) sq_clean: PPL of the dense copy on the same 20 windows (diag_gptq3.py --method dense)
#   3) sq_clean: full WikiText-2 test PPL with SQ_GPTQ3_DENSE=1, tag dense, vs the paper
# SQ_GPTQ3_DENSE=1 is set only for step 3 here; no other job uses the dense copies.
# Submit:  SLURM_PARTITION=ghx4-interactive bash scripts/submit.sh \
#            scripts/tests/dense_gptq3.sbatch --cpus-per-task=8
set -uo pipefail   # not -e: report every model even if one step fails
for M in ${DENSE_MODELS:-phi3 llama7b qwen7b}; do
  echo "################ ${M} gptq3 | $(date)"
  python -u scripts/fix_autogptq_checkpoints.py "${QM_ROOT}/gptq/${M}-gptq3bit-g128" >/dev/null
  if [[ -f "${QM_ROOT}/gptq_dense/${M}-gptq3bit-g128/config.json" ]]; then
    echo "  dense copy exists; skipping dequantization"
  else
    "${AGENV}/bin/python" -u scripts/gptq_dequant_dense.py --model "$M" --bits 3 || { echo "RESULT ${M}: dequantization FAILED"; continue; }
  fi
  python -u scripts/tests/diag_gptq3.py --model "$M" --bits 3 --method dense
  SQ_GPTQ3_DENSE=1 python -u -m experiments.regime_sweep.wikitext_ppl \
    --model "$M" --quantizers gptq3 --seq_len 2048 --tag dense --no_fp32
  python -u scripts/regime_sweep/compare_ppl_rebuilt.py --model "$M" --quant gptq3 --tag dense || true
done
echo "=== summary"; grep -hE '^RESULT|^  [a-z0-9-]+ +gptq3 +[0-9.]+|^####' "scripts/logs/dense_gptq3_${SLURM_JOB_ID}.out" || true
echo "=== done | $(date) ==="
