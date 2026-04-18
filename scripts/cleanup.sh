#!/bin/bash
# SafeQuant cleanup — moves unneeded scripts and logs to _trash/
# Review _trash/ before permanently deleting

set -e
BASE="/jet/home/bhoomire/projects/safequant"
TRASH="$BASE/_trash"

mkdir -p "$TRASH/scripts" "$TRASH/logs" "$TRASH/jobs"

echo "=== Moving unneeded scripts to _trash/scripts/ ==="

# Superseded by run_pipeline.sbatch
for f in \
  run_phase1_phase2.sbatch \
  run_integrity.sbatch \
  run_safequant.sbatch \
  run_safequant_fixed.sbatch \
  run_safequant_gpu.sbatch \
  run_phased_diagnostics.sh \
; do
  [ -f "$BASE/scripts/$f" ] && mv "$BASE/scripts/$f" "$TRASH/scripts/" && echo "  moved $f"
done

# Duplicate per-model variants
for f in \
  run_final_llama.sbatch \
  run_final_mistral.sbatch \
  run_final_qwen7b.sbatch \
  run_final_qwen7b_readwrite.sbatch \
  run_llama_full_test.sbatch \
  run_llama_readwrite.sbatch \
  run_llama_with_fp32.sbatch \
  run_qwen_full_benchmark.sbatch \
  run_qwen_readwrite.sbatch \
  run_diagnostic_llama.sbatch \
  run_diagnostic_mistral.sbatch \
  run_diagnostic_multi.sbatch \
  run_safety_diag_llama.sbatch \
  run_safety_patch_llama.sbatch \
  run_safety_patch_mistral.sbatch \
; do
  [ -f "$BASE/scripts/$f" ] && mv "$BASE/scripts/$f" "$TRASH/scripts/" && echo "  moved $f"
done

# Quantizer iteration junk
for f in \
  test_quantizers.sbatch \
  test_llama_quantizers.sbatch \
  test_all_quantizers_llama.sbatch \
  test_sota_quantizers.sbatch \
  install_and_run_multi_q.sbatch \
  multi_quantizer_test.sbatch \
  run_multi_q_clean.sbatch \
  run_llama_multi_q.sbatch \
  run_dct_llama2.sbatch \
; do
  [ -f "$BASE/scripts/$f" ] && mv "$BASE/scripts/$f" "$TRASH/scripts/" && echo "  moved $f"
done

# One-off tests
for f in \
  mixed_prompt_test.sbatch \
  rerun_phases23.sbatch \
  run_pipeline_large.sbatch \
  test_setup.sh \
  quantize_gptq.py \
; do
  [ -f "$BASE/scripts/$f" ] && mv "$BASE/scripts/$f" "$TRASH/scripts/" && echo "  moved $f"
done

echo ""
echo "=== Moving unneeded logs to _trash/logs/ ==="

# Logs to KEEP (latest successful run of each experiment)
KEEP_LOGS=(
  sq_adv_asr_38413355
  sq_p23_wild_38419570
  sq_diag_38441845
  sq_theory_38497361
  sq_patch_l_38484217
)

for f in "$BASE/scripts/logs/"*; do
  fname=$(basename "$f")
  keep=false
  for k in "${KEEP_LOGS[@]}"; do
    if [[ "$fname" == "$k"* ]]; then
      keep=true
      break
    fi
  done
  if [ "$keep" = false ]; then
    mv "$f" "$TRASH/logs/" && echo "  moved $fname"
  fi
done

echo ""
echo "=== Moving old job logs to _trash/jobs/ ==="
for f in "$BASE/jobs/"*.err "$BASE/jobs/"*.out; do
  [ -f "$f" ] && mv "$f" "$TRASH/jobs/" && echo "  moved $(basename $f)"
done

echo ""
echo "=== Done ==="
echo "Kept scripts:"
ls "$BASE/scripts/"*.sbatch "$BASE/scripts/"*.sh 2>/dev/null | xargs -I{} basename {}
echo ""
echo "Kept logs:"
ls "$BASE/scripts/logs/" 2>/dev/null
echo ""
echo "Trash is at: $TRASH"
echo "Run 'rm -rf $TRASH' when you're satisfied everything important is preserved."
