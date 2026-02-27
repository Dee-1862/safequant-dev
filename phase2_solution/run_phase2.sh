#!/bin/bash
# ============================================================
# SafeQuant Phase 2 v2 — Per-Dimension Vulnerability Pipeline
#
# Usage:
#   bash phase2_solution/run_phase2.sh qwen1.5b
#   bash phase2_solution/run_phase2.sh qwen1.5b --skip_categorical
# ============================================================

set -e

MODEL=${1:-"qwen1.5b"}
SKIP_CAT=${2:-""}

echo "============================================================"
echo "  SafeQuant Phase 2 v2: $MODEL"
echo "============================================================"

echo ""
echo "--- Step 1: Per-Dimension DFA (FP16) ---"
python phase2_solution/diagnosis/per_dim_dfa.py --model $MODEL --precision fp16

echo ""
echo "--- Step 2: Per-Dimension DFA (NF4) ---"
python phase2_solution/diagnosis/per_dim_dfa.py --model $MODEL --precision nf4

echo ""
echo "--- Step 3: Per-Row Quantization Error ---"
python phase2_solution/diagnosis/per_row_quant_error.py --model $MODEL

echo ""
echo "--- Step 4: Vulnerability Scores ---"
python phase2_solution/diagnosis/vulnerability_score.py --model $MODEL

echo ""
echo "--- Step 5: Apply Sparse ΔW Repairs ---"
python phase2_solution/repair/apply_repairs.py --model $MODEL --top_k 100

echo ""
echo "--- Step 6: Evaluate All Conditions ---"
python phase2_solution/evaluate/eval_all.py --model $MODEL

echo ""
echo "--- Compiling Summary ---"
python phase2_solution/summary/compile_phase2.py

echo ""
echo "============================================================"
echo "  Phase 2 v2 COMPLETE for $MODEL"
echo "  Key outputs:"
echo "    diagnosis/outputs/$MODEL/vulnerability_analysis.png"
echo "    evaluate/outputs/$MODEL/asr_comparison.png"
echo "    evaluate/outputs/$MODEL/comparison_results.json"
echo "============================================================"
