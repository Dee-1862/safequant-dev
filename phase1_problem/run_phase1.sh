#!/bin/bash
# ============================================================
# SafeQuant Phase 1 — Run everything in order.
# 
# Usage:
#   bash phase1_problem/run_phase1.sh qwen1.5b
#   bash phase1_problem/run_phase1.sh qwen3b
#   bash phase1_problem/run_phase1.sh all
# ============================================================

set -e

MODEL=${1:-"qwen1.5b"}

run_for_model() {
    local m=$1
    echo "============================================================"
    echo "  Running Phase 1 for: $m"
    echo "============================================================"
    
    # echo ""
    # echo "--- Step 0: Data Preparation ---"
    # python phase1_problem/data/prepare_data.py
    
    # echo ""
    # echo "--- Step 1a: Extract activations (FP16) ---"
    # python phase1_problem/step1_direction/extract_activations.py --model $m --precision fp16
    
    # echo ""
    # echo "--- Step 1b: Extract activations (NF4) ---"
    # python phase1_problem/step1_direction/extract_activations.py --model $m --precision nf4
    
    # echo ""
    # echo "--- Step 1c: Compute directions (FP16) ---"
    # python phase1_problem/step1_direction/compute_directions.py --model $m --precision fp16
    
    # echo ""
    # echo "--- Step 1d: Compute directions (NF4) ---"
    # python phase1_problem/step1_direction/compute_directions.py --model $m --precision nf4
    
    echo ""
    echo "--- Step 1e: Select best direction (FP16) ---"
    python phase1_problem/step1_direction/select_best_direction.py --model $m --precision fp16 --n_val_subset 16
    
    echo ""
    echo "--- Step 1f: Select best direction (NF4) ---"
    python phase1_problem/step1_direction/select_best_direction.py --model $m --precision nf4 --n_val_subset 16
    
    echo ""
    echo "--- Step 1g: Compare directions ---"
    python phase1_problem/step1_direction/compare_directions.py --model $m
    
    echo ""
    echo "--- Step 2a: Measure ASR (FP16) ---"
    python phase1_problem/step2_behavior/measure_asr.py --model $m --precision fp16
    
    echo ""
    echo "--- Step 2b: Measure ASR (NF4) ---"
    python phase1_problem/step2_behavior/measure_asr.py --model $m --precision nf4
    
    echo ""
    echo "--- Step 2c: Measure utility (FP16) ---"
    python phase1_problem/step2_behavior/measure_utility.py --model $m --precision fp16
    
    echo ""
    echo "--- Step 2d: Measure utility (NF4) ---"
    python phase1_problem/step2_behavior/measure_utility.py --model $m --precision nf4
    
    echo ""
    echo "--- Step 3a: Extract components (FP16) ---"
    python phase1_problem/step3_circuit/extract_component_outputs.py --model $m --precision fp16
    
    echo ""
    echo "--- Step 3b: Extract components (NF4) ---"
    python phase1_problem/step3_circuit/extract_component_outputs.py --model $m --precision nf4
    
    echo ""
    echo "--- Step 3c: Compute DFA (FP16) ---"
    python phase1_problem/step3_circuit/compute_dfa.py --model $m --precision fp16
    
    echo ""
    echo "--- Step 3d: Compute DFA (NF4) ---"
    python phase1_problem/step3_circuit/compute_dfa.py --model $m --precision nf4
    
    echo ""
    echo "--- Step 3e: Compare circuits ---"
    python phase1_problem/step3_circuit/compare_and_visualize.py --model $m
    
    echo ""
    echo "============================================================"
    echo "  Phase 1 complete for: $m"
    echo "============================================================"
}

if [ "$MODEL" = "all" ]; then
    run_for_model "qwen1.5b"
    run_for_model "qwen3b"
else
    run_for_model "$MODEL"
fi

echo ""
echo "--- Compiling summary ---"
python phase1_problem/summary/compile_phase1.py

echo ""
echo "============================================================"
echo "  ALL DONE. Check phase1_problem/summary/ for results."
echo "============================================================"
