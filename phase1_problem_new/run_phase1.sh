#!/bin/bash
# ============================================================
# SafeQuant Phase 1 — Prove That the Problem Exists
#
# This phase demonstrates:
#   Finding 1: Quantization breaks safety (ASR doubles)
#   Finding 2: Standard metrics miss it (perplexity, Frobenius)
#   Finding 3: The refusal direction survives
#   Finding 4: But safety circuits weaken
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
    echo ""
    echo "########################################################"
    echo "#  SafeQuant Phase 1: Does Quantization Break Safety?  #"
    echo "#  Model: $m                                           #"
    echo "########################################################"
    echo ""

    # === SETUP ===
    echo ">>> Step 0: Data Preparation"
    python phase1_problem/data/prepare_data.py
    echo ""

    # === STEP 1: REFUSAL DIRECTION ===
    # Extract the direction that separates harmful from harmless activations
    # in the model's residual stream. Do this for both FP16 and NF4.

    echo ">>> Step 1a: Extract activations (FP16)"
    python phase1_problem/step1_direction/extract_activations.py --model $m --precision fp16
    echo ""

    echo ">>> Step 1b: Extract activations (NF4)"
    python phase1_problem/step1_direction/extract_activations.py --model $m --precision nf4
    echo ""

    echo ">>> Step 1c: Compute candidate directions (FP16)"
    python phase1_problem/step1_direction/compute_directions.py --model $m --precision fp16
    echo ""

    echo ">>> Step 1d: Compute candidate directions (NF4)"
    python phase1_problem/step1_direction/compute_directions.py --model $m --precision nf4
    echo ""

    echo ">>> Step 1e: Select best direction (FP16)"
    python phase1_problem/step1_direction/select_best_direction.py --model $m --precision fp16
    echo ""

    echo ">>> Step 1f: Select best direction (NF4)"
    python phase1_problem/step1_direction/select_best_direction.py --model $m --precision nf4
    echo ""

    echo ">>> Step 1g: Compare FP16 vs NF4 directions [FINDING 3]"
    python phase1_problem/step1_direction/compare_directions.py --model $m
    echo ""

    # === STEP 2: BEHAVIORAL MEASUREMENT ===
    # Show safety degrades while standard quality metrics don't notice.

    echo ">>> Step 2a: Measure ASR (FP16) [FINDING 1]"
    python phase1_problem/step2_behavior/measure_asr.py --model $m --precision fp16
    echo ""

    echo ">>> Step 2b: Measure ASR (NF4) [FINDING 1]"
    python phase1_problem/step2_behavior/measure_asr.py --model $m --precision nf4
    echo ""

    echo ">>> Step 2c: Measure Utility (FP16) [FINDING 2]"
    python phase1_problem/step2_behavior/measure_utility.py --model $m --precision fp16
    echo ""

    echo ">>> Step 2d: Measure Utility (NF4) [FINDING 2]"
    python phase1_problem/step2_behavior/measure_utility.py --model $m --precision nf4
    echo ""

    echo ">>> Step 2e: Measure Perplexity (FP16) [FINDING 2]"
    python phase1_problem/step2_behavior/measure_perplexity.py --model $m --precision fp16
    echo ""

    echo ">>> Step 2f: Measure Perplexity (NF4) [FINDING 2]"
    python phase1_problem/step2_behavior/measure_perplexity.py --model $m --precision nf4
    echo ""

    # === STEP 3: CIRCUIT ANALYSIS ===
    # Use DFA to identify which components lose safety signal,
    # and show that Frobenius norm can't identify them.

    echo ">>> Step 3a: Extract component outputs (FP16)"
    python phase1_problem/step3_circuit/extract_component_outputs.py --model $m --precision fp16
    echo ""

    echo ">>> Step 3b: Extract component outputs (NF4)"
    python phase1_problem/step3_circuit/extract_component_outputs.py --model $m --precision nf4
    echo ""

    echo ">>> Step 3c: Compute DFA (FP16)"
    python phase1_problem/step3_circuit/compute_dfa.py --model $m --precision fp16
    echo ""

    echo ">>> Step 3d: Compute DFA (NF4)"
    python phase1_problem/step3_circuit/compute_dfa.py --model $m --precision nf4
    echo ""

    echo ">>> Step 3e: Compare FP16 vs NF4 circuits [FINDING 4]"
    python phase1_problem/step3_circuit/compare_and_visualize.py --model $m
    echo ""

    echo ">>> Step 3f: Global Frobenius norm analysis [FINDING 2c]"
    echo "    (ZERO INFERENCE — just compares weight matrices)"
    python phase1_problem/step3_circuit/global_frobenius.py --model $m
    echo ""

    echo "########################################################"
    echo "#  Phase 1 complete for: $m                            #"
    echo "########################################################"
}

if [ "$MODEL" = "all" ]; then
    run_for_model "qwen1.5b"
    run_for_model "qwen3b"
else
    run_for_model "$MODEL"
fi

echo ""
echo ">>> Compiling summary..."
if [ "$MODEL" = "all" ]; then
    python phase1_problem/summary/compile_phase1.py
else
    python phase1_problem/summary/compile_phase1.py --model $MODEL
fi

echo ""
echo "########################################################"
echo "#  ALL DONE.                                           #"
echo "#  Results: phase1_problem/summary/outputs/            #"
echo "#  Plots:   phase1_problem/step*/outputs/              #"
echo "########################################################"
