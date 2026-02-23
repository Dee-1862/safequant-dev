#!/bin/bash
# ============================================================
# SafeQuant — Clone required repositories
# Run this once before starting any phase.
# ============================================================

set -e

REPO_DIR="repos"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

echo "=== Cloning Arditi's refusal_direction ==="
echo "This is our foundation for direction extraction, selection, ablation."
if [ ! -d "refusal_direction" ]; then
    git clone https://github.com/andyrdt/refusal_direction.git
else
    echo "Already cloned."
fi

echo ""
echo "=== Cloning HarmBench ==="
echo "Our primary safety evaluation benchmark across all phases."
if [ ! -d "HarmBench" ]; then
    git clone https://github.com/centerforaisafety/HarmBench.git
else
    echo "Already cloned."
fi

echo ""
echo "=== Cloning AUSteer ==="
echo "Baseline for Phase 3 comparison."
if [ ! -d "AUSteer" ]; then
    git clone https://github.com/zijian678/AUSteer.git
else
    echo "Already cloned."
fi

echo ""
echo "=== Cloning JailbreakBench ==="
echo "Additional harmful prompt dataset."
if [ ! -d "jailbreakbench" ]; then
    git clone https://github.com/JailbreakBench/jailbreakbench.git
else
    echo "Already cloned."
fi

echo ""
echo "=== All repos cloned successfully ==="
echo ""
echo "Directory structure:"
ls -la
