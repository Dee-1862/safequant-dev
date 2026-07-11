#!/bin/bash
# Download and cache all benchmark datasets
#
# This downloads:
# - AdvBench (harmful prompts)
# - Anthropic HH-RLHF (harmless prompts)
# - GSM8K (reasoning prompts)
#
# Usage:
#   bash download_datasets.sh [n_samples]

N_SAMPLES=${1:-500}

echo "========================================================================"
echo "Downloading Benchmark Datasets for SafeQuant"
echo "========================================================================"
echo "Samples per dataset: $N_SAMPLES"
echo ""

python core/datasets.py --n_samples $N_SAMPLES --output_dir data

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ All datasets downloaded successfully!"
    echo "========================================================================"
    echo ""
    echo "JSON files saved to data/:"
    ls -lh data/*.json
    echo ""
    echo "Datasets are also cached by HuggingFace for faster future loading."
    echo ""
else
    echo ""
    echo "✗ Dataset download failed!"
    echo "Check your internet connection and HuggingFace access."
    exit 1
fi
