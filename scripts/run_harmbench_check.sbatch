#!/bin/bash
#SBATCH --job-name=sq2_harmbench
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_harmbench_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_harmbench_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cd /jet/home/bhoomire/projects/safequant_v2

echo "=== Running F1 calibration sensitivity check (HarmBench) ==="
# Write output to a specific file to avoid overwriting the 14-cell default wildjailbreak JSON
python -u diagnostics/direction_drift.py --model llama7b --quantizer aqlm2 \
  --dataset harmbench \
  --output outputs/direction_drift_llama7b_aqlm2_harmbench.json \
  > outputs/run_harmbench_check.log 2>&1

echo "=== Done | $(date) ==="
