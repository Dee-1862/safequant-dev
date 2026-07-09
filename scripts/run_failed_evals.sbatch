#!/bin/bash
#SBATCH --job-name=sq2_retry
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_retry_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_retry_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Running failed evals | $(date) ==="
cd /jet/home/bhoomire/projects/safequant_v2

echo "Running probe_causal_ablation..."
python -u diagnostics/probe_causal_ablation.py --model llama7b --quantizer aqlm2 \
  --n_eval 200 --max_new_tokens 256 > outputs/run_probe_causal_ablation.log 2>&1

echo "Running direction_drift sensitivity..."
python -u diagnostics/direction_drift.py --model llama7b --quantizer aqlm2 \
  --dataset harmbench > outputs/run_direction_drift.log 2>&1

echo "=== Done | $(date) ==="
