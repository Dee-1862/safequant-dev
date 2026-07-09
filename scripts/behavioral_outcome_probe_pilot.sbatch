#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Behavioral-outcome probe — PILOT on Llama-3.1-8B / AQLM-2 (well-posed cell).
#
# Trains a per-layer probe that predicts response label (jailbroken vs
# refused) on the quant model's residuals, then ablates the best-layer
# probe direction and measures held-out ASR change. If the direction is
# operative (|ΔASR| ≥ 5pp) we extend to 2-3 more conditions.
#
# Output: outputs/behavioral_outcome_probe_llama7b_aqlm2.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_bop_pilot
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/bop_pilot_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/bop_pilot_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | Behavioral-outcome probe PILOT (Llama/AQLM-2) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

python -u diagnostics/behavioral_outcome_probe.py \
    --model llama7b --quantizer aqlm2 \
    --n_train_prompts 200 --n_eval_ablation 100

echo "=== Behavioral-outcome probe PILOT complete | $(date) ==="
