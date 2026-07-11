#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F2a — Random-direction ablation distribution, Qwen2.5-7B / GPTQ-3.
#
# RESCOPED FROM FIRST RUN (40882624, timed out at 4h with 2/20 trials done).
# Qwen GPTQ-3 uses the auto_gptq Triton kernel which is ~10× slower than
# AWQ/AQLM kernels: ~45-50s per generation. To make the run fit:
#   - K=10 (was 20). Halves trials; SD estimate still meaningful at N=10.
#   - n_eval=50 (was 100). Halves prompts per trial.
# Combined: ~11 evals × 50 prompts × 50s = ~9000s = ~2.5h core work +
# load/classify ~30min = ~3h actual. 8h walltime gives buffer.
#
# Output: outputs/random_ablation_dist_qwen7b_gptq3.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_rad_qwen_gptq3
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rad_qwen_gptq3_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/rad_qwen_gptq3_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | F2a random ablation Qwen/GPTQ-3 (rescoped) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

python -u diagnostics/random_ablation_distribution.py \
    --model qwen7b --quantizer gptq3 \
    --n_eval 50 --n_random 10

echo "=== F2a random ablation Qwen/GPTQ-3 complete | $(date) ==="
