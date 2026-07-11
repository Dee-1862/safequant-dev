#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -o roundtrip_gate_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
cd /jet/home/bhoomire/projects/safequant_v2

# Tier 2 bridge gate: prove the HF<->repo fold-writeback round-trips the headline
# AQLM-2 checkpoint with ZERO delta bit-identically, removing the bridge as a
# suspect before any DPO is run.
python -u -m experiments.trained_repair.qresafe_fold
