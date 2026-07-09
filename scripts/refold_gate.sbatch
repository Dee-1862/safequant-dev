#!/bin/bash
#SBATCH -A cis250287p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -o refold_gate_%j.log

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
module load gcc/10.2.0
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
cd /jet/home/bhoomire/projects/safequant_v2

# Crux gate: prove the AQLM requant-fold drives reconstruction to the quant floor
# AND preserves bit width, before building LoRA/DPO around it.
python -u -m diagnostics.qresafe_refold
