#!/bin/bash
#SBATCH --job-name=sq2_remaining
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_remaining_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/sq2_remaining_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

cd /jet/home/bhoomire/projects/safequant_v2

echo "=== Running Test B (behavioral_outcome_probe) ==="
python -u diagnostics/behavioral_outcome_probe.py --model llama7b --quantizer aqlm2 \
  --n_eval_ablation 200 --max_new_tokens 256 > outputs/run_behavioral_outcome_probe.log 2>&1

echo "=== Running F1/F2 across 14 cells ==="
> outputs/run_f1_f2.log

models=("llama7b" "mistral7b-v02" "qwen7b")
for model in "${models[@]}"; do
  if [ "$model" = "qwen7b" ]; then
    quantizers=("awq" "gptq4" "gptq3" "gptq2")
  else
    quantizers=("awq" "gptq4" "gptq3" "gptq2" "aqlm2")
  fi

  for q in "${quantizers[@]}"; do
    echo "Running F1 (direction_drift) on $model / $q" >> outputs/run_f1_f2.log
    python -u diagnostics/direction_drift.py --model "$model" --quantizer "$q" >> outputs/run_f1_f2.log 2>&1
    
    echo "Running F2 (layer_routing) on $model / $q" >> outputs/run_f1_f2.log
    python -u diagnostics/layer_routing.py --model "$model" --quantizer "$q" >> outputs/run_f1_f2.log 2>&1
  done
done

echo "=== Done | $(date) ==="
