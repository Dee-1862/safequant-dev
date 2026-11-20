#!/bin/bash
#SBATCH --job-name=sq_heldout
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/heldout_llama3_8b_aqlm2_%j.out
#SBATCH --error=scripts/logs/heldout_llama3_8b_aqlm2_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Held-out two-bit AQLM check: a checkpoint NOT used while designing the screen.
#   child  = config.py ("llama3-8b", "aqlm2") -> ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16
#   parent = config.py MODELS["llama3-8b"]    -> meta-llama/Meta-Llama-3-8B-Instruct (gated)
# Needs `huggingface-cli login` with Llama access on the node/HF_HOME used here.
#
# Submit:  bash scripts/submit.sh scripts/headline/headline_eval_llama3_8b_aqlm2_heldout.sbatch
# Output:  outputs/headline/headline_<model>_aqlm2_heldout.json  (new name; never overwrites)
#
# Screen order: coherence (ARC, in headline) -> behaviour vs parent (paired McNemar,
# below) -> collapse check (read ARC + ASR in the headline JSON).

MODEL="${EXP_MODEL:-llama3-8b}"

echo "=== held-out headline eval ${MODEL}/aqlm2 | $(date) ==="
python -u -m experiments.headline.headline_eval \
  --model "${MODEL}" --quantizer aqlm2 \
  --n_eval 200 --n_reasoning 100 --n_boot 2000 \
  --tag heldout

echo "=== paired ΔASR screen (McNemar) ${MODEL}/aqlm2 | $(date) ==="
python -u -m experiments.regime_sweep.paired_screen \
  --cells "${MODEL}:aqlm2" --variant heldout
echo "=== done | $(date) ==="
