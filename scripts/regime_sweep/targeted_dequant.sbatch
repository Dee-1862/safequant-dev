#!/bin/bash
#SBATCH --job-name=sq_tdequant
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/tdequant_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/tdequant_%j.err
source "${SLURM_SUBMIT_DIR:-/jet/home/bhoomire/projects/safequant_v2}/scripts/_env.sh"

# Section 8 -- targeted 4-bit (AWQ) restoration into the 2-bit (AQLM) cell.
# Two comparable read-side conditions so we can see if the localized middle-band
# repair matches the full read-side one:
#   1) read-side, ALL layers      -> read_awq4_all
#   2) read-side, MIDDLE band 11-22 -> read_awq4_L11-22
MODEL="${TD_MODEL:-llama7b}"

echo "=== targeted 4-bit dequant | ${MODEL} | $(date) ==="
echo "---- read-side / all layers ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer aqlm2 --source_quant awq --role read \
  --n_eval 200 --n_reasoning 50 || echo "FAILED read/all"
echo "---- read-side / middle band (11-22) ----"
python -u -m experiments.regime_sweep.targeted_dequant \
  --model ${MODEL} --quantizer aqlm2 --source_quant awq --role read \
  --layer_lo 11 --layer_hi 22 --n_eval 200 --n_reasoning 50 || echo "FAILED read/middle"
echo "=== done | $(date) ==="
