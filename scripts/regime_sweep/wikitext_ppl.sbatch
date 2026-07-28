#!/bin/bash
#SBATCH --job-name=sq_wtppl
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/wtppl_%j.out
#SBATCH --error=scripts/logs/wtppl_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# WikiText-2 perplexity for ONE model across all quantizer cells (fp32 + 5 quant).
# Continuous coherence metric complementing ARC. Submit once per model:
#   EXP_MODEL in {llama7b, mistral7b-v02, qwen7b, phi3}
# phi3/awq is the NA cell (arch mismatch) and is skipped gracefully.
MODEL="${EXP_MODEL:-llama7b}"
QUANTS="${WTPPL_QUANTS:-awq gptq4 aqlm2 gptq3 gptq2}"

echo "=== wikitext2 ppl | ${MODEL} | quants='${QUANTS}' | $(date) ==="
python -u -m experiments.regime_sweep.wikitext_ppl \
  --model ${MODEL} --quantizers ${QUANTS} --seq_len 2048
echo "=== done | $(date) ==="
