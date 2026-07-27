#!/bin/bash
#SBATCH --job-name=sq_smoke_new
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/smoke_new_%j.out
#SBATCH --error=scripts/logs/smoke_new_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Smoke the pipelines added this cycle at tiny n. Catches import/API breakage and
# exercises label dump, scope log, whole-model bits, and PPL before full jobs.
# NOT for paper numbers.
set -euo pipefail
MODEL="${SMOKE_MODEL:-llama7b}"

echo "=== [1/4] paired_screen (CPU, reads flags_by_cond from outputs/) ==="
python -u -m experiments.regime_sweep.paired_screen

echo "=== [2/4] wikitext2 ppl (2 windows, fp32 + aqlm2) ==="
python -u -m experiments.regime_sweep.wikitext_ppl \
  --model "${MODEL}" --quantizers aqlm2 --seq_len 512 --max_chunks 2

echo "=== [3/4] restoration_sweep (verify flags_by_cond persists for the screen) ==="
python -u -m experiments.regime_sweep.restoration_sweep \
  --model "${MODEL}" --quantizer aqlm2 --n_eval 16 --n_reasoning 8
python - <<'PY'
import glob, json, os
fs = glob.glob("outputs/**/restoration_sweep_llama_aqlm2*.json", recursive=True)
d = json.load(open(max(fs, key=os.path.getmtime)))
fbc = d.get("flags_by_cond")
assert fbc and "fp32" in fbc and "aqlm2" in fbc, f"flags_by_cond broken: {list(fbc or [])}"
print("  flags_by_cond OK:", list(fbc), "| n =", len(fbc["fp32"]))
PY

echo "=== [4/4] param_matched_restore (scope + param-budget audit log) ==="
python -u -m experiments.regime_sweep.param_matched_restore \
  --model "${MODEL}" --quantizer aqlm2 --n_eval 16 --n_reasoning 8 --seeds 1

echo "=== smoke new pipelines OK | $(date) ==="
