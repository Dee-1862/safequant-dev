#!/bin/bash
#SBATCH --job-name=sq_x1_xstest
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/x1_xstest_llama_gptq4_%j.out
#SBATCH --error=scripts/logs/x1_xstest_llama_gptq4_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# X1 (reviewer item, top priority): XSTest false-refusal on Llama GPTQ-4.
# headline_eval already runs XSTest; this just runs the GPTQ-4 cell.
# Submit:  bash scripts/submit.sh scripts/headline/headline_eval_llama_gptq4_xstest.sbatch
# Output:  outputs/headline/headline_llama_gptq4_x1_n200.json   (new name; never overwrites)
#
# Needs the self-generated checkpoint  ${QM_ROOT}/gptq/llama7b-gptq4bit-g128 .
# If it is missing, build it first (sq_autogptq env, separate job):
#   MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh
# then re-submit this job once that job has finished.
# After the run, GPTQ-4 quant / +Read ASR and ARC are printed next to the paper's values
# (tab:regime: quant 26.0/50, +R 37.5/49) with a warning if any differs by > 2 points.

CKPT="${QM_ROOT}/gptq/llama7b-gptq4bit-g128"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "ERROR: GPTQ-4 checkpoint not found at ${CKPT}" >&2
  echo "Build it:  MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh   (then re-submit)" >&2
  exit 1
fi

# AutoGPTQ writes gptq_model-*.safetensors and no quantization_config in config.json;
# this links model.safetensors and adds the config so transformers/optimum can load it.
# Idempotent (only adds what is missing).
echo "=== patch AutoGPTQ checkpoint for transformers loading | $(date) ==="
python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"

echo "=== X1 headline eval Llama GPTQ-4 (XSTest) | $(date) ==="
python -u -m experiments.headline.headline_eval \
  --model llama7b --quantizer gptq4 \
  --n_eval 200 --n_reasoning 100 --n_boot 2000 \
  --tag x1_n200

echo "=== X1 vs paper (tab:regime, Llama GPTQ-4) | $(date) ==="
python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/headline/headline_llama_gptq4_x1_n200.json")
if not p.exists():
    print(f"  !! {p} not found; cannot compare"); raise SystemExit(0)
d = json.loads(p.read_text())
asr, arc = d.get("asr", {}), d.get("arc", {})
# Paper values (tab:regime, from restoration_sweep_llama_gptq4.json, n_eval 200, ARC on 50 items)
paper = {
    ("quant", "ASR"): ("gptq4", asr, 26.0), ("quant", "ARC"): ("gptq4", arc, 50.0),
    ("+R", "ASR"): ("read_restored", asr, 37.5), ("+R", "ARC"): ("read_restored", arc, 49.0),
}
warn = []
print(f"  {'cond':6s} {'metric':6s} {'X1':>6s} {'paper':>6s} {'diff':>6s}")
for (cond, metric), (key, src, ref) in paper.items():
    got = src.get(key)
    if got is None:
        print(f"  {cond:6s} {metric:6s} {'--':>6s} {ref:6.1f}   missing"); warn.append(f"{cond} {metric} missing"); continue
    diff = got - ref
    flag = "  <-- WARN >2 pts" if abs(diff) > 2.0 else ""
    print(f"  {cond:6s} {metric:6s} {got:6.1f} {ref:6.1f} {diff:+6.1f}{flag}")
    if flag:
        warn.append(f"{cond} {metric} {got:.1f} vs {ref:.1f}")
if warn:
    print("  WARNING: X1 differs from the paper by more than 2 points: " + "; ".join(warn))
    print("  Note: X1 scores ARC on 100 items (--n_reasoning 100); the paper's GPTQ-4 ARC came")
    print("  from restoration_sweep on 50 items, so ARC can move more than ASR. The checkpoint is")
    print("  also rebuilt on Delta, so check it before using X1's XSTest next to Table 2.")
else:
    print("  OK: quant and +R ASR/ARC within 2 points of the paper.")
PY
echo "=== done | $(date) ==="
