#!/bin/bash
#SBATCH --job-name=sq_smoke
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --output=scripts/logs/smoke_%j.out
#SBATCH --error=scripts/logs/smoke_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Smoke version of each reviewer job, used by `scripts/delta_queue.sh --smoke`.
#   bash scripts/submit.sh scripts/tests/smoke_reviewer.sbatch -- <step>
# step: x3 | x4 | x2 | heldout | x1 | loc_perproj | loc_depth | loc_pm | rep | x5
#
# Same python entry points and flags as the real sbatch files, but tiny:
#   --n_eval 8 --n_reasoning 8 (probes: --n_calib 8), headline XSTest capped at 8
#   (--n_xstest), and only the FIRST cell / seed / rep of each job so the whole
#   dry run fits in about an hour. Every output name gets the suffix _${SMOKE_TAG}
#   (default _smoke), so smoke files can never collide with the real runs. To rerun
#   a smoke test after its files exist, use a new SMOKE_TAG (e.g. smoke2).

STEP="${1:?usage: smoke_reviewer.sbatch <x3|x4|x2|heldout|x1|loc_perproj|loc_depth|loc_pm|rep|x5>}"
T="${SMOKE_TAG:-smoke}"
N="--n_eval 8 --n_reasoning 8"
HL="python -u -m experiments.headline.headline_eval"

echo "=== SMOKE ${STEP} (tag suffix _${T}) | $(date) ==="
case "${STEP}" in
  x3)
    python -u -m experiments.signal_geometry.probe_accuracy \
      --model llama7b --quantizer aqlm2 --n_calib 8 --probe_type linear \
      --shuffle_labels --tag "${T}"
    ;;
  x4)
    python -u -m experiments.boundary.probe_refit_control \
      --model llama7b --n_calib 8 --n_splits 100 --tag "${T}"
    ;;
  x2)
    python -u -m experiments.boundary.probe_leakage_control \
      --model llama7b --quantizer aqlm2 --n_calib 8 --n_transfer 8 --tag "${T}"
    ;;
  heldout)
    ${HL} --model llama3-8b --quantizer aqlm2 ${N} --n_boot 200 --n_xstest 8 \
      --tag "heldout_${T}"
    python -u -m experiments.regime_sweep.paired_screen \
      --cells llama3-8b:aqlm2 --variant "heldout_${T}"
    ;;
  x1)
    CKPT="${QM_ROOT}/gptq/llama7b-gptq4bit-g128"
    if [[ ! -f "${CKPT}/config.json" ]]; then
      echo "SKIP x1 smoke: GPTQ-4 checkpoint not built yet (${CKPT})." >&2
      echo "  Build it first: bash scripts/delta_queue.sh --gptq-only" >&2
      exit 1
    fi
    # Same idempotent AutoGPTQ -> transformers patch the real X1 job runs
    python -u scripts/fix_autogptq_checkpoints.py "${CKPT}"
    ${HL} --model llama7b --quantizer gptq4 ${N} --n_boot 200 --n_xstest 8 \
      --tag "x1_n200_${T}"
    ;;
  loc_perproj)
    python -u -m experiments.localization.per_projection \
      --model llama7b --quantizer aqlm2 ${N} --layer_lo 11 --layer_hi 22 --tag "flags_${T}"
    ;;
  loc_depth)
    python -u -m experiments.localization.depth_bands \
      --model llama7b --quantizer aqlm2 --projections read ${N} --tag "flags_${T}"
    ;;
  loc_pm)
    # --seeds 2 (not 5) keeps the smoke job under an hour; same code path.
    python -u -m experiments.regime_sweep.param_matched_restore \
      --model llama7b --quantizer aqlm2 ${N} --seeds 2 --tag "flags_${T}"
    ;;
  rep)
    ${HL} --model llama7b --quantizer aqlm2 ${N} --n_boot 200 --n_xstest 8 \
      --tag "rep1_${T}"
    ;;
  x5)
    ${HL} --model llama7b --quantizer aqlm2 ${N} --n_boot 200 --n_xstest 8 \
      --subset_seed 1 --tag "${T}"
    ;;
  *)
    echo "unknown step: ${STEP}" >&2; exit 2 ;;
esac
echo "=== SMOKE ${STEP} done | $(date) ==="
