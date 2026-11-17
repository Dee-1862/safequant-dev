#!/bin/bash
# Submit the whole reviewer-run queue on Delta with Slurm dependencies.
#
#   bash scripts/delta_queue.sh                # full runs
#   bash scripts/delta_queue.sh --smoke        # tiny dry run of every job (~1 h)
#   bash scripts/delta_queue.sh --dry-run      # print the sbatch commands, submit nothing
#   options: --skip-setup  (checkpoints already downloaded)
#            --skip-gptq   (GPTQ-4 checkpoint already built; auto-detected too)
#
# Graph (full):   delta_setup ──┬─> X3, X4, X2, held-out, Task 3 (x3 jobs), Task 4, X5
#                               └─> GPTQ-4 build ──> X1
# Submission order (the queue may not drain before Nov 20, so priority = order):
#   X3, X4, X2, held-out, [GPTQ-4 build], X1, Task 3, Task 4, X5
# Headline-type jobs are split one per cell/seed/rep (their sbatch files take CELLS /
# SEEDS / REPS) with --time=12:00:00, because 10 or 4 headline evals do not fit in a
# single job. Every job gets --kill-on-invalid-dep=yes: if setup fails, dependants are
# cancelled instead of pending forever.
#
# --smoke: skips setup and the GPTQ build (run the smoke test after setup has
# finished once), submits scripts/tests/smoke_reviewer.sbatch per step with
# --n_eval 8 --n_reasoning 8 and the tag suffix _smoke (SMOKE_TAG overrides), first
# cell/seed only. X1's smoke step fails fast if the GPTQ-4 checkpoint is missing.
#
# Needs .env: SLURM_ACCOUNT + SLURM_PARTITION (GPU jobs), QM_ROOT, HF_HOME, AGENV (GPTQ
# build). Optional: SLURM_CPU_PARTITION + SLURM_CPU_ACCOUNT -> the setup job runs on the
# CPU partition under the CPU account (Delta CPU and GPU allocations are separate
# accounts). Setting the partition without the account is an error; with neither, setup
# runs on the GPU partition under SLURM_ACCOUNT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/load_env.sh

SMOKE=0; DRY=0; SKIP_SETUP=0; SKIP_GPTQ=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --skip-setup) SKIP_SETUP=1 ;;
    --skip-gptq) SKIP_GPTQ=1 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
if [[ $SMOKE -eq 1 ]]; then SKIP_SETUP=1; SKIP_GPTQ=1; fi
if [[ $DRY -eq 0 && -z "${SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set SLURM_ACCOUNT in .env (see .env.example)." >&2; exit 1
fi
# Checked even in --dry-run, before anything is submitted.
if [[ $SKIP_SETUP -eq 0 && -n "${SLURM_CPU_PARTITION:-}" && -z "${SLURM_CPU_ACCOUNT:-}" ]]; then
  echo "ERROR: SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION} is set but SLURM_CPU_ACCOUNT is not." >&2
  echo "       Delta's CPU partitions need the CPU allocation account (see 'accounts';" >&2
  echo "       usually ends in -delta-cpu). Set SLURM_CPU_ACCOUNT in .env, or unset" >&2
  echo "       SLURM_CPU_PARTITION to run setup on the GPU partition. Nothing was submitted." >&2
  exit 1
fi

LIST=()     # "<id>  <label>  <log>" rows for the summary
DRY_N=0

# log path from the job file's --output line (or an --output override), %j -> id
_log() {  # _log <job.sbatch> <id> [override]
  local pat="${3:-$(sed -n 's/^#SBATCH[[:space:]]\+--output=//p' "$1" | head -1)}"
  echo "${pat//%j/$2}"
}

# sub <label> <job.sbatch> [sbatch args...] [-- job args]; sets SUB_ID. Called directly,
# never in $(...), so the LIST / DRY_N updates survive.
sub() {
  local label="$1" job="$2"; shift 2
  local out_override="" a
  for a in "$@"; do [[ "$a" == --output=* ]] && out_override="${a#--output=}"; done
  local id
  if [[ $DRY -eq 1 ]]; then
    DRY_N=$((DRY_N + 1)); id="DRY${DRY_N}"
    echo "[dry-run] ${SUB_ENV:-}bash scripts/submit.sh ${job} --parsable --kill-on-invalid-dep=yes $*" >&2
  else
    id=$(env ${SUB_ENV:-} bash scripts/submit.sh "$job" --parsable --kill-on-invalid-dep=yes "$@")
    id="${id%%;*}"
  fi
  LIST+=("$(printf '%-10s %-34s %s' "$id" "$label" "$(_log "$job" "$id" "$out_override")")")
  SUB_ID="$id"
}

dep() { [[ -n "${1:-}" ]] && echo "--dependency=afterok:$1" || true; }

# --- 1. setup -------------------------------------------------------------------
SETUP=""
if [[ $SKIP_SETUP -eq 0 ]]; then
  if [[ -n "${SLURM_CPU_PARTITION:-}" ]]; then
    SUB_ENV="SLURM_ACCOUNT=${SLURM_CPU_ACCOUNT} SLURM_PARTITION=${SLURM_CPU_PARTITION} SQ_GPUS_PER_NODE=0 " \
      sub "delta_setup (cpu)" scripts/tests/delta_setup.sbatch
  else
    sub "delta_setup (gpu partition)" scripts/tests/delta_setup.sbatch
  fi
  SETUP="$SUB_ID"
fi
D="$(dep "$SETUP")"

H12="--time=12:00:00"

if [[ $SMOKE -eq 1 ]]; then
  # --- smoke: one tiny job per step, same order, no dependencies --------------
  SM=scripts/tests/smoke_reviewer.sbatch
  for step in x3 x4 x2 heldout x1 loc_perproj loc_depth loc_pm rep x5; do
    sub "smoke ${step}" "$SM" --job-name="sq_smoke_${step}" \
      --output="scripts/logs/smoke_${step}_%j.out" --error="scripts/logs/smoke_${step}_%j.err" \
      --export="ALL,SMOKE_TAG=${SMOKE_TAG:-smoke}" -- "$step"
  done
else
  # --- 2. full queue in priority order ------------------------------------------
  sub "X3 shuffled-label probe"    scripts/signal_geometry/probe_accuracy_shuffled_llama.sbatch $D
  sub "X4 refit floor, 100 splits" scripts/boundary/probe_refit_control_splits100_llama.sbatch $D
  sub "X2 leakage, 14 quant cells" scripts/boundary/probe_leakage_control_quant.sbatch $D
  sub "held-out llama3-8b aqlm2"   scripts/headline/headline_eval_llama3_8b_aqlm2_heldout.sbatch $D $H12

  # GPTQ-4 build (skipped if the checkpoint already exists), then X1 after it
  GPTQ=""
  CKPT="${QM_ROOT:-$HOME/quantized_models}/gptq/llama7b-gptq4bit-g128"
  if [[ $SKIP_GPTQ -eq 0 && ! -f "${CKPT}/config.json" ]]; then
    if [[ $DRY -eq 1 ]]; then
      echo "[dry-run] DEPENDENCY=${SETUP:+afterok:$SETUP} MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh" >&2
      DRY_N=$((DRY_N + 1)); GPTQ="DRY${DRY_N}"
    else
      GOUT=$(DEPENDENCY="${SETUP:+afterok:$SETUP}" MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh)
      echo "$GOUT" >&2
      GPTQ=$(echo "$GOUT" | sed -n 's/^JOBIDS: //p' | awk '{print $1}')
    fi
    LIST+=("$(printf '%-10s %-34s %s' "$GPTQ" "GPTQ-4 build llama7b" "scripts/logs/quantize_gptq_${GPTQ}.out")")
  else
    echo "GPTQ-4 checkpoint present or --skip-gptq: not building (${CKPT})" >&2
  fi
  X1DEP="$(dep "${GPTQ:-$SETUP}")"
  sub "X1 XSTest llama gptq4"      scripts/headline/headline_eval_llama_gptq4_xstest.sbatch $X1DEP $H12

  for step in perproj depth pm; do
    sub "Task 3 localization ${step}" scripts/localization/localization_with_flags.sbatch $D \
      --export="ALL,STEPS=${step}"
  done

  for m in llama7b phi3; do
    for rep in rep2 rep3; do
      sub "Task 4 X6 ${m} ${rep}" scripts/headline/headline_repeats_aqlm2.sbatch $D $H12 \
        --export="ALL,CELLS=${m},REPS=${rep}"
    done
  done

  for m in llama7b phi3; do
    for k in 1 2 3 4 5; do
      sub "X5 ${m} subset${k}" scripts/regime_sweep/subset_seeds_aqlm2.sbatch $D $H12 \
        --export="ALL,CELLS=${m},SEEDS=${k}"
    done
  done
fi

echo ""
echo "job id     job                                log"
printf '%s\n' "${LIST[@]}"
echo ""
echo "watch: squeue -u \$USER    cancel all: scancel -u \$USER"
[[ $SMOKE -eq 1 ]] && echo "smoke outputs carry the suffix _${SMOKE_TAG:-smoke}; they are never read by reviewer_runs.py."
exit 0
