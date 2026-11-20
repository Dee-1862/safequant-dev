#!/bin/bash
# Submit the whole reviewer-run queue on a Slurm cluster with job dependencies.
#
#   bash scripts/queue.sh                # full runs
#   bash scripts/queue.sh --dry-run      # print the sbatch commands, submit nothing
#   bash scripts/queue.sh --gptq-only    # submit only the GPTQ-4 build (llama7b)
#   bash scripts/queue.sh --gptq-all     # full queue, rebuilding every GPTQ cell
#   bash scripts/queue.sh --gptq-only --gptq-all   # only the GPTQ build + PPL jobs
#   bash scripts/queue.sh --awq-ppl      # full queue + AWQ PPL checks gating X2's AWQ cells
#   bash scripts/queue.sh --wave1        # X3, X4, X2 non-GPTQ cells, held-out, Tables 5-7
#   bash scripts/queue.sh --limits       # print expected runtime vs time limit per job
#   options: --skip-setup  (checkpoints already downloaded)
#            --skip-gptq   (GPTQ-4 checkpoint already built; auto-detected too)
#
# Run order (README): envs -> setup -> --gptq-only ->
#   bash scripts/queue.sh --skip-setup --skip-gptq
#
# Graph (full):   setup ──┬─> X3, X4, X2, held-out, Task 3 (x3 jobs), Task 4
#                               └─> GPTQ-4 build ──> X1
# Submission order (priority = order):
#   X3, X4, X2, held-out, [GPTQ-4 build], X1, Task 3, Task 4
# Headline-type jobs are split one per cell/seed/rep (their sbatch files take CELLS /
# SEEDS / REPS) with a per-job time limit (EST table), because 10 or 6 headline evals do not fit in a
# single job. Every job gets --kill-on-invalid-dep=yes: if setup fails, dependants are
# cancelled instead of pending forever.
#
# --gptq-only: submits MODELS=llama7b BITS=4 quantize_gptq_all.sh and nothing else (no
# dependency; run it after setup has finished). Exits without submitting if the
# checkpoint already exists.
#
# --gptq-all: instead of the single llama7b GPTQ-4 build, submits one build job per
# MISSING checkpoint among llama7b {4,3}, qwen7b {4,3}, mistral7b-v02 {4}, phi3 {4,3}
# (GPTQ_CELLS="phi3:4 llama7b:4" narrows the list), each depending on setup with
# a per-job time limit (EST table), and after each build a PPL job (afterok) that runs
# experiments/regime_sweep/wikitext_ppl on that cell under the tag rebuilt
# (PPL_TAG overrides) and prints it next to the paper value, warning above 2%.
# Each model's gptq4 PPL job also measures fp32 (environment control).
# GPTQ_BACKEND=autogptq (default, $AGENV) | gptqmodel ($CONDA_ENV) picks the builder.
# Builds get --cpus-per-task=${GPTQ_CPUS:-8} (auto-gptq packs on the CPU).
# X1 then depends on the llama7b gptq4 build + PPL jobs, and X2 is split in two: the
# non-GPTQ cells (depend on setup only) and the GPTQ cells (depend on every build + PPL).
# With --gptq-only, only the build + PPL jobs are submitted.
#
# --awq-ppl: one WikiText-2 PPL job per AWQ cell (llama7b, qwen7b, mistral7b-v02, phi3;
# public checkpoints) after setup, tag rerun (AWQ_PPL_TAG overrides), printed next to
# the paper value with a warning above 2%; each also measures fp32 under that tag.
# Reason: on aarch64 autoawq runs its Triton kernels (no awq_ext wheels); the original
# x86 runs used awq_ext. X2's AWQ cells then run as their own job after these four. Combine with
# --gptq-only --gptq-all to submit only the GPTQ build/PPL and AWQ PPL jobs.
#
# Time limits: every job gets --time = 1.5 x its expected runtime (rounded up to 15 min,
# capped at 48 h) from the EST table below, so short jobs can backfill. TIME_SCALE=<x>
# multiplies every estimate (e.g. TIME_SCALE=1.5 on slower nodes). Estimates are guesses
# unless the table says "measured"; check them against the first real runs. --limits prints the table and exits.
#
# --wave1: only X3, X4, X2's non-GPTQ cells (AQLM + AWQ; AWQ waits for its PPL checks
# with --awq-ppl), held-out, the Tables 5-7 reruns (Task 3 localization), and X1 when
# the llama7b gptq4 checkpoint already exists. No GPTQ builds, X2 GPTQ cells or X6.
# Every queue job gets --cpus-per-task=${JOB_CPUS:-8} (threads sized in _env.sh).
#
# Needs .env: SLURM_ACCOUNT + SLURM_PARTITION (GPU jobs), QM_ROOT, HF_HOME, AGENV (GPTQ
# build). Optional: SLURM_CPU_PARTITION + SLURM_CPU_ACCOUNT -> the setup job runs on the
# CPU partition under the CPU account (for clusters where CPU and GPU allocations are
# separate accounts). Setting the partition without the account is an error; with neither, setup
# runs on the GPU partition under SLURM_ACCOUNT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/load_env.sh

DRY=0; SKIP_SETUP=0; SKIP_GPTQ=0; GPTQ_ONLY=0; GPTQ_ALL=0; AWQ_PPL=0
WAVE1=0; LIMITS=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --skip-setup) SKIP_SETUP=1 ;;
    --skip-gptq) SKIP_GPTQ=1 ;;
    --gptq-all) GPTQ_ALL=1 ;;
    --awq-ppl) AWQ_PPL=1 ;;
    --gptq-only) GPTQ_ONLY=1 ;;
    --wave1) WAVE1=1 ;;
    --limits) LIMITS=1 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/{/^set -euo/!p}' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
if [[ $WAVE1 -eq 1 && ( $GPTQ_ONLY -eq 1 || $GPTQ_ALL -eq 1 ) ]]; then
  echo "ERROR: --wave1 is a full-queue subset; not with --gptq-only or --gptq-all." >&2; exit 2
fi
if [[ $AWQ_PPL -eq 1 && $GPTQ_ONLY -eq 1 && $GPTQ_ALL -eq 0 ]]; then
  echo "ERROR: --awq-ppl with --gptq-only needs --gptq-all too." >&2; exit 2
fi
if [[ $GPTQ_ALL -eq 1 && $SKIP_GPTQ -eq 1 ]]; then
  echo "ERROR: --gptq-all cannot be combined with --skip-gptq." >&2; exit 2
fi
GPTQ_BACKEND="${GPTQ_BACKEND:-autogptq}"
case "$GPTQ_BACKEND" in
  autogptq)  GPTQ_ENV="${AGENV:-}" ;;
  gptqmodel) GPTQ_ENV="${CONDA_ENV:-}" ;;
  *) echo "ERROR: GPTQ_BACKEND must be autogptq or gptqmodel, got '$GPTQ_BACKEND'" >&2; exit 2 ;;
esac
if [[ $GPTQ_ALL -eq 1 && -z "$GPTQ_ENV" ]]; then
  echo "ERROR: GPTQ_BACKEND=$GPTQ_BACKEND needs $([[ $GPTQ_BACKEND == autogptq ]] && echo AGENV || echo CONDA_ENV) in .env." >&2; exit 1
fi
# --gptq-only --gptq-all: build + PPL jobs only; they still wait for setup unless skipped
if [[ $GPTQ_ONLY -eq 1 && $GPTQ_ALL -eq 0 ]]; then SKIP_SETUP=1; fi
if [[ $DRY -eq 0 && -z "${SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set SLURM_ACCOUNT in .env (see .env.example)." >&2; exit 1
fi
# Checked even in --dry-run, before anything is submitted.
if [[ $SKIP_SETUP -eq 0 && -n "${SLURM_CPU_PARTITION:-}" && -z "${SLURM_CPU_ACCOUNT:-}" ]]; then
  echo "ERROR: SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION} is set but SLURM_CPU_ACCOUNT is not." >&2
  echo "       A separate CPU partition needs its CPU allocation account." >&2
  echo "       Set SLURM_CPU_ACCOUNT in .env, or unset" >&2
  echo "       SLURM_CPU_PARTITION to run setup on the GPU partition. Nothing was submitted." >&2
  exit 1
fi

# --- expected runtimes (minutes) -> time limits ---------------------------------------
# "measured" = seen on the reviewer-run nodes; "job-file note" = comment in the job file
# (original H100 runs);
# everything else is an estimate from the workload. Unit is per job unless noted.
TIME_SCALE="${TIME_SCALE:-1}"
declare -A EST=(
  [setup]=30        [gptq_phi3]=25   [gptq_7b]=45    [ppl]=25
  [x3]=120          [x4]=150         [x2_cell]=75
  [heldout]=510     [x1]=450
  [loc_perproj]=360 [loc_depth]=360  [loc_pm]=480
  [x6]=450
)
declare -A EST_NOTE=(
  [setup]="measured 28m44s"
  [gptq_phi3]="measured ~12 min quantize + pack/save with thread fix"
  [gptq_7b]="estimate: ~2x phi3 weights"
  [ppl]="estimate: fp32 + one quant cell, WikiText-2 test, 2048-token windows"
  [x3]="estimate: llama7b + phi3, one load + residuals + probes each"
  [x4]="estimate: 2 FP extractions + 100 probe refits (CPU)"
  [x2_cell]="job-file note: ~1-1.5 h per cell"
  [heldout]="job-file note: headline ~7-8 h, + paired_screen ~1 h"
  [x1]="job-file note: headline ~7-8 h"
  [loc_perproj]="estimate: per_projection, 2 models (24 h cap covered all 3 steps)"
  [loc_depth]="estimate: depth_bands read+write, 2 models"
  [loc_pm]="estimate: param_matched, 5 seeds x 2 models"
  [x6]="job-file note: ~7-8 h per run"
)
limit_min() {  # limit_min <expected minutes>: 1.5 x scale, up to 15 min, <= 48 h
  awk -v m="$1" -v s="$TIME_SCALE" 'BEGIN { x = m * 1.5 * s; q = int((x + 14.999) / 15) * 15; if (q > 2880) q = 2880; print q }'
}
hhmm() { printf '%d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 )); }
tl() { echo "--time=$(hhmm "$(limit_min "$1")")"; }   # tl <expected minutes>
print_limits() {
  printf '%-14s %10s %10s  %s\n' "job" "expected" "limit" "basis   (TIME_SCALE=${TIME_SCALE})"
  local k
  for k in setup gptq_phi3 gptq_7b ppl x3 x4 x2_cell heldout x1 loc_perproj loc_depth loc_pm x6; do
    printf '%-14s %10s %10s  %s\n' "$k" "$(hhmm "${EST[$k]}")" "$(hhmm "$(limit_min "${EST[$k]}")")" "${EST_NOTE[$k]}"
  done
  printf '%-14s %10s %10s  %s\n' "x2 (n cells)" "n x 1:15" "1.5 x n x" "e.g. 3 cells -> $(hhmm "$(limit_min $((3 * EST[x2_cell])))"), 4 -> $(hhmm "$(limit_min $((4 * EST[x2_cell])))"), 7 -> $(hhmm "$(limit_min $((7 * EST[x2_cell])))"), 14 -> $(hhmm "$(limit_min $((14 * EST[x2_cell])))")"
}
x2_time() { local n; n=$(echo $1 | wc -w); tl $(( n * EST[x2_cell] )); }   # x2_time "<cells>"
if [[ $LIMITS -eq 1 ]]; then print_limits; exit 0; fi

LIST=()     # "<id>  <label>  <log>" rows for the summary
DRY_N=0
GPTQ_CKPT="${QM_ROOT:-$HOME/quantized_models}/gptq/llama7b-gptq4bit-g128"

# --- --gptq-only: just the GPTQ-4 build, then stop ----------------------------------
if [[ $GPTQ_ONLY -eq 1 && $GPTQ_ALL -eq 0 ]]; then
  if [[ -f "${GPTQ_CKPT}/config.json" ]]; then
    echo "GPTQ-4 checkpoint already present: ${GPTQ_CKPT} (nothing submitted)"
    exit 0
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "[dry-run] MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh" >&2
    GID="DRY1"
  else
    GOUT=$(MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh)
    echo "$GOUT" >&2
    GID=$(echo "$GOUT" | sed -n 's/^JOBIDS: //p' | awk '{print $1}')
  fi
  echo ""
  echo "job id     job                                log"
  printf '%-10s %-34s %s\n' "$GID" "GPTQ-4 build llama7b" "scripts/logs/quantize_gptq_${GID}.out"
  echo "output -> ${GPTQ_CKPT}"
  exit 0
fi

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
  if [[ " $* " != *" --cpus-per-task="* ]]; then set -- --cpus-per-task="${JOB_CPUS:-8}" "$@"; fi
  if [[ $DRY -eq 1 ]]; then
    DRY_N=$((DRY_N + 1)); id="DRY${DRY_N}"
    echo "[dry-run] ${SUB_ENV:-}bash scripts/submit.sh ${job} --parsable --kill-on-invalid-dep=yes $([[ $# -gt 0 ]] && printf '%q ' "$@")" >&2
  else
    id=$(env ${SUB_ENV:-} bash scripts/submit.sh "$job" --parsable --kill-on-invalid-dep=yes "$@")
    id="${id%%;*}"
  fi
  LIST+=("$(printf '%-10s %-34s %s' "$id" "$label" "$(_log "$job" "$id" "$out_override")")")
  SUB_ID="$id"
}

dep() { [[ -n "${1:-}" ]] && echo "--dependency=afterok:$1" || true; }
# depall <id>... : one afterok over every non-empty id (empty -> nothing)
depall() {
  local ids=() x
  for x in "$@"; do [[ -n "$x" ]] && ids+=("$x"); done
  [[ ${#ids[@]} -gt 0 ]] && echo "--dependency=afterok:$(IFS=:; echo "${ids[*]}")" || true
}

# --- --awq-ppl: one PPL check per AWQ cell (public checkpoints, current stack) ------
AWQGATE=""
awq_ppl_submit() {  # awq_ppl_submit <dependency id or empty>
  local up="$1" m
  for m in llama7b qwen7b mistral7b-v02 phi3; do
    sub "AWQ PPL ${m} vs paper" scripts/regime_sweep/wikitext_ppl_rebuilt.sbatch $(dep "$up") \
      "$(tl "${EST[ppl]}")" --job-name="sq_ppl_${m}_awq" \
      --export="ALL,EXP_MODEL=${m},EXP_QUANT=awq,PPL_TAG=${AWQ_PPL_TAG:-rerun},WTPPL_FP32=1"
    AWQGATE="${AWQGATE:+${AWQGATE} }${SUB_ID}"
  done
}

# --- --gptq-all: one build + one PPL check per missing GPTQ checkpoint --------------
# Fills GATE["model:bits"]="<build id> <ppl id>" for the cells it submits.
declare -A GATE=()
GPTQ_ALL_CELLS="${GPTQ_CELLS:-llama7b:4 llama7b:3 qwen7b:4 qwen7b:3 mistral7b-v02:4 phi3:4 phi3:3}"
gptq_all_submit() {  # gptq_all_submit <dependency id or empty>
  local up="$1" cell m b ck bid
  for cell in $GPTQ_ALL_CELLS; do
    m="${cell%%:*}"; b="${cell##*:}"
    ck="${QM_ROOT:-$HOME/quantized_models}/gptq/${m}-gptq${b}bit-g128"
    if [[ -f "${ck}/config.json" ]]; then
      echo "GPTQ ${m} gptq${b}: checkpoint present, not rebuilding (${ck})" >&2
      continue
    fi
    sub "GPTQ build ${m} gptq${b} [${GPTQ_BACKEND}]" scripts/quantize_gptq.sbatch $(dep "$up") \
      "$(tl "$([[ $m == phi3 ]] && echo "${EST[gptq_phi3]}" || echo "${EST[gptq_7b]}")")" \
      --cpus-per-task="${GPTQ_CPUS:-8}" --job-name="sq_gptq_${m}_${b}" \
      --export="ALL,CONDA_ENV=${GPTQ_ENV},GPTQ_BACKEND=${GPTQ_BACKEND}" -- "$m" "$b" 128
    bid="$SUB_ID"
    sub "PPL check ${m} gptq${b} vs paper" scripts/regime_sweep/wikitext_ppl_rebuilt.sbatch \
      "$(dep "$bid")" "$(tl "${EST[ppl]}")" --job-name="sq_ppl_${m}_${b}" \
      --export="ALL,EXP_MODEL=${m},EXP_QUANT=gptq${b},PPL_TAG=${PPL_TAG:-rebuilt},WTPPL_FP32=$([[ $b == 4 ]] && echo 1 || echo 0)"
    GATE["${m}:${b}"]="${bid} ${SUB_ID}"
  done
}

# --- 1. setup -------------------------------------------------------------------
SETUP=""
if [[ $SKIP_SETUP -eq 0 ]]; then
  if [[ -n "${SLURM_CPU_PARTITION:-}" ]]; then
    SUB_ENV="SLURM_ACCOUNT=${SLURM_CPU_ACCOUNT} SLURM_PARTITION=${SLURM_CPU_PARTITION} SQ_GPUS_PER_NODE=0 " \
      sub "setup (cpu)" scripts/setup.sbatch "$(tl "${EST[setup]}")"
  else
    sub "setup (gpu partition)" scripts/setup.sbatch "$(tl "${EST[setup]}")"
  fi
  SETUP="$SUB_ID"
fi
D="$(dep "$SETUP")"

if [[ $GPTQ_ONLY -eq 1 && $GPTQ_ALL -eq 1 ]]; then
  gptq_all_submit "$SETUP"
  if [[ $AWQ_PPL -eq 1 ]]; then awq_ppl_submit "$SETUP"; fi
  echo ""
  echo "job id     job                                log"
  [[ ${#LIST[@]} -gt 0 ]] && printf '%s\n' "${LIST[@]}" || echo "(nothing to submit: every checkpoint is present)"
  echo ""
  echo "watch: squeue -u \$USER"
  exit 0
fi


# --- 2. full queue in priority order ------------------------------------------
sub "X3 shuffled-label probe"    scripts/signal_geometry/probe_accuracy_shuffled_llama.sbatch $D "$(tl "${EST[x3]}")"
sub "X4 refit floor, 100 splits" scripts/boundary/probe_refit_control_splits100_llama.sbatch $D "$(tl "${EST[x4]}")"
X2_GPTQ_CELLS="llama7b:gptq4 llama7b:gptq3 qwen7b:gptq4 qwen7b:gptq3 mistral7b-v02:gptq4 phi3:gptq4 phi3:gptq3"
X2_AWQ_CELLS="llama7b:awq qwen7b:awq mistral7b-v02:awq phi3:awq"
X2_AQLM_CELLS="llama7b:aqlm2 mistral7b-v02:aqlm2 phi3:aqlm2"
if [[ $GPTQ_ALL -eq 1 || $AWQ_PPL -eq 1 || $WAVE1 -eq 1 ]]; then
  # split X2: cells without a gate run now; AWQ cells after the AWQ PPL checks and
  # GPTQ cells after their rebuild + PPL (both below). --wave1 leaves GPTQ cells out.
  X2_NOW="${X2_AQLM_CELLS}"
  [[ $AWQ_PPL -eq 1 ]] || X2_NOW="${X2_NOW} ${X2_AWQ_CELLS}"
  [[ $GPTQ_ALL -eq 1 || $WAVE1 -eq 1 ]] || X2_NOW="${X2_NOW} ${X2_GPTQ_CELLS}"
  sub "X2 leakage, $(echo ${X2_NOW} | wc -w) ungated cells" scripts/boundary/probe_leakage_control_quant.sbatch $D \
    "$(x2_time "${X2_NOW}")" "--export=ALL,CELLS=${X2_NOW}"
else
  sub "X2 leakage, 14 quant cells" scripts/boundary/probe_leakage_control_quant.sbatch $D \
    "$(x2_time "${X2_AQLM_CELLS} ${X2_AWQ_CELLS} ${X2_GPTQ_CELLS}")"
fi
sub "held-out llama3-8b aqlm2"   scripts/headline/headline_eval_llama3_8b_aqlm2_heldout.sbatch $D "$(tl "${EST[heldout]}")"

if [[ $AWQ_PPL -eq 1 ]]; then
  awq_ppl_submit "$SETUP"
  # shellcheck disable=SC2086
  sub "X2 leakage, 4 AWQ cells"    scripts/boundary/probe_leakage_control_quant.sbatch \
    "$(depall "$SETUP" ${AWQGATE})" "$(x2_time "${X2_AWQ_CELLS}")" "--export=ALL,CELLS=${X2_AWQ_CELLS}"
fi

if [[ $WAVE1 -eq 0 ]]; then

# GPTQ-4 build (skipped if the checkpoint already exists), then X1 after it
GPTQ=""
CKPT="${QM_ROOT:-$HOME/quantized_models}/gptq/llama7b-gptq4bit-g128"
if [[ $GPTQ_ALL -eq 1 ]]; then
  gptq_all_submit "$SETUP"
elif [[ $SKIP_GPTQ -eq 0 && ! -f "${CKPT}/config.json" ]]; then
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
  if [[ ! -f "${CKPT}/config.json" ]]; then
    echo "WARNING: ${CKPT} does not exist yet and X1 gets no dependency on a build." >&2
    echo "         If a --gptq-only job is still running, X1 will fail its checkpoint check;" >&2
    echo "         wait for it, or re-submit X1 after it finishes." >&2
  fi
fi
if [[ $GPTQ_ALL -eq 1 ]]; then
  # shellcheck disable=SC2086
  X1DEP="$(depall "$SETUP" ${GATE["llama7b:4"]:-})"
else
  X1DEP="$(dep "${GPTQ:-$SETUP}")"
fi
sub "X1 XSTest llama gptq4"      scripts/headline/headline_eval_llama_gptq4_xstest.sbatch $X1DEP "$(tl "${EST[x1]}")"

if [[ $GPTQ_ALL -eq 1 ]]; then
  # shellcheck disable=SC2086
  X2DEP="$(depall "$SETUP" ${GATE[*]:-})"
  sub "X2 leakage, 7 GPTQ cells"   scripts/boundary/probe_leakage_control_quant.sbatch $X2DEP \
    "$(x2_time "${X2_GPTQ_CELLS}")" "--export=ALL,CELLS=${X2_GPTQ_CELLS}"
fi
fi   # WAVE1 -eq 0 (GPTQ builds, X1, X2 GPTQ cells)

if [[ $WAVE1 -eq 1 ]]; then
  X1CK="${QM_ROOT:-$HOME/quantized_models}/gptq/llama7b-gptq4bit-g128"
  if [[ -f "${X1CK}/config.json" ]]; then
    sub "X1 XSTest llama gptq4"    scripts/headline/headline_eval_llama_gptq4_xstest.sbatch $D "$(tl "${EST[x1]}")"
  else
    echo "WARNING: --wave1 skips X1: ${X1CK} not built" >&2
  fi
fi

for step in perproj depth pm; do
  sub "Task 3 localization ${step}" scripts/localization/localization_with_flags.sbatch $D \
    "$(tl "${EST[loc_${step}]}")" --export="ALL,STEPS=${step}"
done

if [[ $WAVE1 -eq 0 ]]; then
for m in llama7b phi3; do
  for rep in rep1 rep2 rep3; do
    sub "Task 4 X6 ${m} ${rep}" scripts/headline/headline_repeats_aqlm2.sbatch $D "$(tl "${EST[x6]}")" \
      --export="ALL,CELLS=${m},REPS=${rep}" --nice="${REPEAT_NICE:-500}"
  done
done
fi

echo ""
echo "job id     job                                log"
printf '%s\n' "${LIST[@]}"
echo ""
echo "watch: squeue -u \$USER    cancel all: scancel -u \$USER"
exit 0
