#!/bin/bash
#SBATCH --job-name=sq_probe_pool
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --time=4:30:00
#SBATCH --mem=64G
#SBATCH --output=scripts/logs/probe_pool_%j.out
#SBATCH --error=scripts/logs/probe_pool_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Run probe-only pieces (X2 cells, X3, X4; see scripts/probe_pieces.sh) from POOL on ONE GPU,
# up to MAX_PAR at a time as separate processes. Generation jobs never go through this.
#   single:  POOL="x4:phi3"  MAX_PAR=1   --mem=64G --cpus-per-task=8
#   packed:  POOL="<list>"   MAX_PAR=4   --mem=64G --cpus-per-task=32
# POOL order is priority order. A piece starts only if
#   (a) nobody claimed it and it is ready (GPTQ cells: dense copy exists),
#   (b) for X3 pairs: no other job is computing that family's FP32 pair right now,
#   (c) GPU: memory estimates of running pieces + this one <= GPU_BUDGET_GB (88 of ~95 GB)
#       and nvidia-smi shows that much free,
#   (d) host: no model load in progress (SQ_LOAD_LOCK: one load at a time per job) and the
#       job's cgroup has its load peak + 4 GB free (anonymous memory vs memory.max),
#   (e) the time left covers its expected minutes x 1.5 when sharing the GPU (x 1.3 alone).
# Pieces start >= 90 s apart; GPU memory is logged every 60 s.
set -uo pipefail
source scripts/probe_pieces.sh
POOL="${POOL:-$PROBE_ALL}"
MAX_PAR="${MAX_PAR:-1}"
GPU_BUDGET_GB="${GPU_BUDGET_GB:-88}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / MAX_PAR )); (( THREADS < 1 )) && THREADS=1
END=$(date -d "$(squeue -h -j "${SLURM_JOB_ID}" -o %e)" +%s 2>/dev/null || echo $(( $(date +%s) + 3600 )))
export SQ_LOAD_LOCK="${TMPDIR:-/tmp}/sq_load_lock_${SLURM_JOB_ID}"
: > "$SQ_LOAD_LOCK"
echo "=== probe pool | job ${SLURM_JOB_ID} | MAX_PAR=${MAX_PAR} threads/proc=${THREADS} gpu budget=${GPU_BUDGET_GB}G | SQ_GPTQ_DENSE=${SQ_GPTQ_DENSE:-<unset>} | ends $(date -d @"$END") ==="
echo "pool: ${POOL}"

# job cgroup (cgroup v2): walk up from this process to the first dir with a memory limit
CG=""
_p="/sys/fs/cgroup$(cut -d: -f3- /proc/self/cgroup | head -1)"
while [[ "$_p" != /sys/fs/cgroup && -n "$_p" ]]; do
  if [[ -f "$_p/memory.max" && "$(cat "$_p/memory.max")" != max ]]; then CG="$_p"; break; fi
  _p="${_p%/*}"
done
host_free_gb() {  # memory.max - anonymous memory, GB (page cache is reclaimable, not counted)
  [[ -n "$CG" ]] || { echo 999; return; }
  local max anon
  max=$(cat "$CG/memory.max"); anon=$(awk '$1=="anon"{print $2}' "$CG/memory.stat")
  echo $(( (max - ${anon:-0}) / 1073741824 ))
}
echo "host cgroup: ${CG:-<not found>}  free $(host_free_gb) GB"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 \
  > "scripts/logs/probe_pool_${SLURM_JOB_ID}.gpumem.csv" 2>/dev/null &
SMI=$!

declare -A RUN_PIECE=() RUN_GB=()
PENDING=(${POOL})
USED=0

reap() {
  local pid rc p
  for pid in "${!RUN_PIECE[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    p="${RUN_PIECE[$pid]}"
    if (( rc == 0 )); then piece_status "$p" done; else piece_status "$p" "failed rc=${rc}"; fi
    echo "[$(date +%T)] finished ${p} rc=${rc}"
    USED=$(( USED - RUN_GB[$pid] ))
    unset "RUN_PIECE[$pid]" "RUN_GB[$pid]"
  done
}

on_term() {
  local pid
  echo "[$(date +%T)] job ending (signal): marking running pieces killed"
  for pid in "${!RUN_PIECE[@]}"; do piece_status "${RUN_PIECE[$pid]}" killed; kill "$pid" 2>/dev/null; done
  kill "$SMI" 2>/dev/null; exit 1
}
trap on_term TERM INT

free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }

WHY=""
while :; do
  reap
  KEEP=(); for p in "${PENDING[@]}"; do piece_claimed "$p" || KEEP+=("$p"); done; PENDING=("${KEEP[@]}")
  if (( ${#PENDING[@]} == 0 && ${#RUN_PIECE[@]} == 0 )); then break; fi
  STARTED=0; WAITING=0
  if (( ${#RUN_PIECE[@]} < MAX_PAR && ${#PENDING[@]} > 0 )); then
    LEFT=$(( (END - $(date +%s)) / 60 ))
    FREE=$(free_gb); FREE=${FREE:-0}
    HFREE=$(host_free_gb)
    if flock -n "$SQ_LOAD_LOCK" true; then LOADING=0; else LOADING=1; fi
    for p in "${PENDING[@]}"; do
      piece_ready "$p" || { WAITING=1; WHY="not ready: $p"; continue; }
      piece_waits_for_fp32 "$p" && { WAITING=1; WHY="FP32 pair in progress: $p"; continue; }
      need=$(piece_gb "$p"); hneed=$(piece_host_gb "$p"); mins=$(piece_min "$p")
      if (( MAX_PAR > 1 )); then cont=15; else cont=13; fi      # x10 (packed pieces get their own CPUs)
      (( LEFT * 10 < mins * cont )) && { WHY="time: $p"; continue; }
      (( USED + need > GPU_BUDGET_GB || FREE < need + 4 )) && { WHY="gpu memory: $p"; continue; }
      (( LOADING )) && { WAITING=1; WHY="a model load is in progress"; break; }
      (( HFREE < hneed + 4 )) && { WHY="host memory (${HFREE} GB free): $p"; continue; }
      piece_claim "$p" || continue
      key=$(piece_key "$p")
      echo "[$(date +%T)] start ${p} (est GPU ${need} GB, host ${hneed} GB, ${mins} min | GPU used ${USED}+${need}/${GPU_BUDGET_GB}, free ${FREE} | host free ${HFREE} GB | ${LEFT} min left)"
      ( export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
        echo "=== piece ${p} | job ${SLURM_JOB_ID} | SQ_GPTQ_DENSE=${SQ_GPTQ_DENSE:-<unset>} | $(date) ==="
        exec $(piece_cmd "$p") ) > "scripts/logs/probe_${key}_${SLURM_JOB_ID}.log" 2>&1 &
      RUN_PIECE[$!]="$p"; RUN_GB[$!]=$need; USED=$(( USED + need ))
      KEEP=(); for q in "${PENDING[@]}"; do [[ "$q" == "$p" ]] || KEEP+=("$q"); done; PENDING=("${KEEP[@]}")
      STARTED=1
      break
    done
  fi
  if (( ${#RUN_PIECE[@]} == 0 && STARTED == 0 && WAITING == 0 && ${#PENDING[@]} > 0 )); then
    echo "[$(date +%T)] cannot start (${WHY}): ${PENDING[*]}"; break
  fi
  if (( ${#RUN_PIECE[@]} == 0 && STARTED == 0 && WAITING == 1 )); then
    # only waiting pieces left (not ready / FP32 pair elsewhere): leave them to other jobs
    echo "[$(date +%T)] nothing runnable now (${WHY}); leaving: ${PENDING[*]}"; break
  fi
  if (( STARTED )); then sleep 90 & wait $!; else sleep 20 & wait $!; fi   # wait: TERM trap fires at once
done
kill "$SMI" 2>/dev/null
echo "=== pieces"
for p in ${POOL}; do
  d="${PROBE_CLAIMS}/$(piece_key "$p")"
  [[ -d "$d" ]] && echo "  ${p}: $(cat "$d/status") (job $(cut -d' ' -f1 "$d/job"))" || echo "  ${p}: unclaimed"
done
echo "peak GPU memory used: $(cut -d, -f2 "scripts/logs/probe_pool_${SLURM_JOB_ID}.gpumem.csv" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1) MiB"
echo "=== done | $(date) ==="
