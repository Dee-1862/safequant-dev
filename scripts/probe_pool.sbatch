#!/bin/bash
#SBATCH --job-name=sq_probe_pool
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -N 1
#SBATCH --time=4:30:00
#SBATCH --mem=112G
#SBATCH --output=scripts/logs/probe_pool_%j.out
#SBATCH --error=scripts/logs/probe_pool_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# Run probe-only pieces (X2 cells, X3, X4; see scripts/probe_pieces.sh) from POOL on ONE GPU,
# up to MAX_PAR at a time as separate processes. Generation jobs never go through this.
#   single:  POOL="x3:llama7b"  MAX_PAR=1   --mem=64G --cpus-per-task=8
#   packed:  POOL="<all>"       MAX_PAR=4   --mem=112G --cpus-per-task=32
# A piece starts only if (a) nobody claimed it, (b) the GPU-memory estimates of the running
# pieces plus this one fit GPU_BUDGET_GB (default 88 of ~95 GB) and nvidia-smi shows that
# much free, and (c) the time left covers its expected minutes x CONTENTION (2 when sharing
# the GPU, 1.3 alone). Pieces start 90 s apart; GPU memory is logged every 60 s.
# Submit (packed):  POOL="$PROBE_ALL" MAX_PAR=4 bash scripts/submit.sh scripts/probe_pool.sbatch \
#                     --cpus-per-task=32 --mem=112G --time=4:30:00
set -uo pipefail
source scripts/probe_pieces.sh
POOL="${POOL:-$PROBE_ALL}"
MAX_PAR="${MAX_PAR:-1}"
GPU_BUDGET_GB="${GPU_BUDGET_GB:-88}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / MAX_PAR )); (( THREADS < 1 )) && THREADS=1
END=$(date -d "$(squeue -h -j "${SLURM_JOB_ID}" -o %e)" +%s 2>/dev/null || echo $(( $(date +%s) + 3600 )))
echo "=== probe pool | job ${SLURM_JOB_ID} | MAX_PAR=${MAX_PAR} threads/proc=${THREADS} budget=${GPU_BUDGET_GB}G | ends $(date -d @"$END") ==="
echo "pool: ${POOL}"

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

while :; do
  reap
  # drop pieces someone else claimed
  KEEP=(); for p in "${PENDING[@]}"; do piece_claimed "$p" || KEEP+=("$p"); done; PENDING=("${KEEP[@]}")
  if (( ${#PENDING[@]} == 0 && ${#RUN_PIECE[@]} == 0 )); then break; fi
  STARTED=0
  if (( ${#RUN_PIECE[@]} < MAX_PAR && ${#PENDING[@]} > 0 )); then
    LEFT=$(( (END - $(date +%s)) / 60 ))
    FREE=$(free_gb); FREE=${FREE:-0}
    for p in "${PENDING[@]}"; do
      need=$(piece_gb "$p"); mins=$(piece_min "$p")
      if (( MAX_PAR > 1 )); then cont=20; else cont=13; fi      # x10
      (( LEFT * 10 < mins * cont )) && continue
      (( USED + need > GPU_BUDGET_GB )) && continue
      (( FREE < need + 4 )) && continue
      piece_claim "$p" || continue
      key=$(piece_key "$p")
      echo "[$(date +%T)] start ${p} (est ${need} GB, ${mins} min; used ${USED}+${need}/${GPU_BUDGET_GB} GB, free ${FREE} GB, ${LEFT} min left)"
      ( export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
        exec $(piece_cmd "$p") ) > "scripts/logs/probe_${key}_${SLURM_JOB_ID}.log" 2>&1 &
      RUN_PIECE[$!]="$p"; RUN_GB[$!]=$need; USED=$(( USED + need ))
      KEEP=(); for q in "${PENDING[@]}"; do [[ "$q" == "$p" ]] || KEEP+=("$q"); done; PENDING=("${KEEP[@]}")
      STARTED=1
      break
    done
  fi
  if (( ${#RUN_PIECE[@]} == 0 && STARTED == 0 && ${#PENDING[@]} > 0 )); then
    echo "[$(date +%T)] cannot start (time or memory): ${PENDING[*]}"; break
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
