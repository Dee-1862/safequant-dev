#!/bin/bash
# Login-node watcher (light: squeue/sbatch every 60 s, no compute). The interactive partition
# ($SLURM_INTERACTIVE_PARTITION, from .env) allows ONE
# job per user (queued or running), so dependencies cannot chain there. Whenever that slot is
# free, submit the next unclaimed probe pieces (up to CHAIN_PACK=3 that fit the GPU together;
# order from CHAIN_ORDER) as one scripts/probe_pool.sbatch job on one GPU under 2 h. Also cancels PENDING probe-pool
# jobs (any partition) whose pieces have all been claimed, so no allocation is wasted.
#   nohup setsid bash scripts/probe_interactive_chain.sh >> scripts/logs/probe_chain.log 2>&1 &
# Stops when every piece is claimed. Job -> pool map: $PROBE_CLAIMS/jobs.tsv
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"; export REPO_ROOT
# same view of the world as the jobs: QM_ROOT from .env, dense GPTQ path where copies exist
source scripts/load_env.sh
export QM_ROOT
PART="${SLURM_INTERACTIVE_PARTITION:?set SLURM_INTERACTIVE_PARTITION in .env}"
[[ -z "${SQ_GPTQ_DENSE+x}" && -d "${QM_ROOT}/gptq_dense" ]] && export SQ_GPTQ_DENSE=1
source scripts/probe_pieces.sh
ORDER="${CHAIN_ORDER:-$PROBE_ALL}"
JOBS="${PROBE_CLAIMS}/jobs.tsv"; mkdir -p "${PROBE_CLAIMS}"; touch "$JOBS"
limit() { case "$1" in x4:*) echo 1:55:00 ;; *) echo 1:30:00 ;; esac; }   # short limits backfill sooner; ~1.6x expected
echo "[$(date)] chain start: ${ORDER}"
while :; do
  # cancel pending pool jobs with nothing left to do
  while IFS=$'\t' read -r jid pool; do
    [[ -z "$jid" ]] && continue
    st=$(squeue -h -j "$jid" -o %T 2>/dev/null)
    [[ "$st" == PENDING ]] || continue
    left=0; for p in $pool; do piece_claimed "$p" || left=1; done
    (( left == 0 )) && scancel "$jid" && echo "[$(date)] cancelled pending ${jid} (all pieces claimed: ${pool})"
  done < "$JOBS"
  # next interactive job: up to CHAIN_PACK ready pieces in priority order that fit the GPU
  # budget together (one GPU, 8 CPUs per piece; probe_pool runs them as parallel processes)
  PACK=(); GB=0; MINS=0; ALL=1; FAMS=" "
  for p in $ORDER; do
    piece_claimed "$p" && continue
    ALL=0
    (( ${#PACK[@]} >= ${CHAIN_PACK:-3} )) && continue
    piece_ready "$p" || continue            # e.g. GPTQ cell whose dense copy is not built yet
    piece_waits_for_fp32 "$p" && continue   # its family's FP32 pair is being computed
    IFS=: read -r k m _ <<< "$p"
    # two x3p pieces of one family without a cached FP32 pair would both compute it: one per job
    if [[ "$k" == x3p ]] && ! fp32_cached "$m"; then [[ "$FAMS" == *" $m "* ]] && continue; FAMS+="$m "; fi
    g=$(piece_gb "$p"); (( GB + g > 88 )) && continue
    PACK+=("$p"); GB=$(( GB + g )); mn=$(piece_min "$p"); (( mn > MINS )) && MINS=$mn
  done
  if (( ALL )); then echo "[$(date)] all pieces claimed; chain done"; break; fi
  if (( ${#PACK[@]} > 0 )) && [[ -z "$(squeue -h -u "$USER" -p "$PART" -o %i 2>/dev/null)" ]]; then
    n=${#PACK[@]}; (( n > 1 )) && lim=1:55:00 || lim=$(limit "${PACK[0]}")
    id=$(SLURM_PARTITION="$PART" POOL="${PACK[*]}" MAX_PAR=$n bash scripts/submit.sh scripts/probe_pool.sbatch \
           --parsable --cpus-per-task=$(( 8 * n )) --mem=$(( n > 1 ? 113 : 64 ))G --time="$lim" \
           --job-name="sq_pp_i_$(piece_key "${PACK[0]}")$( (( n > 1 )) && echo "_p$n")" 2>&1 | tail -1)
    id="${id%%;*}"
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      printf '%s\t%s\n' "$id" "${PACK[*]}" >> "$JOBS"; echo "[$(date)] interactive ${id}: ${PACK[*]}"
    else
      echo "[$(date)] submit failed for ${PACK[*]}: ${id}"
    fi
  fi
  sleep 60
done
