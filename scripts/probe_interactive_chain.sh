#!/bin/bash
# Login-node watcher (light: squeue/sbatch every 60 s, no compute). ghx4-interactive allows ONE
# job per user (queued or running), so dependencies cannot chain there. Whenever that slot is
# free, submit the next unclaimed probe piece (order: X3, X4, X2 cells; scripts/probe_pieces.sh)
# as a single-piece scripts/probe_pool.sbatch job under 2 h. Also cancels PENDING probe-pool
# jobs (any partition) whose pieces have all been claimed, so no allocation is wasted.
#   nohup setsid bash scripts/probe_interactive_chain.sh >> scripts/logs/probe_chain.log 2>&1 &
# Stops when every piece is claimed. Job -> pool map: $PROBE_CLAIMS/jobs.tsv
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"; export REPO_ROOT
# same view of the world as the jobs: QM_ROOT from .env, dense GPTQ path where copies exist
[[ -z "${QM_ROOT:-}" && -f .env ]] && QM_ROOT=$(sed -nE 's/^(export )?QM_ROOT=//p' .env | tail -1)
export QM_ROOT
[[ -z "${SQ_GPTQ_DENSE+x}" && -d "${QM_ROOT}/gptq_dense" ]] && export SQ_GPTQ_DENSE=1
source scripts/probe_pieces.sh
ORDER="${CHAIN_ORDER:-$PROBE_ALL}"
JOBS="${PROBE_CLAIMS}/jobs.tsv"; mkdir -p "${PROBE_CLAIMS}"; touch "$JOBS"
limit() { case "$1" in x4:*) echo 1:15:00 ;; *) echo 1:30:00 ;; esac; }   # short limits backfill sooner; ~1.6x expected
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
  NEXT=""; ALL=1
  for p in $ORDER; do
    piece_claimed "$p" && continue
    ALL=0
    piece_ready "$p" || continue            # e.g. GPTQ cell whose dense copy is not built yet
    piece_waits_for_fp32 "$p" && continue   # its family's FP32 pair is being computed
    NEXT="$p"; break
  done
  if (( ALL )); then echo "[$(date)] all pieces claimed; chain done"; break; fi
  if [[ -n "$NEXT" && -z "$(squeue -h -u "$USER" -p ghx4-interactive -o %i 2>/dev/null)" ]]; then
    id=$(SLURM_PARTITION=ghx4-interactive POOL="$NEXT" MAX_PAR=1 bash scripts/submit.sh scripts/probe_pool.sbatch \
           --parsable --cpus-per-task=8 --mem=64G --time="$(limit "$NEXT")" --job-name="sq_pp_i_$(piece_key "$NEXT")" 2>&1 | tail -1)
    id="${id%%;*}"
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      printf '%s\t%s\n' "$id" "$NEXT" >> "$JOBS"; echo "[$(date)] interactive ${id}: ${NEXT}"
    else
      echo "[$(date)] submit failed for ${NEXT}: ${id}"
    fi
  fi
  sleep 60
done
