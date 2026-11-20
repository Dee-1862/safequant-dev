#!/bin/bash
# Login-node runner (light: squeue/sbatch every 60 s): submits the jobs listed in a queue file
# to the interactive partition ($SLURM_INTERACTIVE_PARTITION, from .env) one at a time, in order,
# whenever the lane (1 job per user) is free, and cancels each entry's batch-partition copies
# once its interactive job has started (RUNNING or ended).
#   nohup setsid bash scripts/interactive_queue.sh scripts/logs/iq/queue.sh >> scripts/logs/iq/runner.log 2>&1 &
# Queue file (bash, re-read every loop, so entries can be appended while it runs): one line per job
#   q KEY "CANCEL_IDS" "ENV_ASSIGNMENTS" JOB_FILE "SBATCH_ARGS"
# State: <queue dir>/state.tsv (KEY<TAB>JOBID<TAB>CANCELLED). Stops when every entry has ended.
# The body is one function, read in full before it runs, so editing this file cannot break a
# running copy.
main() {
  set -u
  cd "$(dirname "$0")/.."
  source scripts/load_env.sh
  local PART="${SLURM_INTERACTIVE_PARTITION:?set SLURM_INTERACTIVE_PARTITION in .env}"
  local QF="$1" ST="$(dirname "$1")/state.tsv"
  touch "$ST"
  log() { echo "[$(date '+%F %T')] $*"; }
  log "runner start: $QF"
  while :; do
    KEYS=(); CANC=(); ENVS=(); JOBF=(); ARGS=()
    q() { KEYS+=("$1"); CANC+=("$2"); ENVS+=("$3"); JOBF+=("$4"); ARGS+=("$5"); }
    # shellcheck disable=SC1090
    source "$QF"
    local busy; busy=$(squeue -h -u "$USER" -p "$PART" -o %i 2>/dev/null)
    local all_ended=1 next=-1 i
    for i in "${!KEYS[@]}"; do
      local k=${KEYS[$i]} line jid st cdone
      line=$(awk -F'\t' -v k="$k" '$1 == k' "$ST" | tail -1)
      if [[ -z "$line" ]]; then all_ended=0; (( next < 0 )) && next=$i; continue; fi
      IFS=$'\t' read -r _ jid cdone <<< "$line"
      st=$(squeue -h -j "$jid" -o %T 2>/dev/null)
      [[ -n "$st" ]] && all_ended=0
      if [[ "$cdone" != yes && -n "${CANC[$i]// }" && "$st" != PENDING ]]; then
        scancel ${CANC[$i]} 2>/dev/null
        log "$k ($jid) started: cancelled batch copy ${CANC[$i]}"
        awk -F'\t' -v OFS='\t' -v k="$k" '$1 == k {$3 = "yes"} 1' "$ST" > "$ST.tmp" && mv "$ST.tmp" "$ST"
      fi
    done
    if (( all_ended )); then log "queue done"; break; fi
    if (( next >= 0 )) && [[ -z "$busy" ]]; then
      local out id
      out=$( ( eval "export ${ENVS[$next]}"; SLURM_PARTITION="$PART" bash scripts/submit.sh \
               "${JOBF[$next]}" --parsable ${ARGS[$next]} ) 2>&1 | tail -1)
      id=${out%%;*}
      if [[ "$id" =~ ^[0-9]+$ ]]; then
        printf '%s\t%s\t%s\n' "${KEYS[$next]}" "$id" no >> "$ST"; log "submitted ${KEYS[$next]}: $id"
      else
        log "submit failed for ${KEYS[$next]}: $out"
      fi
    fi
    sleep 60
  done
}
main "$@"
exit
