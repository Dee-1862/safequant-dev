#!/bin/bash
# Probe-only pieces (no generation) shared by scripts/probe_pool.sbatch and
# scripts/probe_interactive_chain.sh. A piece is one model (X3, X4) or one cell (X2):
#   x3:<model>          X3 shuffled-label probe, aqlm2  (= probe_accuracy_shuffled_llama.sbatch)
#   x4:<model>          X4 refit floor, 100 splits      (= probe_refit_control_splits100_llama.sbatch)
#   x2:<model>:<quant>  X2 leakage control on a quantized cell (= probe_leakage_control_quant.sbatch)
# Same commands and arguments as those launchers; outputs keep their names (each script
# refuses to overwrite an existing result).
#
# Claims: whichever job starts a piece first claims it with an atomic mkdir in
# $PROBE_CLAIMS/<piece>; every other job skips it. A claimed piece is never re-run
# automatically (status file: running | done | failed rc=N | killed).

PROBE_CLAIMS="${PROBE_CLAIMS:-${REPO_ROOT:-$(pwd)}/scripts/logs/probe_claims}"
PROBE_ALL="x3:llama7b x3:phi3 x4:llama7b x4:phi3 \
x2:llama7b:aqlm2 x2:mistral7b-v02:aqlm2 x2:phi3:aqlm2 \
x2:llama7b:awq x2:qwen7b:awq x2:mistral7b-v02:awq x2:phi3:awq"

piece_key() { echo "${1//:/_}"; }

piece_cmd() {  # piece_cmd <piece>  -> python command (array-safe words)
  local kind m q
  IFS=: read -r kind m q <<< "$1"
  case "$kind" in
    x3) echo "python -u -m experiments.signal_geometry.probe_accuracy --model $m --quantizer aqlm2 --n_calib 200 --probe_type linear --shuffle_labels" ;;
    x4) echo "python -u -m experiments.boundary.probe_refit_control --model $m --n_calib 200 --n_splits 100" ;;
    x2) echo "python -u -m experiments.boundary.probe_leakage_control --model $m --quantizer $q --n_calib 200 --n_transfer 200" ;;
    *)  return 1 ;;
  esac
}

# GPU memory estimate (GB): weights + ~3 GB CUDA context/activations. fp32 llama7b
# (Llama-3.1-8B) = 32 GB, fp32 phi3 = 15 GB; quantized sizes from the checkpoints on disk.
piece_gb() {
  case "$1" in
    x3:llama7b|x4:llama7b) echo 35 ;;
    x3:phi3|x4:phi3)       echo 19 ;;
    x2:llama7b:aqlm2) echo 7 ;;  x2:mistral7b-v02:aqlm2) echo 5 ;;  x2:phi3:aqlm2) echo 4 ;;
    x2:llama7b:awq)   echo 9 ;;  x2:qwen7b:awq) echo 9 ;;  x2:mistral7b-v02:awq) echo 7 ;;  x2:phi3:awq) echo 5 ;;
    *) echo 35 ;;
  esac
}

# Expected minutes running alone (delta_queue.sh EST: x3 120/2 models, x4 150/2, x2 75/cell)
piece_min() {
  case "$1" in x3:*) echo 60 ;; x4:*) echo 75 ;; x2:*) echo 75 ;; *) echo 90 ;; esac
}

piece_claimed() { [[ -d "${PROBE_CLAIMS}/$(piece_key "$1")" ]]; }

piece_claim() {  # piece_claim <piece> -> 0 if this job now owns it
  mkdir -p "${PROBE_CLAIMS}"
  local d="${PROBE_CLAIMS}/$(piece_key "$1")"
  mkdir "$d" 2>/dev/null || return 1
  echo "${SLURM_JOB_ID:-local} $(date -Is) $(hostname)" > "$d/job"
  echo running > "$d/status"
}

piece_status() { echo "$2" > "${PROBE_CLAIMS}/$(piece_key "$1")/status"; }
