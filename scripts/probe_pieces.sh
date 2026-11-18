#!/bin/bash
# Probe-only pieces (no generation) shared by scripts/probe_pool.sbatch and
# scripts/probe_interactive_chain.sh. One piece = one model (X4) or one cell (X2, X3):
#   x3p:<model>:<quant>[:<tag>]  X3 matched pair: probe_accuracy --paired (real + shuffled-label
#                                probe on the same activations, seed 42); FP32 pair cached once
#                                per model family in $X3_FP32_CACHE
#   x4:<model>                   X4 refit floor, 100 splits (= probe_refit_control_splits100_llama.sbatch)
#   x2:<model>:<quant>[:<tag>]   X2 leakage control on a quantized cell (= probe_leakage_control_quant.sbatch)
#   x3:<model>                   (superseded) X3 shuffled-only, aqlm2
# Collapsed cells use tag "rebuilt" (DeltaAI rebuilt checkpoints, dense GPTQ path).
# Outputs keep their names; every script refuses to overwrite an existing result.
#
# Claims: whichever job starts a piece first claims it with an atomic mkdir in
# $PROBE_CLAIMS/<piece>; every other job skips it. A claimed piece is never re-run
# automatically (status file: running | done | failed rc=N | killed).

PROBE_CLAIMS="${PROBE_CLAIMS:-${REPO_ROOT:-$(pwd)}/scripts/logs/probe_claims}"
X3_FP32_CACHE="${X3_FP32_CACHE:-${QM_ROOT%/*}/sq_cache/x3_fp32}"

# wave 1 (X3 entries here are the superseded shuffled-only runs; both already done)
PROBE_ALL="x3:llama7b x3:phi3 x4:llama7b x4:phi3 \
x2:llama7b:aqlm2 x2:mistral7b-v02:aqlm2 x2:phi3:aqlm2 \
x2:llama7b:awq x2:qwen7b:awq x2:mistral7b-v02:awq x2:phi3:awq"
# tier A: X2 GPTQ cells (dense copies) + X4 for mistral/qwen
PROBE_TIER_A="x2:llama7b:gptq4 x2:llama7b:gptq3 x2:qwen7b:gptq4 x2:qwen7b:gptq3 \
x2:mistral7b-v02:gptq4 x2:phi3:gptq4 x2:phi3:gptq3 x4:mistral7b-v02 x4:qwen7b"
# tier B: X3 matched pair on the 14 coherent cells (one per family first, to fill the FP32 cache)
PROBE_TIER_B="x3p:llama7b:aqlm2 x3p:qwen7b:awq x3p:mistral7b-v02:aqlm2 x3p:phi3:aqlm2 \
x3p:llama7b:awq x3p:llama7b:gptq4 x3p:llama7b:gptq3 x3p:qwen7b:gptq4 x3p:qwen7b:gptq3 \
x3p:mistral7b-v02:awq x3p:mistral7b-v02:gptq4 x3p:phi3:awq x3p:phi3:gptq4 x3p:phi3:gptq3"
# tier C (lowest): the 5 collapsed cells on rebuilt dense checkpoints, X3 pair + X2
PROBE_TIER_C="x3p:llama7b:gptq2:rebuilt x3p:qwen7b:gptq2:rebuilt x3p:phi3:gptq2:rebuilt \
x2:llama7b:gptq2:rebuilt x2:qwen7b:gptq2:rebuilt x2:phi3:gptq2:rebuilt \
x3p:mistral7b-v02:gptq2:rebuilt x3p:mistral7b-v02:gptq3:rebuilt \
x2:mistral7b-v02:gptq2:rebuilt x2:mistral7b-v02:gptq3:rebuilt"

piece_key() { echo "${1//:/_}"; }

piece_cmd() {  # piece_cmd <piece>  -> python command
  local kind m q t
  IFS=: read -r kind m q t <<< "$1"
  case "$kind" in
    x3p) echo "python -u -m experiments.signal_geometry.probe_accuracy --model $m --quantizer $q --n_calib 200 --probe_type linear --paired --fp32_cache_dir ${X3_FP32_CACHE}${t:+ --tag $t}" ;;
    x3)  echo "python -u -m experiments.signal_geometry.probe_accuracy --model $m --quantizer aqlm2 --n_calib 200 --probe_type linear --shuffle_labels" ;;
    x4)  echo "python -u -m experiments.boundary.probe_refit_control --model $m --n_calib 200 --n_splits 100" ;;
    x2)  echo "python -u -m experiments.boundary.probe_leakage_control --model $m --quantizer $q --n_calib 200 --n_transfer 200${t:+ --tag $t}" ;;
    *)   return 1 ;;
  esac
}

# GPU memory (GB): weights + ~3 GB context/activations. fp32: Llama-3.1-8B 32, Qwen2.5-7B 30.5,
# Mistral-7B 29, Phi-3-mini 15. Dense GPTQ = fp16 (half that); AQLM/AWQ from checkpoint size.
fp32_gb()  { case "$1" in llama7b) echo 35 ;; qwen7b) echo 34 ;; mistral7b-v02) echo 32 ;; phi3) echo 19 ;; *) echo 35 ;; esac; }
quant_gb() {
  case "$2" in
    gptq*) case "$1" in llama7b|qwen7b) echo 19 ;; mistral7b-v02) echo 18 ;; phi3) echo 11 ;; *) echo 19 ;; esac ;;
    aqlm2) case "$1" in llama7b) echo 7 ;; mistral7b-v02) echo 5 ;; phi3) echo 4 ;; *) echo 7 ;; esac ;;
    awq)   case "$1" in llama7b|qwen7b) echo 9 ;; mistral7b-v02) echo 7 ;; phi3) echo 5 ;; *) echo 9 ;; esac ;;
    *) echo 19 ;;
  esac
}
fp32_cached() { compgen -G "${X3_FP32_CACHE}/x3_fp32_pair_${1}_*.json" >/dev/null; }

piece_gb() {
  local kind m q t a b
  IFS=: read -r kind m q t <<< "$1"
  case "$kind" in
    x3p) a=$(quant_gb "$m" "$q"); if fp32_cached "$m"; then echo "$a"; else b=$(fp32_gb "$m"); echo $(( a > b ? a : b )); fi ;;
    x3|x4) fp32_gb "$m" ;;
    x2) quant_gb "$m" "$q" ;;
    *) echo 35 ;;
  esac
}

# Host-memory peak (GB) while loading (measured: fp32 llama7b 21 GB, phi3 11 GB)
piece_host_gb() {
  local kind m q t
  IFS=: read -r kind m q t <<< "$1"
  local fp=0
  case "$kind" in x3|x4) fp=1 ;; x3p) fp32_cached "$m" || fp=1 ;; esac
  if (( fp )); then
    case "$m" in phi3) echo 12 ;; *) echo 22 ;; esac
  else
    case "$q" in gptq*) case "$m" in phi3) echo 8 ;; *) echo 16 ;; esac ;; *) echo 8 ;; esac
  fi
}

# Expected minutes running alone. Measured: X3 shuffled-only 34 min (llama) / 42 min (phi3),
# of which model loads + extraction ~10-15 min and probe fits the rest. X4 = one FP32
# extraction + 6400 small logistic fits; X2 = one quantized extraction of ~1050 prompts + fits.
piece_min() {
  local kind m q t
  IFS=: read -r kind m q t <<< "$1"
  case "$kind" in
    x3p) if fp32_cached "$m"; then echo 35; else echo 60; fi ;;
    x3) echo 45 ;; x4) echo 45 ;; x2) echo 50 ;; *) echo 90 ;;
  esac
}

# Ready = its checkpoint can be loaded now (GPTQ cells need the dense copy on Delta)
piece_ready() {
  local kind m q t
  IFS=: read -r kind m q t <<< "$1"
  [[ "$q" == gptq* && "${SQ_GPTQ_DENSE:-0}" == 1 ]] || return 0
  [[ -f "${QM_ROOT}/gptq_dense/${m}-gptq${q#gptq}bit-g128/config.json" ]]
}

# x3p pieces of one family wait while another job computes that family's FP32 pair
piece_waits_for_fp32() {
  local kind m q t d
  IFS=: read -r kind m q t <<< "$1"
  [[ "$kind" == x3p ]] || return 1
  fp32_cached "$m" && return 1
  for d in "${PROBE_CLAIMS}"/x3p_"${m}"_*/; do
    [[ -f "$d/status" && "$(cat "$d/status")" == running ]] && return 0
  done
  return 1
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
