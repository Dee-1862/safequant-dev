#!/bin/bash
# Submit a SafeQuant sbatch job with cluster settings taken from .env.
#
#   bash scripts/submit.sh <path/to/job.sbatch> [extra sbatch args] [-- job args]
#
# Runs:  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
#               --gpus-per-node=1 <extra args> <job.sbatch>
# Command-line flags override the matching #SBATCH lines (YOUR_ALLOCATION /
# YOUR_PARTITION placeholders). The `#SBATCH --gpus=1` header is a different option
# from --gpus-per-node and could be double-counted, so this script strips
# `#SBATCH --gpus=...` lines from a temporary copy that sbatch reads; the files on
# disk are untouched. Set SQ_KEEP_GPUS_LINE=1 to disable the stripping.
# Arguments after a literal `--` are passed to the job script as $1 $2 ...:
#   bash scripts/submit.sh scripts/quantize_gptq.sbatch --parsable -- llama7b 4 128
#
# Env (from .env or the shell):
#   SLURM_ACCOUNT    required   (from `accounts`)
#   SLURM_PARTITION  required   (a GPU partition with >= 40 GB per GPU fits 8B fp16)
#   SQ_GPUS_PER_NODE default 1; 0 = CPU job (no --gpus-per-node passed), e.g.
#     SLURM_PARTITION="$SLURM_CPU_PARTITION" SQ_GPUS_PER_NODE=0 bash scripts/submit.sh job.sbatch
#   (shell values win over .env, see scripts/load_env.sh)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/submit.sh <job.sbatch> [extra sbatch args] [-- job args]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/load_env.sh"

JOB="$1"; shift
SB_ARGS=(); JOB_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; JOB_ARGS=("$@"); break; fi
  SB_ARGS+=("$1"); shift
done
[[ -f "$JOB" ]] || { echo "ERROR: job file not found: $JOB" >&2; exit 2; }
JOB="$(cd "$(dirname "$JOB")" && pwd)/$(basename "$JOB")"

if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: SLURM_ACCOUNT is not set. Put it in .env (see .env.example); find it with 'accounts'." >&2
  exit 1
fi
if [[ -z "${SLURM_PARTITION:-}" ]]; then
  echo "ERROR: SLURM_PARTITION is not set. Put it in .env (see .env.example)." >&2
  exit 1
fi
PARTITION="${SLURM_PARTITION}"
GPUS="${SQ_GPUS_PER_NODE:-1}"

# Job files use paths relative to the repo root (logs, ${SLURM_SUBMIT_DIR}/scripts/_env.sh).
cd "$ROOT"
mkdir -p "${SQ_LOG_DIR:-${ROOT}/scripts/logs}"

ARGS=(--account="$SLURM_ACCOUNT" --partition="$PARTITION")
if [[ "$GPUS" != "0" ]]; then
  ARGS+=(--gpus-per-node="$GPUS")
fi

echo "submit: ${JOB#"$ROOT"/}  account=$SLURM_ACCOUNT partition=$PARTITION gpus-per-node=$GPUS ${SB_ARGS[*]:-} ${JOB_ARGS[*]:+-- ${JOB_ARGS[*]}}" >&2
if [[ "${SQ_KEEP_GPUS_LINE:-0}" == "1" ]]; then
  sbatch "${ARGS[@]}" "${SB_ARGS[@]}" "$JOB" "${JOB_ARGS[@]}"
else
  # sbatch copies the script at submit time, so the temp file can go right after.
  TMP="$(mktemp "${TMPDIR:-/tmp}/sq_job.XXXXXX.sbatch")"
  trap 'rm -f "$TMP"' EXIT
  sed '/^#SBATCH[[:space:]]\+--gpus=/d' "$JOB" > "$TMP"
  sbatch "${ARGS[@]}" "${SB_ARGS[@]}" "$TMP" "${JOB_ARGS[@]}"
fi
