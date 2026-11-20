#!/bin/bash
# scripts/quantize_gptq_all.sh

# Submit AutoGPTQ GPTQ quantization (bits 4/3/2, group 128) for every study model, each as its own SLURM job in the sq_autogptq env. Run from a LOGIN node:
#   bash scripts/quantize_gptq_all.sh


# Override any axis via env vars, e.g. redo just one model / bit-width:
#   MODELS="qwen7b" BITS="4" bash scripts/quantize_gptq_all.sh
#   MODELS="mistral7b-v02 phi3" bash scripts/quantize_gptq_all.sh
# Optional Slurm dependency for every submitted job (used by scripts/delta_queue.sh):
#   DEPENDENCY=afterok:123456 MODELS=llama7b BITS=4 bash scripts/quantize_gptq_all.sh
# The last line printed is "JOBIDS: <id> ..." for scripts that chain on these jobs.


# Each job writes $QM_ROOT/gptq/<model>-gptq<bits>bit-g<group> (where config.py reads it) and prints the config.QUANTIZED_CHECKPOINTS line.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/load_env.sh"

if [[ "${GPTQ_BACKEND:-autogptq}" == "autogptq" && -z "${AGENV:-}" ]]; then
  echo "Set AGENV in .env (path to sq_autogptq conda env)." >&2
  exit 1
fi

# NOTE on Qwen: this uses Qwen2.5-7B (qwen7b).
MODELS="${MODELS:-llama7b qwen7b mistral7b-v02 phi3}"
BITS="${BITS:-4 3 2}"
GROUP="${GROUP:-128}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ "${GPTQ_BACKEND:-autogptq}" == "autogptq" && ! -d "$AGENV" ]]; then
  echo "ERROR: quant env not found: $AGENV" >&2
  echo "Set AGENV in .env (path to sq_autogptq conda env)." >&2
  exit 1
fi

echo "models = $MODELS"
echo "bits   = $BITS    group = $GROUP"
echo "-----------------------------------------------------"
DEP_ARGS=()
if [[ -n "${DEPENDENCY:-}" ]]; then
  DEP_ARGS=(--dependency="${DEPENDENCY}")
  echo "dependency = ${DEPENDENCY}"
fi
# GPTQ_BACKEND=autogptq (default) runs in $AGENV; gptqmodel runs in sq_clean ($CONDA_ENV).
GPTQ_BACKEND="${GPTQ_BACKEND:-autogptq}"
case "$GPTQ_BACKEND" in
  autogptq)  JOB_ENV="$AGENV" ;;
  gptqmodel) JOB_ENV="${CONDA_ENV:?set CONDA_ENV in .env (sq_clean) for GPTQ_BACKEND=gptqmodel}" ;;
  *) echo "ERROR: GPTQ_BACKEND must be autogptq or gptqmodel, got '$GPTQ_BACKEND'" >&2; exit 2 ;;
esac
echo "backend = $GPTQ_BACKEND  (env $JOB_ENV)"
n=0
JOBIDS=""
for m in $MODELS; do
  for b in $BITS; do
    jid=$(bash scripts/submit.sh scripts/quantize_gptq.sbatch --parsable "${DEP_ARGS[@]}" \
          --export=ALL,CONDA_ENV="$JOB_ENV",GPTQ_BACKEND="$GPTQ_BACKEND" -- "$m" "$b" "$GROUP")
    jid="${jid%%;*}"   # --parsable may print "<id>;<cluster>"
    printf "  %-16s gptq%s  -> job %s\n" "$m" "$b" "$jid"
    JOBIDS="${JOBIDS:+${JOBIDS} }${jid}"
    n=$((n + 1))
  done
done
echo "-----------------------------------------------------"
echo "submitted $n jobs.  watch: squeue -u \$USER"
echo "outputs -> \$QM_ROOT/gptq/<model>-gptq<bits>bit-g${GROUP}"
echo "JOBIDS: ${JOBIDS}"
