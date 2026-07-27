#!/bin/bash
# Submit smoke chain with Slurm dependencies:
#   env → (llama + phi in parallel) → optional pairs (llama then phi)
#
# Usage (from repo root on Bridges-2):
#   bash scripts/tests/submit_smoke.sh
#   SKIP_PAIRS=1 bash scripts/tests/submit_smoke.sh
#   SKIP_PHI=1 bash scripts/tests/submit_smoke.sh
#   SKIP_LLAMA=1 bash scripts/tests/submit_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
mkdir -p scripts/logs

ENV_JOB=$(sbatch --parsable scripts/tests/00_env_check.sbatch)
echo "submitted env check: ${ENV_JOB}"

SMOKE_LLAMA=""
SMOKE_PHI=""

if [[ "${SKIP_LLAMA:-0}" != "1" ]]; then
  SMOKE_LLAMA=$(sbatch --parsable --dependency=afterok:"${ENV_JOB}" \
    scripts/tests/01_smoke_llama_aqlm2.sbatch)
  echo "submitted llama/aqlm2 smoke (afterok ${ENV_JOB}): ${SMOKE_LLAMA}"
else
  echo "SKIP_LLAMA=1  -  not submitting 01_smoke_llama_aqlm2.sbatch"
fi

if [[ "${SKIP_PHI:-0}" != "1" ]]; then
  SMOKE_PHI=$(sbatch --parsable --dependency=afterok:"${ENV_JOB}" \
    scripts/tests/03_smoke_phi3_aqlm2.sbatch)
  echo "submitted phi3/aqlm2 smoke (afterok ${ENV_JOB}): ${SMOKE_PHI}"
else
  echo "SKIP_PHI=1  -  not submitting 03_smoke_phi3_aqlm2.sbatch"
fi

if [[ "${SKIP_PAIRS:-0}" != "1" ]]; then
  if [[ -n "${SMOKE_LLAMA}" ]]; then
    PAIRS_LLAMA=$(sbatch --parsable --dependency=afterok:"${SMOKE_LLAMA}" \
      scripts/tests/02_smoke_qresafe_pairs.sbatch)
    echo "submitted llama pairs smoke (afterok ${SMOKE_LLAMA}): ${PAIRS_LLAMA}"
  fi
  if [[ -n "${SMOKE_PHI}" ]]; then
    PAIRS_PHI=$(sbatch --parsable --dependency=afterok:"${SMOKE_PHI}" \
      scripts/tests/04_smoke_qresafe_pairs_phi3.sbatch)
    echo "submitted phi3 pairs smoke (afterok ${SMOKE_PHI}): ${PAIRS_PHI}"
  fi
  if [[ -z "${SMOKE_LLAMA}" && -z "${SMOKE_PHI}" ]]; then
    echo "No cell smokes submitted  -  skipping pairs"
  fi
else
  echo "SKIP_PAIRS=1  -  not submitting pairs smokes"
fi

echo
echo "Watch:  squeue -u \$USER"
echo "Logs:   tail -f scripts/logs/smoke_env_${ENV_JOB}.out"
[[ -n "${SMOKE_LLAMA}" ]] && echo "        tail -f scripts/logs/smoke_llama_aqlm2_${SMOKE_LLAMA}.out"
[[ -n "${SMOKE_PHI}" ]] && echo "        tail -f scripts/logs/smoke_phi3_aqlm2_${SMOKE_PHI}.out"
