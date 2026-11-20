#!/bin/bash
#SBATCH --job-name=sq_delta_setup
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=scripts/logs/delta_setup_%j.out
#SBATCH --error=scripts/logs/delta_setup_%j.err
source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

# One-time Delta setup: fill QM_ROOT (~200 GB) and HF_HOME with every checkpoint,
# held-out model, judge and dataset the reviewer runs need. No GPU needed.
#
# Submit on the cheap CPU partition when .env names one:
#   SLURM_ACCOUNT="$SLURM_CPU_ACCOUNT" SLURM_PARTITION="$SLURM_CPU_PARTITION" \
#     SQ_GPUS_PER_NODE=0 bash scripts/submit.sh scripts/tests/delta_setup.sbatch
# (scripts/delta_queue.sh does this automatically and refuses if SLURM_CPU_PARTITION
# is set without SLURM_CPU_ACCOUNT; with neither it uses the GPU partition/account.)
#
# Steps: (0) HF login check  (1) fetch_checkpoints.sh -> QM_ROOT
#        (2+3) scripts/delta_prefetch.py -> HF_HOME (held-out FP+AQLM, judge, datasets)
#        (4) du -sh HF_HOME QM_ROOT. Exits non-zero if anything failed to download.

set -uo pipefail   # not -e: report every failed step, then exit non-zero

# --- 0. gated models need a HF token ---------------------------------------
echo "=== [0] HF login check | $(date) ==="
if ! WHO=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null); then
  echo "ERROR: not logged in to Hugging Face. On a login node run:" >&2
  echo "         huggingface-cli login      (token with access to meta-llama and mistralai)" >&2
  echo "       then re-submit this job. HF_HOME=${HF_HOME}" >&2
  exit 1
fi
echo "  logged in as: ${WHO}"
command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli whoami || true

FAIL=0

# --- 1. base / AWQ / AQLM checkpoints into QM_ROOT -------------------------
echo "=== [1] fetch_checkpoints.sh -> ${QM_ROOT} | $(date) ==="
FETCH_LOG="${SQ_LOG_DIR}/delta_setup_fetch_${SLURM_JOB_ID:-local}.log"
bash scripts/fetch_checkpoints.sh 2>&1 | tee "${FETCH_LOG}"
# fetch_checkpoints.sh exits 0 even when a repo fails; detect its failure banner.
if grep -q "SOME DOWNLOADS FAILED" "${FETCH_LOG}"; then
  echo "!! fetch_checkpoints.sh reported failures (see ${FETCH_LOG})" >&2
  FAIL=1
fi

# --- 2+3. held-out models, judge, datasets into HF_HOME --------------------
echo "=== [2-3] delta_prefetch.py -> ${HF_HOME} | $(date) ==="
python -u scripts/delta_prefetch.py || FAIL=1

# --- 4. sizes ----------------------------------------------------------------
echo "=== [4] disk usage | $(date) ==="
du -sh "${HF_HOME}" "${QM_ROOT}" 2>/dev/null || true

if [[ ${FAIL} -ne 0 ]]; then
  echo "=== SETUP FAILED: at least one model/dataset did not download (see above) | $(date) ===" >&2
  exit 1
fi
echo "=== setup OK | $(date) ==="
