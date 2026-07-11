#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Reader-side low-bit REPAIR via quant-domain LR-QAT — Llama-3.1-8B / AQLM-2.
#
# Trains ONLY the selected projections at the SAME bit-width as the damaged
# AQLM-2 baseline (INT-2), on neutral C4 web text (no safety data), distilling
# toward the FP16 teacher, then fuses the adapter back into the INT-2 grid.
# Kills two confounds at once: added precision (effective bits stay 2) and
# safety fine-tuning (no harmful/safety examples in training).
#
# Conditions produced (each invocation emits ptq_untrained + lrqat_trained):
#   role=read  INT-2  → the KEY new row  + its untrained INT-2 PTQ control
#   role=write INT-2  → asymmetry control, IDENTICAL budget to read
#   role=read  INT-4  → optional sweep point (~3 eff bits, mixed precision)
#
# READ and WRITE share one BUDGET_ARGS string below → budget is provably matched.
# Baselines (FP16 41.0, AQLM-2 62.5) are read from outputs/readwrite_llama7b_aqlm2.json.
#
# Output: outputs/lrqat_llama7b_aqlm2_{read,write}_int{2,4}.json
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_lrqat_llama
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lrqat_llama_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/lrqat_llama_%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "=== Job ${SLURM_JOB_ID} | LR-QAT repair Llama/AQLM-2 | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /jet/home/bhoomire/projects/safequant_v2

robust_run() {
    local desc="$1"; shift
    echo ""
    echo "########################################################################"
    echo "# $desc"
    echo "# started: $(date)"
    echo "########################################################################"
    "$@" && echo "DONE: $desc | $(date)" || echo "FAILED ($?): $desc | $(date)"
}

# ── IDENTICAL training + eval budget for read and write (matched-budget control)
# group_size=128 keeps the INT-2 grid coherent (per-channel RTN at 2-bit collapses
# the model — see v1 run). n_xstest adds the over-refusal (false-refusal) check.
BUDGET_ARGS="--steps 3000 --lr 1e-4 --rank 32 --alpha 64 \
             --seq_len 512 --grad_accum 8 --n_train_docs 4000 \
             --group_size 128 --n_eval 200 --n_reasoning 50 --n_xstest 50"

# 1) READ-side INT-2 — the key new row (+ untrained INT-2 PTQ control)
robust_run "LR-QAT read INT-2" \
    python -u experiments/trained_repair/lrqat.py \
        --model llama7b --quantizer aqlm2 --role read --n_bits 2 ${BUDGET_ARGS}

# 2) WRITE-side INT-2 — asymmetry control at the SAME budget
robust_run "LR-QAT write INT-2" \
    python -u experiments/trained_repair/lrqat.py \
        --model llama7b --quantizer aqlm2 --role write --n_bits 2 ${BUDGET_ARGS}

# 3) READ-side INT-4 — recovery-vs-precision sweep point (coherent grid)
robust_run "LR-QAT read INT-4 (sweep)" \
    python -u experiments/trained_repair/lrqat.py \
        --model llama7b --quantizer aqlm2 --role read --n_bits 4 ${BUDGET_ARGS}

# 4) WRITE-side INT-4 — asymmetry control at a COHERENT bit-width, matched budget
robust_run "LR-QAT write INT-4 (asymmetry @ coherent bits)" \
    python -u experiments/trained_repair/lrqat.py \
        --model llama7b --quantizer aqlm2 --role write --n_bits 4 ${BUDGET_ARGS}

echo ""
echo "=== LR-QAT repair Llama complete | $(date) ==="
