#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# F3a — AQLM-2 quantization of Llama-3.1-8B-Instruct with custom calibration.
#
# Runs the AQLM main.py entry-point on ONE calibration variant. The variant
# is selected via the VARIANT env var:
#     baseline           — pass --dataset c4 (AQLM's own loader)
#     refusal_stripped   — pass --dataset $CALIB_DIR/refusal_stripped.pt
#     refusal_enriched   — pass --dataset $CALIB_DIR/refusal_enriched.pt
#
# Submit one job per variant:
#     CALIB_DIR=/ocean/projects/cis240116p/bhoomire/aqlm_calib_variants
#     for v in baseline refusal_stripped refusal_enriched; do
#         sbatch --export=ALL,VARIANT=$v,CALIB_DIR=$CALIB_DIR \
#             scripts/quantize_aqlm_llama_calibvar.sbatch
#     done
#
# Quantization config matches the ISTA-DASLab Meta-Llama-3.1-8B-Instruct-AQLM
# 2Bit-1x16 setting (num_codebooks=1, nbits_per_codebook=16, in_group_size=8)
# minus the PV-tuning step. Per-variant intra-block finetune is on
# (finetune_max_epochs=5, AQLM default).
#
# Walltime: AQLM 2-bit on Llama-7B ≈ 8-12h on H100-80 with offload_activations.
# Output (raw AQLM format): /ocean/projects/cis240116p/bhoomire/.cache/huggingface/aqlm_cache/llama7b_calib_<VARIANT>-aqlm-2bit-raw
# After conversion to HF format (separate step): same path without -raw suffix.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sq2_aqlm_quant
#SBATCH --account=cis250287p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=14:00:00
#SBATCH --mem=128G
#SBATCH --output=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/aqlm_quant_%j.out
#SBATCH --error=/jet/home/bhoomire/projects/safequant_v2/scripts/logs/aqlm_quant_%j.err

# Use the dedicated AQLM env (NOT sq_clean — AQLM pins transformers==4.40.1).
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/aqlm_quant
module load gcc/10.2.0

export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export CUDA_HOME=/opt/packages/cuda/v12.4.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

if [ -z "${VARIANT:-}" ]; then
    echo "ERROR: VARIANT env var is required (baseline | refusal_stripped | refusal_enriched)"
    exit 1
fi
if [ -z "${CALIB_DIR:-}" ]; then
    export CALIB_DIR=/ocean/projects/cis240116p/bhoomire/aqlm_calib_variants
fi

echo "=== Job ${SLURM_JOB_ID} | AQLM-2 quantize Llama-7B / VARIANT=${VARIANT} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${VARIANT}" in
    baseline)
        DATASET_ARG="c4"
        ;;
    refusal_stripped|refusal_enriched)
        DATASET_ARG="${CALIB_DIR}/${VARIANT}.pt"
        if [ ! -f "${DATASET_ARG}" ]; then
            echo "ERROR: calibration file ${DATASET_ARG} not found. "
            echo "Run scripts/build_aqlm_calibration_variants.py first."
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unknown VARIANT='${VARIANT}'"
        exit 1
        ;;
esac

OUT_DIR=${HF_HOME}/aqlm_cache/llama7b_calib_${VARIANT}-aqlm-2bit-raw
mkdir -p "$(dirname "${OUT_DIR}")"
echo "  dataset:      ${DATASET_ARG}"
echo "  output (raw): ${OUT_DIR}"

cd /jet/home/bhoomire/projects/AQLM

python -u main.py \
    meta-llama/Llama-3.1-8B-Instruct \
    "${DATASET_ARG}" \
    --nsamples=1024 \
    --val_size=128 \
    --num_codebooks=1 \
    --nbits_per_codebook=16 \
    --in_group_size=8 \
    --relative_mse_tolerance=0.01 \
    --finetune_max_epochs=5 \
    --finetune_early_stop=3 \
    --finetune_batch_size=1 \
    --finetune_lr=1e-5 \
    --offload_activations \
    --save="${OUT_DIR}" \
    --seed=42

echo "=== AQLM-2 quantize VARIANT=${VARIANT} complete | $(date) ==="
echo "Next step: convert raw AQLM checkpoint to HF format (see "
echo "AQLM/convert_to_hf.py); save to llama7b_calib_${VARIANT}-aqlm-2bit/."
