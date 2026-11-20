# SafeQuant model aliases, HF cache paths, and quantized checkpoint registry.
import os

import torch

# project root is wherever this file lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# -- .env + paths -------------------------------------------------------------
def _load_dotenv():
    """Load .env from the repo root (never overrides variables already set in the shell)."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = os.path.expandvars(val.strip().strip('"').strip("'"))
            if key:
                os.environ.setdefault(key, val)


_load_dotenv()

# Set HF_HOME and QM_ROOT in .env (see .env.example). Laptop defaults if unset.
HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
# QUANTIZED_MODELS is the legacy name for QM_ROOT; still honoured.
QM_ROOT = (
    os.environ.get("QM_ROOT")
    or os.environ.get("QUANTIZED_MODELS")
    or os.path.join(os.path.expanduser("~"), "quantized_models")
)
# Tell HuggingFace libraries where to cache -- must be set before any import
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("QM_ROOT", QM_ROOT)

# Consolidated local checkpoint tree:
#   <_QM>/base/<alias>/  <_QM>/awq/<alias>/  <_QM>/aqlm/<alias>/
#   <_QM>/gptq/<alias>-gptq<bits>bit-g128/   (self-generated GPTQ)
# Populated by scripts/fetch_checkpoints.sh + scripts/quantize_gptq.py.
_QM = QM_ROOT

# -- Models -------------------------------------------------------------------
# Short aliases -> HuggingFace model IDs.
# Any HF model ID can also be passed directly (no alias needed).
MODELS = {
    "llama7b":  "meta-llama/Llama-3.1-8B-Instruct",
    "qwen7b":   "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",
    # Phi-3-mini (fused-projection architecture: qkv_proj / gate_up_proj)
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    # Llama-3-8B (held-out AQLM-2 screen)
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# FP-base revision pins. Some HF repos updated their WEIGHTS after a pre-quantized
# checkpoint was made, so HEAD no longer matches the quantizer's source weights.
# Pin the FP base to the revision the AQLM checkpoint was quantized from, else the
# read/write restoration injects mismatched weights (both_restored != FP32).
#   phi3: AQLM created 2024-05-29; microsoft updated weights on 2024-07-01
#         ("Update model #77"). ff07dc01 (2024-06-11) is the last original-weight commit.
MODEL_REVISIONS = {
    "phi3": "ff07dc01615f8113924aed013115ab2abd32115b",
}

# Pre-quantized checkpoint aliases (for multi-quantizer comparison)
# These map (base_model_alias, quantizer) -> HuggingFace checkpoint ID
QUANTIZED_CHECKPOINTS = {
    # GPTQ: self-generated via scripts/quantize_gptq.py using AutoGPTQ (bits/group 128,
    # desc_act, sym, damp 0.1, WikiText-2 calib n=128 x 2048, seed 42 via
    # scripts/quantize_gptq.sbatch -- the hugging-quants recipe). AutoGPTQ (not gptqmodel)
    # makes gptq4 match the literature (cos >0.97); gptq3/gptq2 give the low-bit
    # degradation curve under the SAME recipe. See docs/MODELS_QUANTIZATION.md.
    # Llama-3.1-8B-Instruct
    ("llama7b", "gptq4"):  f"{_QM}/gptq/llama7b-gptq4bit-g128",
    ("llama7b", "gptq3"):  f"{_QM}/gptq/llama7b-gptq3bit-g128",
    ("llama7b", "gptq2"):  f"{_QM}/gptq/llama7b-gptq2bit-g128",
    ("llama7b", "awq"):    f"{_QM}/awq/llama7b",   # hugging-quants/…-AWQ-INT4
    ("llama7b", "aqlm2"):  f"{_QM}/aqlm/llama7b",  # ISTA-DASLab/…-AQLM-PV-2Bit-1x16-hf
    # Qwen2.5-7B-Instruct (no AQLM-2 cell by design)
    ("qwen7b", "gptq4"):   f"{_QM}/gptq/qwen7b-gptq4bit-g128",
    ("qwen7b", "gptq3"):   f"{_QM}/gptq/qwen7b-gptq3bit-g128",
    ("qwen7b", "gptq2"):   f"{_QM}/gptq/qwen7b-gptq2bit-g128",
    ("qwen7b", "awq"):     f"{_QM}/awq/qwen7b",             # Qwen/Qwen2.5-7B-Instruct-AWQ
    # Mistral-7B-Instruct-v0.2 (matches Kharinaev et al. 2025)
    ("mistral7b-v02", "gptq4"): f"{_QM}/gptq/mistral7b-v02-gptq4bit-g128",
    ("mistral7b-v02", "gptq3"): f"{_QM}/gptq/mistral7b-v02-gptq3bit-g128",
    ("mistral7b-v02", "gptq2"): f"{_QM}/gptq/mistral7b-v02-gptq2bit-g128",
    ("mistral7b-v02", "awq"):   f"{_QM}/awq/mistral7b-v02",   # TheBloke/…-v0.2-AWQ
    ("mistral7b-v02", "aqlm2"): f"{_QM}/aqlm/mistral7b-v02",  # ISTA-DASLab/…-v0.2-AQLM-2Bit-2x8
    # Phi-3-mini-4k-instruct (AQLM-PV 2-bit: fused qkv/gate_up projections)
    ("phi3", "gptq4"): f"{_QM}/gptq/phi3-gptq4bit-g128",
    ("phi3", "gptq3"): f"{_QM}/gptq/phi3-gptq3bit-g128",
    ("phi3", "gptq2"): f"{_QM}/gptq/phi3-gptq2bit-g128",
    ("phi3", "awq"):   f"{_QM}/awq/phi3",             # Sreenington/Phi-3-mini-4k-instruct-AWQ
    ("phi3", "aqlm2"): f"{_QM}/aqlm/phi3",            # ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf
    # Llama-3-8B-Instruct (held-out AQLM-2 screen)
    ("llama3-8b", "aqlm2"): "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
}

# fall back to cpu if no gpu is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
