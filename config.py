# SafeQuant model aliases, HF cache paths, and quantized checkpoint registry.
#Front Matter
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """Load environment variables from `.env` at the repo root."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding = "utf-8") as fh:
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
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)


_load_dotenv()

# Set HF_HOME and QM_ROOT in .env (see .env.example). Laptop defaults if unset.
HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
QM_ROOT = os.environ.get("QM_ROOT") or os.path.join(os.path.expanduser("~"), "quantized_models")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("QM_ROOT", QM_ROOT)

_QM = QM_ROOT
# Short aliases -> HuggingFace model IDs (paper grid: llama, mistral, qwen, phi3).
MODELS = {
    "llama7b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# Pin FP bases to the weight revision each quant checkpoint was built from.
# phi3 needs ff07dc01 because Microsoft updated the repo after the AQLM quant run.
MODEL_REVISIONS = {
    "phi3": "ff07dc01615f8113924aed013115ab2abd32115b",
}

# Pre-quantized checkpoint aliases: (base_model_alias, quantizer) -> HuggingFace path.
# Self-generated GPTQ lives under $QM_ROOT/gptq/ (see scripts/quantize_gptq.py).
QUANTIZED_CHECKPOINTS = {
    ("llama7b", "gptq4"): f"{_QM}/gptq/llama7b-gptq4bit-g128",
    ("llama7b", "gptq3"): f"{_QM}/gptq/llama7b-gptq3bit-g128",
    ("llama7b", "gptq2"): f"{_QM}/gptq/llama7b-gptq2bit-g128",
    ("llama7b", "awq"): f"{_QM}/awq/llama7b",
    ("llama7b", "aqlm2"): f"{_QM}/aqlm/llama7b",
    ("mistral7b-v02", "gptq4"): f"{_QM}/gptq/mistral7b-v02-gptq4bit-g128",
    ("mistral7b-v02", "gptq3"): f"{_QM}/gptq/mistral7b-v02-gptq3bit-g128",
    ("mistral7b-v02", "gptq2"): f"{_QM}/gptq/mistral7b-v02-gptq2bit-g128",
    ("mistral7b-v02", "awq"): f"{_QM}/awq/mistral7b-v02",
    ("mistral7b-v02", "aqlm2"): f"{_QM}/aqlm/mistral7b-v02",
    ("qwen7b", "gptq4"): f"{_QM}/gptq/qwen7b-gptq4bit-g128",
    ("qwen7b", "gptq3"): f"{_QM}/gptq/qwen7b-gptq3bit-g128",
    ("qwen7b", "gptq2"): f"{_QM}/gptq/qwen7b-gptq2bit-g128",
    ("qwen7b", "awq"): f"{_QM}/awq/qwen7b",
    ("phi3", "gptq4"): f"{_QM}/gptq/phi3-gptq4bit-g128",
    ("phi3", "gptq3"): f"{_QM}/gptq/phi3-gptq3bit-g128",
    ("phi3", "gptq2"): f"{_QM}/gptq/phi3-gptq2bit-g128",
    ("phi3", "awq"): f"{_QM}/awq/phi3",
    ("phi3", "aqlm2"): f"{_QM}/aqlm/phi3",
}

# fall back to cpu if no gpu is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# how many prompts to use at each stage (SafeQuant: n = 1000 for both ASR and ARC)
N_TRAIN = 128         # used for direction extraction (Arditi protocol)
N_VAL = 32            # used for direction selection and validation
N_EVAL = 1000         # WildJailbreak prompts for ASR evaluation
N_ARC = 1000          # ARC-Challenge prompts for capability evaluation
MAX_NEW_TOKENS = 256  # cap on generation length during ASR measurement

# bitsandbytes NF4 config - note: float32 compute dtype is required on CPU.
# float16 compute fails on CPU because there is no hardware float16 matmul support.
NF4_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float32,
    "bnb_4bit_use_double_quant": True,
}
