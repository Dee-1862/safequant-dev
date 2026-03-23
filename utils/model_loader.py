"""
utils/model_loader.py

CPU-compatible model loading for SafeQuant experiments.

Key facts about BitsAndBytes on CPU (v0.49.2+):
  - NF4 quantization works on CPU with bnb_4bit_compute_dtype=torch.float32
  - float16 compute dtype FAILS on CPU (no float16 matmul support)
  - .dequantize() on CPU returns shape (numel, 1) instead of (out, in)
    → use get_fp32_weight() which uses an identity-matrix forward pass instead

Usage:
    from utils.model_loader import load_model, get_fp32_weight
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import DTYPE, NF4_CONFIG


def load_model(model_name: str, precision: str, device: str = "cpu"):
    """
    Load model in specified precision, fully CPU-compatible.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
        precision:  "nf4" for BitsAndBytes 4-bit, anything else for full fp32.
        device:     "cpu" (default). Pass "cuda" only if you have a GPU.

    Returns:
        (model, tokenizer) — model is in eval mode.
    """
    if precision == "nf4":
        # Override compute dtype to float32 — float16 fails on CPU
        cpu_nf4_config = {**NF4_CONFIG, "bnb_4bit_compute_dtype": torch.float32}
        bnb_config = BitsAndBytesConfig(**cpu_nf4_config)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        # CPU doesn't support float16 matrix multiplications; use float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_fp32_weight(module: nn.Module) -> torch.Tensor:
    """
    Extract the weight matrix of a Linear (or Linear4bit) layer as a plain
    float32 tensor of shape (out_features, in_features).

    Why not just .weight.data?
      BitsAndBytes on CPU stores packed 4-bit weights.  Calling
      param.dequantize() returns shape (numel, 1) instead of (out, in).
      The identity-matrix trick forces BitsAndBytes to dequantize internally
      and returns a correctly shaped result.

    Args:
        module: nn.Linear or bitsandbytes.nn.Linear4bit layer.

    Returns:
        Float32 tensor, shape (out_features, in_features), on CPU.
    """
    in_features = module.in_features
    identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        output = module(identity)          # (in_features, out_features)
    return output.float().T.contiguous()   # (out_features, in_features)
