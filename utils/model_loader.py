# CPU-compatible model loading for SafeQuant experiments.
#
# The main gotcha with BitsAndBytes on CPU (v0.49.2+):
#   - NF4 works fine, but you MUST use bnb_4bit_compute_dtype=torch.float32
#     because CPU hardware doesn't support float16 matrix multiplication
#   - Calling .dequantize() on a Params4bit weight returns shape (numel, 1)
#     instead of (out_features, in_features) - this is a known BNB CPU bug
#   - The workaround is the identity-matrix trick in get_fp32_weight() below

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import DTYPE, NF4_CONFIG


def load_model(model_name, precision, device="cpu"):
    """Load a model in the specified precision, works on CPU with no GPU required.

    precision: "nf4" loads with BitsAndBytes 4-bit, anything else loads as float32.
    Returns (model, tokenizer) with model in eval mode.
    """
    if precision == "nf4":
        # always force float32 compute - float16 compute crashes on CPU
        cpu_nf4_config = {**NF4_CONFIG, "bnb_4bit_compute_dtype": torch.float32}
        bnb_config = BitsAndBytesConfig(**cpu_nf4_config)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        # float32 because CPU doesn't have float16 matmul
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


def get_fp32_weight(module):
    """Extract the weight of a Linear or Linear4bit layer as a float32 tensor.

    Returns shape (out_features, in_features).

    We do this via a forward pass through an identity matrix instead of
    calling .dequantize() directly, because BitsAndBytes on CPU returns
    the wrong shape from .dequantize() - it gives (numel, 1) instead of (out, in).
    The identity-matrix trick forces BNB to do the dequantization internally
    and gives back a correctly shaped result.
    """
    in_features = module.in_features
    identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        output = module(identity)   # (in_features, out_features)
    return output.float().T.contiguous()  # (out_features, in_features)
