import os
# Force gptqmodel to use pure-pytorch backend (no triton/exllama kernel issues)
os.environ["GPTQMODEL_FORCE_BACKEND"] = "torch"
# Disable torch.compile which causes PY_SSIZE_T_CLEAN errors on older torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS, DEVICE, HF_HOME, QUANTIZED_CHECKPOINTS
try:
    from config import MODEL_REVISIONS
except ImportError:
    MODEL_REVISIONS = {}


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve a model name to a HuggingFace model path.

    Accepts either:
    - A short alias defined in config.MODELS (e.g. 'qwen1.5b')
    - A direct HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-1B-Instruct')
    - A local path to a model directory
    """
    return MODELS.get(model_name, model_name)


def _use_remote_code(model_path: str) -> bool:
    """Phi-3's bundled modeling code calls DynamicCache.seen_tokens (removed in
    transformers >=4.41); use transformers' native Phi-3 instead. All other
    models keep trust_remote_code=True as before."""
    p = str(model_path).lower()
    return "phi-3" not in p and "phi3" not in p


def load_fp32(model_name: str, device: str = None):
    """
    Load model in FP32 precision.

    Args:
        model_name: Short alias (e.g. 'qwen1.5b') OR any HuggingFace model ID
                    (e.g. 'meta-llama/Llama-3.2-1B-Instruct') OR a local path.
        device: Device to load on (default: config.DEVICE)

    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = DEVICE

    model_path = _resolve_model_path(model_name)

    print(f"Loading FP32 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    rev = MODEL_REVISIONS.get(model_name)
    if rev:
        print(f"  Revision pin: {rev[:12]} (FP weights matched to the quantizer's source)")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=_use_remote_code(model_path),
        cache_dir=HF_HOME,
        revision=rev,
    )

    print(f"  OK Loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    print(f"  Cache: {HF_HOME}")

    return model, tokenizer


def load_nf4(model_name: str, device: str = None):
    """
    Load model in NF4 quantization (bitsandbytes 4-bit NormalFloat).

    Args:
        model_name: Short alias (e.g. 'qwen1.5b') OR any HuggingFace model ID
                    (e.g. 'meta-llama/Llama-3.2-1B-Instruct') OR a local path.
        device: Device to load on (default: config.DEVICE)

    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = DEVICE

    model_path = _resolve_model_path(model_name)

    print(f"Loading NF4 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # Use float32 compute on CPU, float16 on GPU
    compute_dtype = torch.float32 if device == "cpu" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )

    print(f"  OK Loaded in 4-bit NF4 (compute dtype: {compute_dtype})")
    print(f"  Cache: {HF_HOME}")

    return model, tokenizer


def load_int8(model_name: str, device: str = None):
    """
    Load model in INT8 quantization (bitsandbytes 8-bit).
    Less aggressive than NF4 -- useful for ablation: does 8-bit also collapse?

    Args:
        model_name: Short alias or any HF model ID.
        device: Device to load on (default: config.DEVICE)

    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = DEVICE

    model_path = _resolve_model_path(model_name)

    print(f"Loading INT8 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )

    print(f"  OK Loaded in 8-bit INT8")
    print(f"  Cache: {HF_HOME}")

    return model, tokenizer


def load_quantized(model_name: str, quantizer: str = "nf4", device: str = None):
    """
    Unified loader that dispatches to the right quantizer.

    Args:
        model_name: Short alias or any HF model ID.
        quantizer: One of:
            "fp32"   - no quantization (baseline)
            "nf4"    - bitsandbytes 4-bit NormalFloat (default)
            "int8"   - bitsandbytes 8-bit
            "awq"    - AWQ 4-bit (pre-quantized checkpoint)
            "gptq4"  - GPTQ 4-bit (pre-quantized checkpoint)
            "gptq3"  - GPTQ 3-bit (on-the-fly quantization)
            "gptq2"  - GPTQ 2-bit (on-the-fly quantization)
            "gptq8"  - GPTQ 8-bit (pre-quantized checkpoint)
            "gptq"   - alias for gptq4
        device: Device override.

    Returns:
        (model, tokenizer)
    """
    quantizer = quantizer.lower()

    if quantizer == "fp32":
        return load_fp32(model_name, device)
    elif quantizer == "nf4":
        return load_nf4(model_name, device)
    elif quantizer == "int8":
        return load_int8(model_name, device)
    elif quantizer == "awq":
        return _load_awq(model_name, device)
    elif quantizer in ("gptq", "gptq4"):
        return _load_gptq(model_name, device, bits=4)
    elif quantizer == "gptq3":
        return _load_gptq(model_name, device, bits=3)
    elif quantizer == "gptq2":
        return _load_gptq(model_name, device, bits=2)
    elif quantizer == "gptq8":
        return _load_gptq(model_name, device, bits=8)
    elif quantizer == "aqlm2":
        return _load_aqlm(model_name, device, bits=2)
    elif quantizer == "aqlm3":
        return _load_aqlm(model_name, device, bits=3)
    elif quantizer in ("quip2", "quip4"):
        bits = 2 if quantizer == "quip2" else 4
        return _load_quip(model_name, device, bits=bits)
    else:
        raise ValueError(
            f"Unknown quantizer: '{quantizer}'. "
            f"Choose from: fp32, nf4, int8, awq, gptq4, gptq3, gptq2, gptq8, aqlm2, aqlm3, quip2, quip4"
        )


def _load_awq(model_name: str, device: str = None):
    """
    Load a pre-quantized AWQ model via transformers native AWQ support.

    The AWQ GEMM kernel in transformers 4.47.1 has a PY_SSIZE_T_CLEAN bug
    with Python 3.11. We work around it by dequantizing AWQ layers to FP16
    nn.Linear modules after loading, which also makes weight extraction and
    module-swap restoration work seamlessly.
    """
    if device is None:
        device = DEVICE

    ckpt_key = (model_name, "awq")
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AWQ model: {model_name} -> {model_path}")
    else:
        model_path = _resolve_model_path(model_name)
        print(f"Loading AWQ model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME,
                                               trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        cache_dir=HF_HOME,
        torch_dtype=torch.float16,
    )

    model.eval()
    print(f"  OK Loaded AWQ 4-bit model")
    return model, tokenizer


def _dequantize_awq_inplace(model):
    """
    Replace all AWQ WQLinear modules with standard nn.Linear by unpacking
    the INT4 weights using AWQ's own dequantization utilities.

    This avoids the PY_SSIZE_T_CLEAN kernel bug (which affects both
    forward() and generate()) and makes weight extraction / module-swap
    restoration work seamlessly.
    """
    import torch.nn as nn

    # Collect AWQ layers -- detect by class name OR attributes
    replacements = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        is_awq = (
            "WQLinear" in cls_name
            or (hasattr(module, 'qweight') and hasattr(module, 'scales')
                and hasattr(module, 'qzeros'))
        )
        if is_awq:
            replacements.append((name, module))

    if not replacements:
        print(f"  Warning: no AWQ layers found to dequantize")
        return

    print(f"  Found {len(replacements)} AWQ layers, dequantizing...")

    count = 0
    for name, module in replacements:
        try:
            in_f = module.in_features
            out_f = module.out_features
            has_bias = module.bias is not None
            # AWQ exllama-v2 stores data as buffers, not parameters
            device = module.qweight.device
            w_bit = getattr(module, 'w_bit', 4)
            group_size = getattr(module, 'group_size', 128)

            # Pure-torch AWQ INT4 unpacking (no kernel dependencies)
            qweight = module.qweight  # (in_features // pack_factor, out_features), int32
            qzeros = module.qzeros    # (in_features // group_size // pack_factor, out_features), int32
            scales = module.scales    # (in_features // group_size, out_features), float16

            # Unpack INT4 from INT32: each int32 packs 8 x 4-bit values
            shifts = torch.arange(0, 32, w_bit, device=device)
            iweight = torch.bitwise_right_shift(
                qweight[:, :, None], shifts[None, None, :]
            ).to(torch.int8)
            iweight = iweight.view(iweight.shape[0], -1)  # unpack columns
            iweight = torch.bitwise_and(iweight, (2 ** w_bit) - 1)

            izeros = torch.bitwise_right_shift(
                qzeros[:, :, None], shifts[None, None, :]
            ).to(torch.int8)
            izeros = izeros.view(izeros.shape[0], -1)
            izeros = torch.bitwise_and(izeros, (2 ** w_bit) - 1)

            # Reverse AWQ interleave order
            reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]
            num_groups = iweight.shape[-1] // (32 // w_bit)
            reverse_idx = torch.arange(iweight.shape[-1], device=device, dtype=torch.int32)
            reverse_idx = reverse_idx.view(-1, 32 // w_bit)[:, reverse_order].reshape(-1)
            iweight = iweight[:, reverse_idx]
            izeros = izeros[:, reverse_idx]

            # Dequantize: weight = (iweight - izeros) * scales
            scales_expanded = scales.repeat_interleave(group_size, dim=0)
            izeros_expanded = izeros.repeat_interleave(group_size, dim=0)
            weight = ((iweight.float() - izeros_expanded.float()) * scales_expanded.float()).T

            new_linear = nn.Linear(in_f, out_f, bias=has_bias, device=device,
                                   dtype=torch.float16)
            new_linear.weight = nn.Parameter(weight.to(device).to(torch.float16),
                                             requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(module.bias.data.clone(),
                                               requires_grad=False)

            # Replace in model
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_linear)
            count += 1
        except Exception as e:
            print(f"  Warning: failed to dequantize {name}: {type(e).__name__}: {e}")
            continue

    print(f"  Dequantized {count}/{len(replacements)} AWQ layers to FP16 nn.Linear")


def _load_aqlm(model_name: str, device: str = None, bits: int = 2):
    """
    Load an AQLM-quantized model (SOTA for 2-bit).

    Checks for pre-quantized checkpoint first. AQLM models load natively
    through transformers -- just needs `pip install aqlm[gpu]`.

    Args:
        model_name: model alias or HF ID
        device: target device
        bits: 2 or 3
    """
    if device is None:
        device = DEVICE

    bit_suffix = f"aqlm{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AQLM-{bits}bit model: {model_name} -> {model_path}")
    else:
        # Check local aqlm_cache
        safe_name = model_name.replace("/", "_")
        local_path = os.path.join(HF_HOME, "aqlm_cache", f"{safe_name}-aqlm-{bits}bit")
        if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            model_path = local_path
            print(f"Loading AQLM-{bits}bit model: {model_name} (cached at {local_path})")
        else:
            raise ValueError(
                f"No pre-quantized AQLM checkpoint for ({model_name}, {bit_suffix}). "
                f"AQLM requires pre-quantized checkpoints -- on-the-fly quantization "
                f"takes hours. Add the checkpoint to config.QUANTIZED_CHECKPOINTS."
            )

    # AQLM checkpoints often lack a chat template, so load tokenizer from
    # the base model to get the full chat template and special tokens.
    base_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=HF_HOME,
                                               trust_remote_code=_use_remote_code(base_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=_use_remote_code(model_path),
        cache_dir=HF_HOME,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  OK Loaded AQLM {bits}-bit model")
    return model, tokenizer


def _load_quip(model_name: str, device: str = None, bits: int = 4):
    """
    Load a pre-quantized QuIP# model.
    These use custom weight formats but some checkpoints work through transformers.
    Tokenizer is loaded from the base model.
    """
    if device is None:
        device = DEVICE

    bit_suffix = f"quip{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading QuIP#-{bits}bit model: {model_name} -> {model_path}")
    else:
        raise ValueError(f"No pre-quantized QuIP# checkpoint for ({model_name}, {bit_suffix})")

    # Tokenizer from base model (QuIP# repos don't include tokenizer)
    base_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=HF_HOME,
                                               trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        cache_dir=HF_HOME,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  OK Loaded QuIP# {bits}-bit model")
    return model, tokenizer


def _load_gptq(model_name: str, device: str = None, bits: int = 4):
    """
    Load a GPTQ-quantized model.

    For 4-bit and 8-bit: loads pre-quantized HF checkpoint via GPTQConfig.
    For 3-bit and 2-bit: quantizes on-the-fly (requires optimum + auto-gptq).

    Args:
        model_name: model alias or HF ID
        device: target device
        bits: quantization bits (2, 3, 4, or 8)
    """
    if device is None:
        device = DEVICE

    from transformers import GPTQConfig

    # Check for pre-quantized checkpoint
    bit_suffix = {4: "gptq4", 3: "gptq3", 8: "gptq8"}.get(bits)
    ckpt_key_alias = (model_name, bit_suffix) if bit_suffix else None

    pre_quantized = None
    if ckpt_key_alias and ckpt_key_alias in QUANTIZED_CHECKPOINTS:
        pre_quantized = QUANTIZED_CHECKPOINTS[ckpt_key_alias]

    if pre_quantized:
        # Decoding Compressed Trust models use revision branches (e.g. "4bit_128g")
        revision = None
        if "compressed-llm" in pre_quantized:
            revision = f"{bits}bit_128g"
            print(f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized} (rev: {revision})")
        else:
            print(f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized}")

        # Load via transformers native GPTQ support (GPU dequantization, no
        # extra CUDA/triton kernels needed -- avoids auto_gptq version conflicts)
        if "compressed-llm" in pre_quantized:
            base_path = _resolve_model_path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=HF_HOME,
                                                       trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pre_quantized, cache_dir=HF_HOME, trust_remote_code=True,
                revision=revision,
            )
        gptq_cfg = GPTQConfig(bits=bits, use_exllama=False)
        model = AutoModelForCausalLM.from_pretrained(
            pre_quantized,
            quantization_config=gptq_cfg,
            device_map=device,
            trust_remote_code=True,
            cache_dir=HF_HOME,
            torch_dtype=torch.float16,
            revision=revision,
        )
        model.eval()
        print(f"  OK Loaded GPTQ {bits}-bit (pre-quantized)")
        return model, tokenizer

    # On-the-fly quantization for 3-bit and 2-bit
    # Check if we already cached a quantized version on Ocean
    model_path = _resolve_model_path(model_name)
    safe_name = model_name.replace("/", "_")
    cache_path = os.path.join(HF_HOME, "gptq_cache", f"{safe_name}-gptq-{bits}bit")

    if os.path.isdir(cache_path) and os.path.exists(os.path.join(cache_path, "config.json")):
        print(f"Loading GPTQ-{bits}bit model: {model_name} (cached at {cache_path})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
        except Exception as e:
            print(f"  cache-path tokenizer load failed ({type(e).__name__}); loading from base HF repo {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        gptq_cfg = GPTQConfig(bits=bits, use_exllama=False)
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            quantization_config=gptq_cfg,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        print(f"  OK Loaded GPTQ {bits}-bit from cache")
        return model, tokenizer

    # Quantize from scratch
    print(f"Loading GPTQ-{bits}bit model: {model_name} (on-the-fly quantization)")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME,
                                               trust_remote_code=True)

    # Load calibration data as a list of strings to avoid optimum trying to
    # download allenai/c4 at runtime (fails on compute nodes without internet,
    # and the dataset= string param doesn't reliably propagate to optimum).
    from datasets import load_dataset as _load_ds
    try:
        _wiki = _load_ds("wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=HF_HOME)
        calib_texts = [t for t in _wiki["text"] if len(t.strip()) > 100][:128]
    except Exception:
        _wiki = _load_ds("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [t for t in _wiki["text"] if len(t.strip()) > 100][:128]

    gptq_config = GPTQConfig(
        bits=bits,
        dataset=calib_texts,
        tokenizer=tokenizer,
        group_size=128,
        use_exllama=False,
    )

    print(f"  Quantizing to {bits}-bit GPTQ (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map=device,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )
    model.eval()
    print(f"  OK Quantized to GPTQ {bits}-bit on-the-fly")

    # Save to cache for future runs
    try:
        os.makedirs(cache_path, exist_ok=True)
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path)
        print(f"  OK Saved to cache: {cache_path}")
    except Exception as e:
        print(f"  Warning: could not cache quantized model: {e}")

    return model, tokenizer


def get_weight(module, as_dtype=torch.float32):
    """
    Extract weight from a module, handling both regular and quantized weights.

    Args:
        module: nn.Linear module (possibly quantized)
        as_dtype: Target dtype for output

    Returns:
        Weight tensor of shape (out_features, in_features)
    """
    import torch.nn as nn

    # Get dimensions -- different quantizers use different attribute names
    in_f = getattr(module, 'in_features', None) or getattr(module, 'infeatures', None)
    out_f = getattr(module, 'out_features', None) or getattr(module, 'outfeatures', None)

    # Check if it's an auto_gptq QuantLinear
    if hasattr(module, 'qweight'):
        # auto_gptq QuantLinear -- use identity matrix forward pass to dequantize
        device = module.qweight.device
        identity = torch.eye(in_f, device=device, dtype=torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Check if it's a regular nn.Linear or bitsandbytes quantized
    if not isinstance(module, nn.Linear):
        # Unknown module type -- try identity matrix forward pass as fallback
        device = next(module.parameters()).device
        identity = torch.eye(in_f, device=device, dtype=torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Check if quantized (BitsAndBytes Params4bit -- NF4)
    if hasattr(module.weight, 'quant_type'):
        try:
            import bitsandbytes.functional as bnb_F
            weight = bnb_F.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).to(torch.float32).view(out_f, in_f)
        except Exception:
            identity = torch.eye(in_f, device=module.weight.device, dtype=torch.float32)
            with torch.no_grad():
                weight = module(identity).T
        return weight.to(as_dtype)

    # Check if quantized (BitsAndBytes Int8Params -- LLM.int8)
    if hasattr(module.weight, 'CB') or module.weight.dtype == torch.int8:
        # Int8 stores weights as int8 + row-wise absmax scales
        # Use identity matrix forward pass to get dequantized weight
        device = module.weight.device
        identity = torch.eye(in_f, device=device, dtype=torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Regular weight
    return module.weight.data.to(as_dtype)


if __name__ == "__main__":
    # Test loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen0.5b",
                        help="Short alias (qwen0.5b/qwen1.5b/qwen3b) or any HF model ID")
    parser.add_argument("--precision", choices=["fp32", "nf4"], default="fp32")
    args = parser.parse_args()

    if args.precision == "fp32":
        model, tokenizer = load_fp32(args.model)
    else:
        model, tokenizer = load_nf4(args.model)

    print(f"\nOK Model loaded successfully")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
