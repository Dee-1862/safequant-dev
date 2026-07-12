import os

# Force gptqmodel to use pure-pytorch backend (no triton/exllama kernel issues)
os.environ["GPTQMODEL_FORCE_BACKEND"] = "torch"
# Disable torch.compile which causes PY_SSIZE_T_CLEAN errors on older torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import sys
from transformers import GPTQConfig
from datasets import load_dataset as _load_ds

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

    Input:
    - model_name (str): the name of the model to resolve

    Output:
    - model_path (str): the path to the model
    """
    return MODELS.get(model_name, model_name)


def _use_remote_code(model_path: str) -> bool:
    """
    Phi-3's bundled modeling code calls DynamicCache.seen_tokens (removed in transformers >=4.41); use transformers' native Phi-3 instead. All other models keep trust_remote_code=True as before.

    Input:
    - model_path (str): the path to the model

    Output:
    - use_remote_code (bool): whether to use remote code
    """
    # Get the path of the model
    p = str(model_path).lower()
    return "phi-3" not in p and "phi3" not in p


def load_fp32(model_name: str, device: str = None):
    """
    Load model in FP32 precision.

    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the path of the model
    model_path = _resolve_model_path(model_name)

    print(f"Loading FP32 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # Get the revision of the model
    rev = MODEL_REVISIONS.get(model_name)
    if rev:
        print(f"  Revision pin: {rev[:12]} (FP weights matched to the quantizer's source)")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=HF_HOME, revision=rev
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=_use_remote_code(model_path),
        cache_dir=HF_HOME,
        revision=rev,
    )

    print(
        f"  OK Loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)"
    )
    print(f"  Cache: {HF_HOME}")

    return model, tokenizer


def load_nf4(model_name: str, device: str = None):
    """
    Load model in NF4 quantization (bitsandbytes 4-bit NormalFloat).

    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the path of the model
    model_path = _resolve_model_path(model_name)

    print(f"Loading NF4 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # Use float32 compute on CPU, float16 on GPU
    compute_dtype = torch.float32 if device == "cpu" else torch.float16

    # Get the BitsAndBytes configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load the tokenizer and the model
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
    Less aggressive than NF4 and is useful for ablation: does 8-bit also collapse?

    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the path of the model
    model_path = _resolve_model_path(model_name)

    print(f"Loading INT8 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # Get the BitsAndBytes configuration
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = bnb_config,
        device_map = device,
        trust_remote_code = True,
        cache_dir = HF_HOME,
    )

    print(f"  OK Loaded in 8-bit INT8")
    print(f"  Cache: {HF_HOME}")

    return model, tokenizer


def load_quantized(model_name: str, quantizer: str = "nf4", device: str = None):
    """
    Unified loader that dispatches to the right quantizer.

    Input:
        model_name: Short alias or any HF model ID.
    - quantizer (str): the quantizer to use
    - device (str): the device to load the model on (default: DEVICE)

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
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
        return _load_gptq(model_name, device, bits = 4)
    elif quantizer == "gptq3":
        return _load_gptq(model_name, device, bits = 3)
    elif quantizer == "gptq2":
        return _load_gptq(model_name, device, bits = 2)
    elif quantizer == "aqlm2":
        return _load_aqlm(model_name, device, bits = 2)
    elif quantizer == "aqlm3":
        return _load_aqlm(model_name, device, bits = 3)
    elif quantizer in ("quip2", "quip4"):
        bits = 2 if quantizer == "quip2" else 4
        return _load_quip(model_name, device, bits=bits)
    else:
        raise ValueError(
            f"Unknown quantizer: '{quantizer}'. "
            f"Choose from: fp32, nf4, int8, awq, gptq4, gptq3, gptq2, aqlm2, aqlm3, quip2, quip4"
        )


def _load_awq(model_name: str, device: str = None):
    """
    Load a pre-quantized AWQ model via transformers native AWQ support.

    The AWQ GEMM kernel in transformers 4.47.1 has a PY_SSIZE_T_CLEAN bug with Python 3.11. We work around it by dequantizing AWQ layers to FP16 nn.Linear modules after loading, which also makes weight extraction and module-swap restoration work seamlessly.

    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the checkpoint key
    ckpt_key = (model_name, "awq")
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        # Get the model path
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AWQ model: {model_name} -> {model_path}")
    else:
        model_path = _resolve_model_path(model_name)
        print(f"Loading AWQ model: {model_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=HF_HOME, trust_remote_code=True
    )

    # Load the model
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


def _load_aqlm(model_name: str, device: str = None, bits: int = 2):
    """
    Load an AQLM-quantized model (SOTA for 2-bit).

    Checks for pre-quantized checkpoint first. AQLM models load natively through transformers and just needs `pip install aqlm[gpu]`.

    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)
    - bits (int): the number of bits to quantize the model to

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the bit suffix
    bit_suffix = f"aqlm{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        # Get the model path
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AQLM-{bits}bit model: {model_name} -> {model_path}")
    else:
        # Check local aqlm_cache
        safe_name = model_name.replace("/", "_")
        local_path = os.path.join(HF_HOME, "aqlm_cache", f"{safe_name}-aqlm-{bits}bit")

        # Check if the local path exists
        if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            # Get the model path
            model_path = local_path
            # Print the model path
            print(f"Loading AQLM-{bits}bit model: {model_name} (cached at {local_path})")
        else:
            # Raise an error
            raise ValueError(
                f"No pre-quantized AQLM checkpoint for ({model_name}, {bit_suffix}). "
                f"AQLM requires pre-quantized checkpoints -- on-the-fly quantization "
                f"takes hours. Add the checkpoint to config.QUANTIZED_CHECKPOINTS."
            )

    # AQLM checkpoints often lack a chat template, so load tokenizer from the base model to get the full chat template and special tokens.
    base_path = _resolve_model_path(model_name)

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=HF_HOME, trust_remote_code=_use_remote_code(base_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device,
        trust_remote_code = _use_remote_code(model_path),
        cache_dir=HF_HOME,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  OK Loaded AQLM {bits}-bit model")
    return model, tokenizer


def _load_quip(model_name: str, device: str = None, bits: int = 4):
    """
    Load a pre-quantized QuIP# model.
    
    Input:
    - model_name (str): the name of the model to load
    - device (str): the device to load the model on (default: DEVICE)
    - bits (int): the number of bits to quantize the model to

    Output:
    - model (torch.nn.Module): the loaded model
    - tokenizer (transformers.Tokenizer): the tokenizer for the model
    """
    # Get the device
    if device is None:
        device = DEVICE

    # Get the bit suffix and checkpoint key
    bit_suffix = f"quip{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading QuIP#-{bits}bit model: {model_name} -> {model_path}")
    else:
        raise ValueError(f"No pre-quantized QuIP# checkpoint for ({model_name}, {bit_suffix})")

    # Tokenizer from base model (QuIP# repos don't include tokenizer)
    base_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, cache_dir=HF_HOME, trust_remote_code=True
    )

    # Load the model
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
    # Get the device
    if device is None:
        device = DEVICE

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
            print(
                f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized} (rev: {revision})"
            )
        else:
            print(f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized}")

        # Load via transformers native GPTQ support
        if "compressed-llm" in pre_quantized:
            base_path = _resolve_model_path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                base_path, cache_dir=HF_HOME, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pre_quantized,
                cache_dir=HF_HOME,
                trust_remote_code=True,
                revision=revision,
            )

        # Create the GPTQ configuration
        gptq_cfg = GPTQConfig(bits=bits, use_exllama=False)
        model = AutoModelForCausalLM.from_pretrained(
            pre_quantized,
            quantization_config = gptq_cfg,
            device_map = device,
            trust_remote_code = True,
            cache_dir = HF_HOME,
            torch_dtype = torch.float16,
            revision = revision,
        )
        model.eval()
        print(f"  OK Loaded GPTQ {bits}-bit (pre-quantized)")
        return model, tokenizer

    # On-the-fly quantization for 3-bit and 2-bit
    # Check if we already cached a quantized version on Ocean
    model_path = _resolve_model_path(model_name)
    safe_name = model_name.replace("/", "_")
    cache_path = os.path.join(HF_HOME, "gptq_cache", f"{safe_name}-gptq-{bits}bit")

    if os.path.isdir(cache_path) and os.path.exists(
        os.path.join(cache_path, "config.json")
    ):
        print(f"Loading GPTQ-{bits}bit model: {model_name} (cached at {cache_path})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cache_path, trust_remote_code=True
            )
        except Exception as e:
            print(
                f"  cache-path tokenizer load failed ({type(e).__name__}); loading from base HF repo {model_path}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=HF_HOME, trust_remote_code=True
    )

    # Load calibration data as a list of strings to avoid optimum trying to download allenai/c4 at runtime (fails on compute nodes without internet, and the dataset= string param doesn't reliably propagate to optimum).
    try:
        _wiki = _load_ds(
            "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=HF_HOME
        )
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

    # Quantize the model
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
    in_f = getattr(module, "in_features", None) or getattr(module, "infeatures", None)
    out_f = getattr(module, "out_features", None) or getattr(
        module, "outfeatures", None
    )

    # Check if it's an auto_gptq QuantLinear
    if hasattr(module, "qweight"):
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
    if hasattr(module.weight, "quant_type"):
        try:
            import bitsandbytes.functional as bnb_F

            weight = (
                bnb_F.dequantize_4bit(
                    module.weight.data,
                    module.weight.quant_state,
                )
                .to(torch.float32)
                .view(out_f, in_f)
            )
        except Exception:
            identity = torch.eye(in_f, device=module.weight.device, dtype=torch.float32)
            with torch.no_grad():
                weight = module(identity).T
        return weight.to(as_dtype)

    # Check if quantized (BitsAndBytes Int8Params -- LLM.int8)
    if hasattr(module.weight, "CB") or module.weight.dtype == torch.int8:
        # Int8 stores weights as int8 + row-wise absmax scales
        # Use identity matrix forward pass to get dequantized weight
        device = module.weight.device
        identity = torch.eye(in_f, device=device, dtype=torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Regular weight
    return module.weight.data.to(as_dtype)
