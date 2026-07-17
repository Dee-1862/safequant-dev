#!/usr/bin/env python
"""scripts/quantize_gptq.py: Reproducible GPTQ quantization via AutoGPTQ.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, MODEL_REVISIONS, HF_HOME  # noqa: E402
from core.datasets import load_dataset

def default_output_dir(model: str, bits: int, group_size: int) -> Path:
    """
    Save checkpoints on Ocean (next to the HF cache), never in $HOME.

    Inputs:
        - model (str): alias from config.MODELS (llama7b, ...) or a raw HF id
        - bits (int): quantization bits (2, 3, 4, 8)
        - group_size (int): quantization group size (128)

    Outputs:
        - output_dir (Path): path to the output directory
    """
    base = Path(HF_HOME).parent.parent  # /ocean/.../<user>/.cache/hf -> /ocean/.../<user>
    out_dir = base / "quantized_models" / "gptq" / f"{model}-gptq{bits}bit-g{group_size}"
    return out_dir


def build_calibration(tokenizer, n_samples, seqlen, seed):
    """128 x seqlen-token windows sampled from WikiText-2-raw train, exactly the hugging-quants recipe. Deterministic given seed (fixed RNG -> same windows).

    Inputs:
        - tokenizer (Tokenizer): tokenizer object
        - n_samples (int): number of calibration windows
        - seqlen (int): length of each calibration window
        - seed (int): random seed

    Outputs:
        - calib (list): list of calibration windows
    """
    # WikiText-2-raw is version-stable; content-hash reproducible across runs.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    n_tokens = enc.input_ids.shape[1]
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, n_tokens - seqlen - 1)
        ids = enc.input_ids[:, i:i + seqlen]
        samples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return samples


def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Reproducible GPTQ quantization via AutoGPTQ.")
    ap.add_argument("--model", required=True,
                    help="alias from config.MODELS (llama7b, ...) or a raw HF id")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--n_calib", type=int, default=128,
                    help="calibration windows (GPTQ standard = 128)")
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="tokens per calibration window (GPTQ standard = 2048)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    # Get model path and revision
    model_path = MODELS.get(args.model, args.model)
    revision = MODEL_REVISIONS.get(args.model)  # None if not pinned

    # Get output directory
    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.model, args.bits, args.group_size))
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  AutoGPTQ quantize: {args.model}  ({model_path})")
    print(f"  bits={args.bits}  group_size={args.group_size}  "
          f"n_calib={args.n_calib}  seqlen={args.seqlen}  seed={args.seed}")

    # Print model information and output directory
    if revision:
        print(f"  FP-base revision pin: {revision[:12]}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    tok_kwargs = {"use_fast": True, "cache_dir": HF_HOME}
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)

    # Build calibration windows
    print(f"\n[1/3] building {args.n_calib} x {args.seqlen}-token calibration "
          f"from WikiText-2 ...")
    calib = build_calibration(tokenizer, args.n_calib, args.seqlen, args.seed)
    print(f"  built {len(calib)} calibration windows")

    # Load FP base and quantize
    print("\n[2/3] loading FP base + quantizing (AutoGPTQ) ...")
    quantize_config = BaseQuantizeConfig(
        bits = args.bits,
        group_size = args.group_size,
        desc_act = True,
        sym = True,
        damp_percent = 0.1,
    )
    init_kwargs = {"cache_dir": HF_HOME, "trust_remote_code": True,
                   "torch_dtype": torch.float16}

    # Set revision if provided
    if revision:
        init_kwargs["revision"] = revision

    # auto-gptq 0.7.1 predates Phi-3 and errors "phi3 isn't supported yet."
    # Register Phi-3's layer layout (fused qkv_proj / gate_up_proj) so AutoGPTQ
    # quantizes it with the SAME recipe. No-op for other model types.
    try:
        from auto_gptq.modeling._base import BaseGPTQForCausalLM
        from auto_gptq.modeling import _const as _agc, auto as _aga
        if "phi3" not in _agc.SUPPORTED_MODELS:
            class Phi3GPTQForCausalLM(BaseGPTQForCausalLM):
                layer_type = "Phi3DecoderLayer"
                layers_block_name = "model.layers"
                outside_layer_modules = ["model.embed_tokens", "model.norm"]
                inside_layer_modules = [
                    ["self_attn.qkv_proj"],
                    ["self_attn.o_proj"],
                    ["mlp.gate_up_proj"],
                    ["mlp.down_proj"],
                ]
            _agc.SUPPORTED_MODELS.append("phi3")
            _aga.GPTQ_CAUSAL_LM_MODEL_MAP["phi3"] = Phi3GPTQForCausalLM
            print("  registered Phi-3 support in auto-gptq (fused qkv/gate_up)")
    except Exception as _e:  # noqa: BLE001
        print(f"  WARN: phi3 auto-gptq registration failed: {_e}")

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, **init_kwargs)

    # Pin RotaryEmbedding module onto the position_ids device on every forward
    def _pin_rotary(mod, args):
        try:
            # Get the device of the position_ids argument
            dev = args[1].device if len(args) > 1 else args[0].device
            mod.to(dev)
        except Exception:
            pass

    # Register forward pre-hook for RotaryEmbedding modules
    for _name, _mod in model.model.named_modules():
        if _mod.__class__.__name__.endswith("RotaryEmbedding"):
            _mod.register_forward_pre_hook(_pin_rotary)

    # Quantize the model
    model.quantize(calib)

    # Save the quantized model
    print(f"\n[3/3] saving to {out_dir} ...")
    model.save_quantized(str(out_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(out_dir))

    print("\nDONE. Registered in config.QUANTIZED_CHECKPOINTS as:")
    print(f'    ("{args.model}", "gptq{args.bits}"): "{out_dir}",')


if __name__ == "__main__":
    main()
