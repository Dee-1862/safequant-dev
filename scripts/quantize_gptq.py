#!/usr/bin/env python
"""scripts/quantize_gptq.py -- Reproducible GPTQ quantization via AutoGPTQ.

We self-generate every GPTQ cell (no reliance on HF-hosted pre-quantized models)
under ONE fixed recipe. AutoGPTQ is the toolchain behind the reputable
hugging-quants / TheBloke checkpoints; using it (instead of gptqmodel, which gave
an anomalous refusal-direction cosine ~0.47) makes our self-generated GPTQ-4
match the literature (>0.97). gptq3 / gptq2 use the SAME recipe at lower bits and
give the genuine low-bit degradation curve.

Recipe (pinned -- record in the paper), identical to hugging-quants:
  - AutoGPTQ BaseQuantizeConfig: bits from CLI, group_size 128, desc_act=True,
    sym=True, damp_percent=0.1.
  - Calibration: 128 windows of 2048 tokens sampled (seed-fixed) from
    WikiText-2-raw train, concatenated -- exactly the published recipe.
  - FP base pinned to config.MODEL_REVISIONS where set.
Only the FP base comes from HF; the quantized checkpoint is produced locally.

Runs in the sq_clean env (has auto-gptq + transformers 4.57). Loads back through
core.model_loader._load_gptq.

Usage:
  python scripts/quantize_gptq.py --model llama7b --bits 4 --group_size 128 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, MODEL_REVISIONS, HF_HOME  # noqa: E402


def default_output_dir(model: str, bits: int, group_size: int) -> Path:
    """Save checkpoints on Ocean (next to the HF cache), never in $HOME."""
    base = Path(HF_HOME).parent.parent  # /ocean/.../<user>/.cache/hf -> /ocean/.../<user>
    return base / "quantized_models" / f"{model}-gptq{bits}bit-g{group_size}"


def build_calibration(tokenizer, n_samples, seqlen, seed):
    """128 x seqlen-token windows sampled from WikiText-2-raw train, exactly the
    hugging-quants recipe. Deterministic given seed (fixed RNG -> same windows)."""
    from datasets import load_dataset

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

    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    model_path = MODELS.get(args.model, args.model)
    revision = MODEL_REVISIONS.get(args.model)  # None if not pinned
    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.model, args.bits, args.group_size))
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  AutoGPTQ quantize: {args.model}  ({model_path})")
    print(f"  bits={args.bits}  group_size={args.group_size}  "
          f"n_calib={args.n_calib}  seqlen={args.seqlen}  seed={args.seed}")
    if revision:
        print(f"  FP-base revision pin: {revision[:12]}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    tok_kwargs = {"use_fast": True, "cache_dir": HF_HOME}
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)

    print(f"\n[1/3] building {args.n_calib} x {args.seqlen}-token calibration "
          f"from WikiText-2 ...")
    calib = build_calibration(tokenizer, args.n_calib, args.seqlen, args.seed)
    print(f"  built {len(calib)} calibration windows")

    print("\n[2/3] loading FP base + quantizing (AutoGPTQ) ...")
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=True,
        sym=True,
        damp_percent=0.1,
    )
    init_kwargs = {"cache_dir": HF_HOME, "trust_remote_code": True,
                   "torch_dtype": torch.float16}
    if revision:
        init_kwargs["revision"] = revision
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, **init_kwargs)

    # transformers >=4.43 keeps a single model-level RotaryEmbedding whose inv_freq
    # buffer can stay on CPU while auto-gptq runs each layer on GPU -> a
    # "cpu and cuda:0" device mismatch inside apply_rotary_pos_emb. Pin any
    # RotaryEmbedding module onto the position_ids device on every forward.
    def _pin_rotary(mod, args):
        try:
            dev = args[1].device if len(args) > 1 else args[0].device
            mod.to(dev)
        except Exception:
            pass

    for _name, _mod in model.model.named_modules():
        if _mod.__class__.__name__.endswith("RotaryEmbedding"):
            _mod.register_forward_pre_hook(_pin_rotary)

    model.quantize(calib)

    print(f"\n[3/3] saving to {out_dir} ...")
    model.save_quantized(str(out_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(out_dir))

    print("\nDONE. Registered in config.QUANTIZED_CHECKPOINTS as:")
    print(f'    ("{args.model}", "gptq{args.bits}"): "{out_dir}",')


if __name__ == "__main__":
    main()
