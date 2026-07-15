#!/usr/bin/env python
"""
scripts/quantize_gptq.py -- Reproducible GPTQ quantization of the study's base
models via GPTQModel (ModelCloud, the maintained successor to auto-gptq), so
every GPTQ cell shares ONE controlled recipe instead of mismatched third-party
checkpoints (ModelCloud / TheBloke / Qwen ...).

Fixed recipe (the community-STANDARD GPTQ operating point, pinned explicitly for
reproducibility -- record these in the paper):
  - bits / group_size from CLI (default 4 / 128).
  - desc_act=True (act-order), sym=True, damp_percent=0.05: standard settings used
    by shipped GPTQ checkpoints (ModelCloud/TheBloke). act-order is the dominant
    quality lever -- pinned ON explicitly, never left to a library default.
  - calibration: 128 windows of 2048 tokens concatenated from a DETERMINISTIC C4
    slice (streams the same documents every run -- no random sampling). Full-length
    token windows give a well-conditioned Hessian.
  - --seed fixes torch/numpy/python RNG. The GPTQ algorithm itself is
    deterministic given fixed calibration data (Hessian -> Cholesky -> sequential
    error-feedback), so calibration set + settings fully determine the output.
    Use --deterministic for bit-exact (pins cuBLAS + torch deterministic kernels).
  - FP base pinned to config.MODEL_REVISIONS where set (e.g. phi3), so the GPTQ
    weights come from the SAME source revision the AQLM checkpoints used.

Output loads through core.model_loader._load_gptq (transformers-native GPTQ) and
is compatible with diagnostics/gptq_scale_safety_tune.py (auto-gptq layout).

Usage:
  python scripts/quantize_gptq.py --model phi3 --bits 4 --group_size 128 --seed 42
  # then paste the printed line into config.QUANTIZED_CHECKPOINTS

Requires:  pip install gptqmodel     (pin the version -- see MODELS_QUANTIZATION.md)
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, MODEL_REVISIONS, HF_HOME  # noqa: E402


def load_c4_texts(min_chars, cache_dir=None):
    """Inlined C4 streamer (same deterministic slice + revision pin as
    core.datasets.load_c4_calibration) so this script does NOT import
    core.datasets -- which pulls in jailbreakbench/pandas and would force them
    into the isolated quantization env. Returns raw documents (concatenated into
    fixed-length token windows by build_calibration); streams the same first
    documents every run -- no random sampling. Only needs the `datasets` pkg."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True,
                      revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                      cache_dir=cache_dir)
    out, total = [], 0
    for doc in ds:
        text = (doc.get("text") or "").strip()
        if len(text) < 100:
            continue
        out.append(text)
        total += len(text)
        if total >= min_chars:
            break
    return out


def build_calibration(tokenizer, texts, n_samples, seqlen):
    """Standard GPTQ calibration: concatenate the corpus and slice it into
    `n_samples` windows of exactly `seqlen` tokens (the auto-gptq / GPTQModel
    convention). Full 2048-token rows give a well-conditioned Hessian; short
    ~500-token rows under-estimate it and degrade the quantization -- especially
    for subtle features like the refusal direction. Deterministic: the same
    C4 slice always yields the same windows."""
    ids = []
    need = (n_samples + 1) * seqlen
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    samples = []
    for i in range(n_samples):
        chunk = ids[i * seqlen:(i + 1) * seqlen]
        if len(chunk) < seqlen:
            break
        samples.append({"input_ids": chunk, "attention_mask": [1] * seqlen})
    return samples


def set_determinism(seed: int, strict: bool = False):
    """Fix all RNGs. In strict mode also pin cuBLAS + deterministic kernels for
    bit-exact reproducibility (slower; may error on unsupported ops)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        # CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def default_output_dir(model: str, bits: int, group_size: int) -> Path:
    """Save checkpoints on Ocean (next to the HF cache), never in $HOME."""
    base = Path(HF_HOME).parent.parent  # /ocean/.../<user>/.cache/hf -> /ocean/.../<user>
    return base / "quantized_models" / f"{model}-gptq{bits}bit-g{group_size}"


def main():
    ap = argparse.ArgumentParser(
        description="Reproducible GPTQ quantization via GPTQModel.")
    ap.add_argument("--model", required=True,
                    help="alias from config.MODELS (llama7b, phi3, mistral7b-v02, "
                         "qwen7b, ...) or a raw HF id")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--n_calib", type=int, default=128,
                    help="calibration samples (GPTQ standard = 128)")
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="tokens per calibration sample (GPTQ standard = 2048)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true",
                    help="bit-exact mode (torch deterministic algorithms + cuBLAS "
                         "workspace pin; slower, may fail on unsupported ops)")
    ap.add_argument("--output_dir", default=None,
                    help="save location (default: "
                         "<ocean>/quantized_models/<model>-gptq<bits>bit-g<gs>)")
    args = ap.parse_args()

    set_determinism(args.seed, strict=args.deterministic)

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError:
        sys.exit("[ERROR] gptqmodel not installed. Run:  pip install gptqmodel")

    model_path = MODELS.get(args.model, args.model)
    revision = MODEL_REVISIONS.get(args.model)  # None if not pinned
    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.model, args.bits, args.group_size))
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  GPTQ quantize: {args.model}  ({model_path})")
    print(f"  bits={args.bits}  group_size={args.group_size}  "
          f"n_calib={args.n_calib}  seed={args.seed}"
          f"{'  [deterministic]' if args.deterministic else ''}")
    if revision:
        print(f"  FP-base revision pin: {revision[:12]} (matches AQLM source weights)")
    print(f"  output: {out_dir}")
    print("=" * 72)

    # 1) load FP base under the pinned STANDARD GPTQ recipe.
    print("\n[1/3] loading FP base ...")
    # Pin every setting explicitly -- do NOT inherit library defaults, which
    # drift between gptqmodel versions AND (as we found) can default to a
    # NON-standard operating point. These are the community-standard values used
    # by distributed GPTQ checkpoints (ModelCloud/TheBloke), so our GPTQ cells
    # reflect how GPTQ is actually deployed, not a handicapped variant:
    #   desc_act=True   -> activation-order quantization. THE dominant quality
    #     lever and the mainstream setting (GPTQ paper + auto-gptq/GPTQModel
    #     recommend it; ~all shipped GPTQ checkpoints use it). Our earlier
    #     desc_act=False run drifted the refusal direction to cos 0.51 vs the
    #     act-order ModelCloud checkpoint's 0.98 -- an artifact, now removed.
    #   sym=True, damp_percent=0.05 -> match what our prior run already used
    #     (gptqmodel 7.1.0 defaults) so act-order + calibration length are the
    #     ONLY intentional changes. It's a generic quality recipe, not tuned
    #     toward (or against) the refusal direction.
    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=True,
        sym=True,
        damp_percent=0.05,
    )
    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    model = GPTQModel.load(model_path, quant_config, **load_kwargs)

    # 2) deterministic calibration + quantize. Concatenate a fixed C4 slice into
    #    n_calib windows of exactly seqlen tokens (same windows every run) -- the
    #    standard GPTQ calibration, replacing the earlier ~500-token snippets.
    print(f"\n[2/3] building {args.n_calib} x {args.seqlen}-token calibration + "
          f"quantizing (Hessian -> GPTQ) ...")
    raw = load_c4_texts(min_chars=(args.n_calib + 2) * args.seqlen * 6,
                        cache_dir=HF_HOME)
    calib = build_calibration(model.tokenizer, raw, args.n_calib, args.seqlen)
    print(f"  built {len(calib)} calibration windows of {args.seqlen} tokens")
    model.quantize(calib, batch_size=1)

    # 3) save checkpoint + tokenizer
    print(f"\n[3/3] saving to {out_dir} ...")
    model.save(str(out_dir))
    try:
        model.tokenizer.save_pretrained(str(out_dir))
    except Exception:
        pass

    # Keep the checkpoint SELF-CONTAINED and loadable under transformers 4.x.
    # gptqmodel runs under transformers 5, which stamps tokenizer_class
    # "TokenizersBackend" into tokenizer_config.json -- a class transformers 4.x
    # cannot import. Strip only that metadata field; the checkpoint's own
    # tokenizer.json (identical to the base, version-stable) then loads via the
    # inferred fast tokenizer. The checkpoint keeps ITS OWN tokenizer, not the base's.
    import json as _json
    _tcfg = out_dir / "tokenizer_config.json"
    if _tcfg.exists():
        _d = _json.loads(_tcfg.read_text())
        if _d.pop("tokenizer_class", None) is not None:
            _tcfg.write_text(_json.dumps(_d, indent=2))
            print("  patched tokenizer_config.json (dropped transformers-5 "
                  "'tokenizer_class' -> transformers 4.x loads the checkpoint tokenizer)")

    print("\nDONE. Register it in config.QUANTIZED_CHECKPOINTS:")
    print(f'    ("{args.model}", "gptq{args.bits}"): "{out_dir}",')


if __name__ == "__main__":
    main()
