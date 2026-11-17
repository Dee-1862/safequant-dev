
# Reproducible GPTQ quantization via AutoGPTQ (default) or GPTQModel.
#
# Backend: GPTQ_BACKEND env var or --backend, "autogptq" (default) | "gptqmodel".
#   autogptq   auto-gptq 0.7.1 in the sq_autogptq env ($AGENV). The paper's checkpoints.
#   gptqmodel  gptqmodel in the sq_clean env ($CONDA_ENV). Fallback where auto-gptq
#              cannot run (e.g. if the source build fails on DeltaAI aarch64).
# Both use the same recipe: bits/group_size from the CLI, desc_act=True, sym=True,
# damp 0.1, and the same WikiText-2 calibration windows (build_calibration below, one
# code path for both backends; the sbatch passes 128 x 2048 tokens, seed 42).
#
# The checkpoint is written to <out_dir>.partial-<job> and renamed to <out_dir> only
# after a successful save, so a failed build never leaves a folder that looks complete.
# An existing <out_dir>/config.json is never overwritten. sq_build_info.json in the
# checkpoint folder records the backend, recipe and library versions.
import argparse
import datetime
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Repo root must be on path when invoked as scripts/quantize_gptq.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, MODEL_REVISIONS, HF_HOME, QM_ROOT  # noqa: E402
from core.datasets import load_dataset  # noqa: E402

BACKENDS = ("autogptq", "gptqmodel")
PACK_INFO = {"mode": "backend default", "patched": []}  # filled by quantize_autogptq


def default_output_dir(model, bits, group_size):
    """
    Project-filesystem path for a self-quantized GPTQ checkpoint (never under $HOME).

    Inputs:
        - model (str): Model alias from config.MODELS
        - bits (int): Quantization bit width
        - group_size (int): GPTQ group size

    Outputs:
        - out_dir (Path): Target directory for saved weights
    """
    # Write where config.QUANTIZED_CHECKPOINTS reads: <QM_ROOT>/gptq/<model>-gptq<b>bit-g<g>.
    # (Was HF_HOME/../../quantized_models, which equals QM_ROOT only for the Bridges-2
    # layout; on Delta HF_HOME and QM_ROOT are set independently in .env.)
    return Path(QM_ROOT) / "gptq" / f"{model}-gptq{bits}bit-g{group_size}"


def build_calibration(tokenizer, n_samples, seqlen, seed):
    """
    WikiText-2 calibration windows (hugging-quants recipe), shared by both backends.

    Inputs:
        - tokenizer: HuggingFace tokenizer
        - n_samples (int): Number of windows
        - seqlen (int): Tokens per window
        - seed (int): RNG seed for window starts

    Outputs:
        - samples (list): Dicts with input_ids and attention_mask
    """
    # Load pinned WikiText-2-raw train split
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "train")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors = "pt")

    # Pin RNG before drawing window offsets
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


def recipe(args):
    """
    The GPTQ recipe common to both backends.

    Inputs:
        - args (Namespace): Parsed CLI arguments

    Outputs:
        - r (dict): bits, group_size, desc_act, sym, damp_percent
    """
    return {"bits": args.bits, "group_size": args.group_size, "desc_act": True,
            "sym": True, "damp_percent": 0.1}


def _pkg_version(name):
    """
    Installed version of a distribution, or None.

    Inputs:
        - name (str): Distribution name

    Outputs:
        - version (str | None)
    """
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _git_sha():
    """
    Repository HEAD SHA, or None.

    Inputs:
        - None

    Outputs:
        - sha (str | None)
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text = True, stderr = subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_info(args, backend, model_path, revision, n_calib_built):
    """
    Provenance record written to <checkpoint>/sq_build_info.json.

    Inputs:
        - args (Namespace): Parsed CLI arguments
        - backend (str): "autogptq" | "gptqmodel"
        - model_path (str): HF id of the FP base
        - revision (str | None): FP-base revision pin
        - n_calib_built (int): Calibration windows actually built

    Outputs:
        - info (dict)
    """
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "backend": backend,
        "model": args.model,
        "model_path": model_path,
        "revision": revision,
        "recipe": recipe(args),
        "calibration": {"dataset": "Salesforce/wikitext wikitext-2-raw-v1 train",
                        "n_samples": n_calib_built, "seqlen": args.seqlen,
                        "seed": args.seed},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": _pkg_version("transformers"),
            "numpy": _pkg_version("numpy"),
            "accelerate": _pkg_version("accelerate"),
            "auto_gptq": _pkg_version("auto-gptq") or _pkg_version("auto_gptq"),
            "gptqmodel": _pkg_version("gptqmodel"),
            "optimum": _pkg_version("optimum"),
        },
        "host": {"machine": platform.machine(), "node": platform.node(), "gpu": gpu,
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "packing": dict(PACK_INFO),
        "torch_threads": torch.get_num_threads(),
        "git_sha": _git_sha(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def quantize_autogptq(args, model_path, revision, calib, tmp_dir):
    """
    Quantize and save with auto-gptq 0.7.1 (the paper's backend).

    Inputs:
        - args (Namespace): Parsed CLI arguments
        - model_path (str): HF id of the FP base
        - revision (str | None): FP-base revision pin
        - calib (list): Calibration windows
        - tmp_dir (Path): Directory to save into

    Outputs:
        - None
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    # Vectorized packing (scripts/gptq_fast_pack.py): same arithmetic per element,
    # bit-identical output (scripts/tests/test_gptq_fast_pack.py), minutes faster.
    # GPTQ_FAST_PACK=0 keeps auto-gptq's per-column loop.
    if os.environ.get("GPTQ_FAST_PACK", "1") != "0":
        import gptq_fast_pack
        PACK_INFO["patched"] = gptq_fast_pack.apply()
        PACK_INFO["mode"] = "vectorized (bit-identical)" if PACK_INFO["patched"] else "auto-gptq loop (patch not applicable)"
        print(f"  packing: {PACK_INFO['mode']}  {PACK_INFO['patched']}")
    else:
        PACK_INFO["mode"] = "auto-gptq loop (GPTQ_FAST_PACK=0)"

    quantize_config = BaseQuantizeConfig(**recipe(args))
    init_kwargs = {
        "cache_dir": HF_HOME,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if revision:
        init_kwargs["revision"] = revision

    # Register Phi-3 fused layout for auto-gptq 0.7.1 (no-op on other archs)
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
    except Exception as _e:
        print(f"  WARN: phi3 auto-gptq registration failed: {_e}")

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, **init_kwargs)

    # Pin RotaryEmbedding onto position_ids device (Phi-3 / long-context stability)
    def _pin_rotary(mod, args_hook):
        try:
            dev = args_hook[1].device if len(args_hook) > 1 else args_hook[0].device
            mod.to(dev)
        except Exception:
            pass

    for _name, _mod in model.model.named_modules():
        if _mod.__class__.__name__.endswith("RotaryEmbedding"):
            _mod.register_forward_pre_hook(_pin_rotary)

    import time
    t0 = time.time()
    model.quantize(calib)
    PACK_INFO["quantize_and_pack_seconds"] = round(time.time() - t0, 1)
    print(f"  quantize + pack took {PACK_INFO['quantize_and_pack_seconds']} s")
    print(f"\n[3/3] saving to {tmp_dir} ...")
    model.save_quantized(str(tmp_dir), use_safetensors = True)


def quantize_gptqmodel(args, model_path, revision, calib, tmp_dir):
    """
    Quantize and save with GPTQModel, same recipe and calibration as auto-gptq.

    Inputs:
        - args (Namespace): Parsed CLI arguments
        - model_path (str): HF id of the FP base
        - revision (str | None): FP-base revision pin
        - calib (list): Calibration windows
        - tmp_dir (Path): Directory to save into

    Outputs:
        - None
    """
    import dataclasses
    from gptqmodel import GPTQModel, QuantizeConfig

    fields = {f.name for f in dataclasses.fields(QuantizeConfig)}
    kw = dict(recipe(args))
    missing = [k for k in kw if k not in fields]
    if missing:
        raise RuntimeError(f"gptqmodel QuantizeConfig has no field(s) {missing}; "
                           "cannot reproduce the auto-gptq recipe with this version")
    # Newer gptqmodel defaults would change the recipe: act_group_aware (GAR) replaces
    # plain desc_act ordering and is rejected together with desc_act=True. Keep plain
    # GPTQ act-order and the standard gptq checkpoint format (what transformers loads).
    if "act_group_aware" in fields:
        kw["act_group_aware"] = False
    if "format" in fields:
        kw["format"] = "gptq"
    qcfg = QuantizeConfig(**kw)
    print(f"  gptqmodel QuantizeConfig: {kw}")

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16,
                   "cache_dir": HF_HOME}
    if revision:
        load_kwargs["revision"] = revision
    model = GPTQModel.load(model_path, qcfg, **load_kwargs)
    model.quantize(calib, batch_size = 1)
    print(f"\n[3/3] saving to {tmp_dir} ...")
    model.save(str(tmp_dir))


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser(description = "Reproducible GPTQ quantization.")
    ap.add_argument("--model", required = True,
                    help = "alias from config.MODELS (llama7b, ...) or a raw HF id")
    ap.add_argument("--bits", type = int, default = 4, choices = [2, 3, 4, 8])
    ap.add_argument("--group_size", type = int, default = 128)
    ap.add_argument("--n_calib", type = int, default = 128,
                    help = "calibration windows (GPTQ standard = 128)")
    ap.add_argument("--seqlen", type = int, default = 2048,
                    help = "tokens per calibration window (GPTQ standard = 2048)")
    # 42 = what scripts/quantize_gptq.sbatch has always passed (the paper checkpoints)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--output_dir", default = None)
    ap.add_argument("--backend", default = os.environ.get("GPTQ_BACKEND", "autogptq"),
                    help = "autogptq (default) | gptqmodel; default from $GPTQ_BACKEND")
    args = ap.parse_args()
    backend = args.backend.strip().lower()
    if backend not in BACKENDS:
        ap.error(f"--backend / GPTQ_BACKEND must be one of {BACKENDS}, got {args.backend!r}")

    # Resolve HF id and optional revision pin
    model_path = MODELS.get(args.model, args.model)
    revision = MODEL_REVISIONS.get(args.model)

    # Pick output directory; never overwrite a finished checkpoint
    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.model, args.bits, args.group_size))
    if (out_dir / "config.json").exists():
        print(f"ERROR: checkpoint already exists at {out_dir}; not overwriting. "
              "Move it away first to rebuild.", file = sys.stderr)
        sys.exit(1)
    out_dir.parent.mkdir(parents = True, exist_ok = True)
    tmp_dir = out_dir.with_name(f"{out_dir.name}.partial-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("=" * 72)
    print(f"  GPTQ quantize [{backend}]: {args.model}  ({model_path})")
    print(f"  bits={args.bits}  group_size={args.group_size}  desc_act=True sym=True damp=0.1")
    print(f"  calib: n={args.n_calib}  seqlen={args.seqlen}  seed={args.seed}")
    if revision:
        print(f"  FP-base revision pin: {revision[:12]}")
    print(f"  output: {out_dir}   (staging: {tmp_dir.name})")
    print("=" * 72)

    tok_kwargs = {"use_fast": True, "cache_dir": HF_HOME}
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)

    # Build calibration windows (identical for both backends)
    print(f"\n[1/3] building {args.n_calib} x {args.seqlen}-token calibration "
          f"from WikiText-2 ...")
    calib = build_calibration(tokenizer, args.n_calib, args.seqlen, args.seed)
    print(f"  built {len(calib)} calibration windows")

    # Load FP base, quantize, save to the staging dir
    print(f"\n[2/3] loading FP base + quantizing ({backend}) ...")
    if backend == "autogptq":
        quantize_autogptq(args, model_path, revision, calib, tmp_dir)
    else:
        quantize_gptqmodel(args, model_path, revision, calib, tmp_dir)
    tokenizer.save_pretrained(str(tmp_dir))

    info = build_info(args, backend, model_path, revision, len(calib))
    (tmp_dir / "sq_build_info.json").write_text(json.dumps(info, indent = 2))
    print(f"  backend + versions -> {out_dir / 'sq_build_info.json'}")
    print("  " + json.dumps(info["versions"]))

    # Publish atomically
    if not (tmp_dir / "config.json").exists():
        print(f"ERROR: {tmp_dir} has no config.json after save; leaving it for inspection.",
              file = sys.stderr)
        sys.exit(1)
    if out_dir.exists():
        # a folder without config.json (checked above): move it aside, never delete it
        stale = out_dir.with_name(f"{out_dir.name}.stale-{datetime.datetime.now():%Y%m%d%H%M%S}")
        out_dir.rename(stale)
        print(f"  moved incomplete {out_dir.name} aside -> {stale.name}")
    tmp_dir.rename(out_dir)

    print("\nDONE. Registered in config.QUANTIZED_CHECKPOINTS as:")
    print(f'    ("{args.model}", "gptq{args.bits}"): "{out_dir}",')


if __name__ == "__main__":
    main()
