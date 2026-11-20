# WikiText-2 perplexity across the regime cells.
#
# A continuous coherence / language-modeling-quality metric to complement the
# discrete ARC-Challenge accuracy (addresses the "is ARC a sufficient coherence
# proxy?" review comment). Runs one model's fp32 reference + every quantizer cell
# in a single job; the phi3/awq NA cell is skipped gracefully (status recorded).

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from core.datasets import load_wikitext2
from core.evaluation import measure_perplexity
from core.manifest import RunConfig, assert_new_result, result_path, save_result
from core.model_loader import load_fp32, load_quantized
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _variant(tag):
    # Tagged runs (e.g. --tag rebuilt) get their own file names and never overwrite.
    """
    Output variant for a tag.

    Inputs:
        - tag (str or None): --tag value

    Outputs:
        - variant (str): wikitext2_ppl or wikitext2_ppl_<tag>
    """
    return f"wikitext2_ppl_{tag}" if tag else "wikitext2_ppl"


def _save(model, quantizer, seq_len, ppl, ppl_fp32, status, tag = None, extra = None):
    """
    Write one PPL result JSON.

    Inputs:
        - model (str): Model alias
        - quantizer (str): Quantizer key (fp32 for the reference)
        - seq_len (int): Window length
        - ppl (float or None): Perplexity
        - ppl_fp32 (float or None): fp32 reference perplexity
        - status (str): ok, nan or an error label
        - tag (str or None): Output tag
        - extra (dict or None): Additional fields

    Outputs:
        - path (Path): Saved file
    """
    ratio = (ppl / ppl_fp32) if (ppl is not None and ppl_fp32) else None
    cfg = RunConfig(model = model, quantizer = quantizer, variant = _variant(tag))
    return save_result(_SCRIPT, cfg, {
        "model": model, "quantizer": quantizer, "seq_len": seq_len,
        "wikitext2_ppl": ppl, "wikitext2_ppl_fp32": ppl_fp32,
        "ppl_ratio": ratio, "status": status, "seed": SEED,
        "dataset": "wikitext-2-raw-v1/test",
        **(extra or {}),
    })


def _paper_fp32(model):
    """
    fp32 PPL from the untagged (paper) result, for --no_fp32 runs.

    Inputs:
        - model (str): Model alias

    Outputs:
        - ppl (float or None): fp32 perplexity, None if the file is missing
        - path (str): File read
    """
    p = result_path(_SCRIPT, RunConfig(model = model, quantizer = "fp32", variant = "wikitext2_ppl"))
    try:
        return json.loads(p.read_text()).get("wikitext2_ppl"), str(p)
    except (OSError, ValueError):
        return None, str(p)


def _build_info(model, quantizer):
    """
    sq_build_info.json of a self-built checkpoint (backend and versions), if present.

    Inputs:
        - model (str): Model alias
        - quantizer (str): Quantizer key

    Outputs:
        - info (dict or None): Build record
    """
    try:
        from config import QUANTIZED_CHECKPOINTS
        ck = QUANTIZED_CHECKPOINTS.get((model, quantizer))
        if ck and Path(ck, "sq_build_info.json").exists():
            return json.loads(Path(ck, "sq_build_info.json").read_text())
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="WikiText-2 perplexity per regime cell.")
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizers", nargs="+",
                    default=["awq", "gptq4", "aqlm2", "gptq3", "gptq2"])
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="cap number of PPL windows (smoke/quick-eval only)")
    # Opt-in (defaults unchanged): --tag suffixes every output file name
    # (wikitext_ppl_<m>_<q>_wikitext2_ppl_<tag>.json) and refuses to overwrite;
    # --no_fp32 skips the fp32 measurement and takes the fp32 reference for ppl_ratio
    # from the untagged (paper) fp32 JSON.
    ap.add_argument("--tag", default = None)
    ap.add_argument("--no_fp32", action = "store_true")
    args = ap.parse_args()

    if args.tag:
        cells = ([] if args.no_fp32 else ["fp32"]) + list(args.quantizers)
        for q in cells:
            assert_new_result(_SCRIPT, RunConfig(model = args.model, quantizer = q,
                                                 variant = _variant(args.tag)))

    text = load_wikitext2("test")
    print(f"=== WikiText-2 PPL | {args.model} | seq_len={args.seq_len} | seed={SEED}"
          f"{' | tag=' + args.tag if args.tag else ''} ===")

    if args.no_fp32:
        ppl_fp32, src = _paper_fp32(args.model)
        print(f"  fp32 not measured (--no_fp32); reference from {src}: {ppl_fp32}")
        fp32_extra = {"wikitext2_ppl_fp32_source": src}
    else:
        # FP32 reference (per model)
        fp32_model, tokenizer = load_fp32(args.model)
        ppl_fp32 = measure_perplexity(fp32_model, tokenizer, text, seq_len = args.seq_len,
                                      max_chunks = args.max_chunks, desc = f"{args.model}/fp32")
        print(f"  fp32 PPL = {ppl_fp32:.3f}")
        del fp32_model
        gc.collect()
        torch.cuda.empty_cache()
        _save(args.model, "fp32", args.seq_len, ppl_fp32, ppl_fp32,
              "ok" if math.isfinite(ppl_fp32) else "nan", tag = args.tag)
        fp32_extra = {}

    # Each quantizer cell
    for q in args.quantizers:
        print(f"\n---- {args.model} / {q} ----")
        try:
            qm, tok = load_quantized(args.model, quantizer=q)
            ppl_q = measure_perplexity(qm, tok, text, seq_len=args.seq_len,
                                       max_chunks=args.max_chunks, desc=f"{args.model}/{q}")
            status = "ok" if math.isfinite(ppl_q) else "nan"   # a nan PPL is a failed load, not ok
            rel = f"  (x{ppl_q / ppl_fp32:.3f} vs fp32)" if ppl_fp32 else ""
            print(f"  {q} PPL = {ppl_q:.3f}{rel}")
            del qm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [skip] {args.model}/{q}: {type(e).__name__}: {e}")
            ppl_q, status = None, f"load_failed:{type(e).__name__}"
        extra = dict(fp32_extra)
        if args.tag:
            extra["checkpoint_build"] = _build_info(args.model, q)
        path = _save(args.model, q, args.seq_len, ppl_q, ppl_fp32, status,
                     tag = args.tag, extra = extra)
        print(f"  saved {path}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
