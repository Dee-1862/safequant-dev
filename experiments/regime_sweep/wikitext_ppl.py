# WikiText-2 perplexity across the regime cells.
#
# A continuous coherence / language-modeling-quality metric to complement the
# discrete ARC-Challenge accuracy (addresses the "is ARC a sufficient coherence
# proxy?" review comment). Runs one model's fp32 reference + every quantizer cell
# in a single job; the phi3/awq NA cell is skipped gracefully (status recorded).

import argparse
import gc
from pathlib import Path

import torch

from core.datasets import load_wikitext2
from core.evaluation import measure_perplexity
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _save(model, quantizer, seq_len, ppl, ppl_fp32, status):
    ratio = (ppl / ppl_fp32) if (ppl is not None and ppl_fp32) else None
    cfg = RunConfig(model=model, quantizer=quantizer, variant="wikitext2_ppl")
    return save_result(_SCRIPT, cfg, {
        "model": model, "quantizer": quantizer, "seq_len": seq_len,
        "wikitext2_ppl": ppl, "wikitext2_ppl_fp32": ppl_fp32,
        "ppl_ratio": ratio, "status": status, "seed": SEED,
        "dataset": "wikitext-2-raw-v1/test",
    })


def main():
    ap = argparse.ArgumentParser(description="WikiText-2 perplexity per regime cell.")
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizers", nargs="+",
                    default=["awq", "gptq4", "aqlm2", "gptq3", "gptq2"])
    ap.add_argument("--seq_len", type=int, default=2048)
    args = ap.parse_args()

    text = load_wikitext2("test")
    print(f"=== WikiText-2 PPL | {args.model} | seq_len={args.seq_len} | seed={SEED} ===")

    # FP32 reference (per model)
    fp32_model, tokenizer = load_fp32(args.model)
    ppl_fp32 = measure_perplexity(fp32_model, tokenizer, text, seq_len=args.seq_len,
                                  desc=f"{args.model}/fp32")
    print(f"  fp32 PPL = {ppl_fp32:.3f}")
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()
    _save(args.model, "fp32", args.seq_len, ppl_fp32, ppl_fp32, "ok")

    # Each quantizer cell
    for q in args.quantizers:
        print(f"\n---- {args.model} / {q} ----")
        try:
            qm, tok = load_quantized(args.model, quantizer=q)
            ppl_q = measure_perplexity(qm, tok, text, seq_len=args.seq_len,
                                       desc=f"{args.model}/{q}")
            status = "ok"
            print(f"  {q} PPL = {ppl_q:.3f}  (x{ppl_q / ppl_fp32:.3f} vs fp32)")
            del qm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [skip] {args.model}/{q}: {type(e).__name__}: {e}")
            ppl_q, status = None, f"load_failed:{type(e).__name__}"
        path = _save(args.model, q, args.seq_len, ppl_q, ppl_fp32, status)
        print(f"  saved {path}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
