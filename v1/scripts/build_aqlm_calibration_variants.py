import argparse
import os
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_HOME

# Inlined from the old top-level config.REFUSAL_PREFIXES (removed 2026-07).
# v1-only: cheap substring filter for calibration-set construction, never
# used to score results. Main-pipeline ASR is judged by the HarmBench
# classifier in core/evaluation.py.
REFUSAL_PREFIXES = [
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "As an",
    "As an AI",
    "I'm an",
    "I'm just",
    "As a language model",
    "As an Assistant",
    "I cannot",
    "I can't",
    "I do not",
    "I don't",
    "is illegal and unethical",
    "I'm really sorry",
    "unethical",
    "not ethical",
    "not legal",
    "My apologies",
    "I'm not able to",
    "I am really sorry",
    "I cannot create",
    "I cannot support",
    "I can't provide",
    "I can't assist",
    "I am not able to",
    "I must respectfully",
    "I'm unable to",
]

# Light-touch refusal patterns extracted from the local REFUSAL_PREFIXES above.
# Match case-insensitive substring; sequences containing any of these are
# stripped from the refusal-stripped corpus. Same patterns are wrapped into
# templated refusals for refusal-enriched.
REFUSAL_PATTERNS_LOWER = [p.lower() for p in REFUSAL_PREFIXES]


def load_c4_text(n_docs):
    """Stream C4 docs until we have n_docs non-empty entries."""
    print(f"Loading C4 (streaming) ...")
    ds = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        cache_dir=HF_HOME,
    )
    docs = []
    for ex in ds:
        t = ex.get("text", "").strip()
        if len(t) > 200:
            docs.append(t)
        if len(docs) >= n_docs:
            break
    print(f"  collected {len(docs)} C4 docs")
    return docs


def filter_refusal_substrings(docs):
    """Drop docs containing any of the refusal-pattern substrings."""
    kept = []
    dropped = 0
    for d in docs:
        dl = d.lower()
        if any(p in dl for p in REFUSAL_PATTERNS_LOWER):
            dropped += 1
        else:
            kept.append(d)
    print(f"  refusal-stripped filter: kept {len(kept)} / dropped {dropped}")
    return kept


def load_advbench_prompts(n):
    print(f"Loading AdvBench prompts ...")
    ds = load_dataset("walledai/AdvBench", cache_dir=HF_HOME, split="train")
    prompts = [ex.get("prompt") for ex in ds][:n]
    prompts = [p for p in prompts if p]
    print(f"  collected {len(prompts)} AdvBench prompts")
    return prompts


REFUSAL_TEMPLATES = [
    "I'm sorry, but I cannot help with that request. It involves potentially harmful or illegal activity, and I am not able to provide assistance.",
    "I cannot and will not provide information or instructions on that topic. If you have other questions I can help with, I'm happy to.",
    "I'm not able to assist with that. The request involves activity that could cause real-world harm, so I have to decline.",
    "Sorry, that's not something I can help with. Let me know if there's a different question I can answer.",
    "I cannot provide that information as it could be used for harmful purposes. I'd be glad to help with a different question.",
]


def build_refusal_pairs(prompts, rng):
    """Format harmful prompt + templated refusal as a single training string."""
    pairs = []
    for p in prompts:
        refusal = rng.choice(REFUSAL_TEMPLATES)
        text = (
            f"User: {p}\n\n"
            f"Assistant: {refusal}\n\n"
        )
        pairs.append(text)
    return pairs


def tokenize_to_chunks(texts, tokenizer, seqlen, nsamples, rng):
    """Concatenate texts and slice into nsamples non-overlapping chunks of
    seqlen tokens. Returns list[Tensor[1, seqlen]]."""
    print(f"  tokenizing concatenated corpus ({sum(len(t) for t in texts)/1e6:.1f} MB) ...")
    big = "\n\n".join(texts)
    enc = tokenizer(big, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]   # [T]
    T = ids.shape[0]
    print(f"  tokenized: {T} tokens")
    if T < seqlen * nsamples:
        # Fallback: sample with replacement (overlapping slices).
        print(f"  WARNING: corpus too small ({T} < {seqlen}*{nsamples}); sampling with overlap")
    chunks = []
    for _ in range(nsamples):
        if T - seqlen <= 0:
            i = 0
        else:
            i = rng.randint(0, T - seqlen - 1)
        chunks.append(ids[i:i + seqlen].unsqueeze(0).clone())
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--nsamples", type=int, default=1024,
                    help="sequences per variant (AQLM default is 1024)")
    ap.add_argument("--seqlen", type=int, default=4096,
                    help="tokens per sequence (AQLM default 4096)")
    ap.add_argument("--n_refusal_pairs", type=int, default=200,
                    help="refusal pairs to interleave for refusal-enriched")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"  AQLM calibration variants builder")
    print(f"  model_id: {args.model_id}")
    print(f"  nsamples: {args.nsamples}  seqlen: {args.seqlen}")
    print(f"  output_dir: {out_dir}")
    print("=" * 70)

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=HF_HOME)

    # Need enough docs to fill nsamples x seqlen tokens. Average web doc is
    # ~500 tokens, so request 5x for slack.
    n_docs_needed = max(2000, 5 * args.nsamples * args.seqlen // 500)
    base_docs = load_c4_text(n_docs_needed)

    # -- Variant 1: baseline (raw C4) --------------------------------------
    print(f"\n[variant 1/3] baseline ...")
    chunks = tokenize_to_chunks(base_docs, tokenizer, args.seqlen, args.nsamples, rng)
    out = out_dir / "baseline.pt"
    torch.save(chunks, out)
    print(f"  saved {len(chunks)} chunks -> {out}")

    # -- Variant 2: refusal-stripped ----------------------------------------
    print(f"\n[variant 2/3] refusal-stripped ...")
    stripped_docs = filter_refusal_substrings(base_docs)
    if len(stripped_docs) < 100:
        print(f"  WARNING: only {len(stripped_docs)} docs after stripping; "
              f"variant will be near-identical to baseline by random subsetting")
    chunks = tokenize_to_chunks(stripped_docs, tokenizer, args.seqlen, args.nsamples, rng)
    out = out_dir / "refusal_stripped.pt"
    torch.save(chunks, out)
    print(f"  saved {len(chunks)} chunks -> {out}")

    # -- Variant 3: refusal-enriched ----------------------------------------
    print(f"\n[variant 3/3] refusal-enriched ...")
    advbench = load_advbench_prompts(args.n_refusal_pairs)
    refusal_pairs = build_refusal_pairs(advbench, rng)
    print(f"  built {len(refusal_pairs)} refusal pairs")
    enriched_docs = base_docs + refusal_pairs * 3   # triple them so they appear
    rng.shuffle(enriched_docs)
    chunks = tokenize_to_chunks(enriched_docs, tokenizer, args.seqlen, args.nsamples, rng)
    out = out_dir / "refusal_enriched.pt"
    torch.save(chunks, out)
    print(f"  saved {len(chunks)} chunks -> {out}")

    print(f"\nOK All three calibration variants saved to {out_dir}")


if __name__ == "__main__":
    main()
