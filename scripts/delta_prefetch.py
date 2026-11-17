#!/usr/bin/env python
"""Pre-download the HF models and datasets the reviewer runs need (no GPU).

    python scripts/delta_prefetch.py           # download everything, exit 1 on any failure
    python scripts/delta_prefetch.py --list    # print what would be fetched, download nothing

Models go to HF_HOME with the same cache layout `from_pretrained(cache_dir=HF_HOME)`
uses (core/model_loader.py, core/evaluation.py), with config.MODEL_REVISIONS pins:
  - llama3-8b      FP parent + ("llama3-8b", "aqlm2")      (held-out check)
  - mistral-nemo   FP parent + ("mistral-nemo", "aqlm2")   (held-out fallback)
  - the HarmBench judge used by core.evaluation.classify_batch
Datasets are warmed by calling the repo's own loaders (pinned revisions):
  WildJailbreak (train + eval), XSTest, HarmBench, Alpaca (harmless), ARC-Challenge.
The base/AWQ/AQLM checkpoints under QM_ROOT are fetched separately by
scripts/fetch_checkpoints.sh.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same exclusions as scripts/fetch_checkpoints.sh (no original/consolidated/gguf duplicates)
IGNORE = ["original/*", "*.gguf", "*.pth", "consolidated*"]
HELDOUT_ALIASES = ("llama3-8b", "mistral-nemo")
JUDGE_ID = "cais/HarmBench-Mistral-7b-val-cls"   # must equal core.evaluation._CLS_MODEL_ID


def model_targets():
    """[(label, repo_id, revision)] for every model to pre-download."""
    import config
    out = []
    for alias in HELDOUT_ALIASES:
        out.append((f"{alias} (FP)", config.MODELS[alias], config.MODEL_REVISIONS.get(alias)))
        out.append((f"{alias} (aqlm2)", config.QUANTIZED_CHECKPOINTS[(alias, "aqlm2")], None))
    out.append(("HarmBench judge", JUDGE_ID, None))
    return out


def dataset_targets():
    """[(label, zero-arg loader)] using the repo's pinned loaders."""
    from core import datasets as D
    return [
        ("WildJailbreak train", lambda: D.load_wildjailbreak_prompts(n_samples = 8, config = "train")),
        ("WildJailbreak eval", lambda: D.load_wildjailbreak_prompts(n_samples = 8, config = "eval")),
        ("XSTest", lambda: D.load_xstest()),
        ("HarmBench", lambda: D.load_harmbench_prompts(n_samples = 8)),
        ("Alpaca (harmless)", lambda: D.load_harmless_prompts(n_samples = 8)),
        ("ARC-Challenge", lambda: D.load_arc_challenge(n_samples = 8)),
    ]


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    ap.add_argument("--list", action = "store_true", help = "print targets and exit (no downloads)")
    ap.add_argument("--skip_datasets", action = "store_true", help = "models only")
    args = ap.parse_args()

    import config  # sets HF_HOME / HF_DATASETS_CACHE from .env before any HF import
    models = model_targets()
    if args.list:
        print(f"HF_HOME={config.HF_HOME}")
        for label, rid, rev in models:
            print(f"  model    {label:22s} {rid}" + (f" @ {rev[:12]}" if rev else ""))
        for label, _ in dataset_targets():
            print(f"  dataset  {label}")
        return 0

    from core.evaluation import _CLS_MODEL_ID
    if _CLS_MODEL_ID != JUDGE_ID:
        print(f"!! judge id drift: core.evaluation uses {_CLS_MODEL_ID}, prefetch has {JUDGE_ID}")
        models[-1] = ("HarmBench judge", _CLS_MODEL_ID, None)

    from huggingface_hub import snapshot_download
    failed = []
    for label, rid, rev in models:
        print(f"=== model {label}: {rid}" + (f" @ {rev}" if rev else ""), flush = True)
        try:
            path = snapshot_download(rid, revision = rev, cache_dir = config.HF_HOME,
                                     ignore_patterns = IGNORE)
            print(f"  ok -> {path}")
        except Exception as e:
            print(f"  !! FAILED {rid}: {type(e).__name__}: {e}")
            failed.append(rid)

    if not args.skip_datasets:
        for label, fn in dataset_targets():
            print(f"=== dataset {label}", flush = True)
            try:
                items = fn()
                print(f"  ok ({len(items)} items loaded)")
            except Exception as e:
                print(f"  !! FAILED {label}: {type(e).__name__}: {e}")
                failed.append(label)

    if failed:
        print(f"\nPREFETCH FAILED for: {failed}")
        print("Gated repos need `huggingface-cli login` with access granted on the model page.")
        return 1
    print("\nPREFETCH OK: all models and datasets present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
