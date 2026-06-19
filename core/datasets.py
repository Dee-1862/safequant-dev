import random
from datasets import load_dataset


# -- Safety: harmful prompts ---------------------------------------------------

def load_harmful_prompts(n_samples=None, split="harmful", cache_dir=None):
    """
    Load harmful prompts from JailbreakBench (primary safety benchmark).

    Dataset: JailbreakBench/JBB-Behaviors (ungated, no HF token required)
    Format:  {"Behavior": str, "BehaviorID": str, "Category": str, ...}

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset(
            "JailbreakBench/JBB-Behaviors",
            "behaviors",
            split="harmful",
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load JailbreakBench/JBB-Behaviors: {e}\n"
            "Fix: ensure HF_HOME points to Ocean and run the job with network access."
        ) from e

    # Goal = the actual harmful instruction to send to the model
    # Behavior = short label used by the HarmBench judge as "behavior"
    # Target = expected harmful response (also useful for the judge)
    items = [
        {
            "goal":     item["Goal"],
            "behavior": item["Behavior"],
            "target":   item["Target"],
        }
        for item in dataset
        if item.get("Goal", "").strip()
    ]
    if not items:
        raise RuntimeError("JBB-Behaviors loaded but returned 0 items -- check dataset schema.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_harmbench_prompts(n_samples=None, cache_dir=None):
    """
    Load harmful behaviors from HarmBench (secondary safety benchmark) directly from GitHub.
    """
    import pandas as pd
    try:
        url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load HarmBench CSV from GitHub: {e}\n"
            "Fix: ensure network access to raw.githubusercontent.com."
        ) from e

    # The CSV usually has 'Behavior' column
    prompts = [str(b).strip() for b in df["Behavior"].dropna() if str(b).strip()]
    if not prompts:
        raise RuntimeError("HarmBench loaded but 'Behavior' field is empty -- check CSV schema.")

    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


# -- Safety: adversarial jailbreak prompts ------------------------------------

def load_wildjailbreak_prompts(n_samples=None, config="train", cache_dir=None):
    """
    Load adversarial jailbreak prompts from WildJailbreak (NeurIPS 2024).

    Dataset: allenai/wildjailbreak
    Configs: "train" (262K examples, has vanilla+adversarial) or "eval" (2,210, no vanilla)
    Uses the adversarial_harmful subset -- prompts with jailbreak tactics composed in.

    Returns list of dicts: {"prompt": str, "vanilla": str, "data_type": str}
      - prompt:  the adversarial (jailbroken) prompt to send to the model
      - vanilla: the original harmful intent (short, used as behavior label for judge)
    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset(
            "allenai/wildjailbreak",
            config,
            delimiter="\t",
            keep_default_na=False,
            cache_dir=cache_dir,
        )
        # Config defines one split; access via "train" key
        dataset = dataset["train"]
    except Exception as e:
        raise RuntimeError(
            f"Cannot load WildJailbreak (allenai/wildjailbreak, config='{config}'): {e}\n"
            "Fix: ensure HF_HOME points to Ocean and run the job with network access."
        ) from e

    # Filter for adversarial harmful prompts only
    items = []
    for row in dataset:
        dtype = row.get("data_type", "")
        if "adversarial_harmful" not in dtype:
            continue
        adv = row.get("adversarial", "").strip()
        if not adv:
            continue
        vanilla = row.get("vanilla", "").strip()
        items.append({
            "prompt":    adv,
            "vanilla":   vanilla,
            "data_type": dtype,
        })

    if not items:
        raise RuntimeError("WildJailbreak loaded but no adversarial_harmful items found.")

    # Fixed seed ensures same prompts across phases in a single pipeline run
    random.Random(42).shuffle(items)
    if n_samples:
        items = items[:n_samples]
    return items


def load_strongreject_prompts(n_samples=None, cache_dir=None):
    """
    Load forbidden prompts from StrongREJECT (Souly et al., 2024).

    Dataset: walledai/StrongREJECT
    Format:  {"forbidden_prompt": str, "category": str, "source": str}

    Returns list of dicts: {"prompt": str, "category": str, "source": str}
    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset(
            "walledai/StrongREJECT",
            split="train",
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load StrongREJECT (walledai/StrongREJECT): {e}\n"
            "Fix: ensure HF_HOME points to Ocean and run the job with network access."
        ) from e

    items = []
    for row in dataset:
        # Column may be "forbidden_prompt" or "prompt" depending on version
        prompt = (row.get("forbidden_prompt") or row.get("prompt") or "").strip()
        if not prompt:
            continue
        items.append({
            "prompt":   prompt,
            "category": row.get("category", ""),
            "source":   row.get("source", ""),
        })

    if not items:
        raise RuntimeError("StrongREJECT loaded but no forbidden_prompt fields found.")

    if n_samples:
        items = items[:n_samples]
    return items


# -- Safety: harmless prompts (for refusal direction) -------------------------

def load_harmless_prompts(n_samples=None, split="train", cache_dir=None):
    """
    Load harmless prompts from Alpaca-cleaned (contrast class for refusal direction).

    Dataset: yahma/alpaca-cleaned
    Format:  {"instruction": str, "input": str, "output": str}
    Prompt:  instruction [ + '\n' + input if input is non-empty ]

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset("yahma/alpaca-cleaned", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load Alpaca-cleaned (yahma/alpaca-cleaned, split='{split}'): {e}\n"
            "Fix: ensure HF_HOME points to Ocean and run the job with network access."
        ) from e

    prompts = []
    for item in dataset:
        instr = item.get("instruction", "").strip()
        inp   = item.get("input", "").strip()
        if not instr:
            continue
        prompts.append(f"{instr}\n{inp}" if inp else instr)

    if not prompts:
        raise RuntimeError("Alpaca-cleaned loaded but no instruction fields found -- check dataset schema.")

    random.Random(42).shuffle(prompts)
    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


# -- Utility benchmarks --------------------------------------------------------

def load_hellaswag(n_samples=None, split="validation", cache_dir=None):
    """
    HellaSwag: 4-way sentence-completion task.

    Dataset: hellaswag
    Returns: list of {"ctx": str, "endings": [str]*4, "label": int}

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset("hellaswag", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load HellaSwag (hellaswag, split='{split}'): {e}"
        ) from e

    items = [
        {
            "ctx":     item["ctx"],
            "endings": item["endings"],
            "label":   int(item["label"]),
        }
        for item in dataset
    ]
    if not items:
        raise RuntimeError("HellaSwag loaded but returned 0 items.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_boolq(n_samples=None, split="validation", cache_dir=None):
    """
    BoolQ: binary yes/no reading-comprehension task.

    Dataset: google/boolq
    Returns: list of {"question": str, "passage": str, "answer": bool}

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset("google/boolq", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load BoolQ (google/boolq, split='{split}'): {e}"
        ) from e

    items = [
        {
            "question": item["question"],
            "passage":  item["passage"],
            "answer":   bool(item["answer"]),
        }
        for item in dataset
    ]
    if not items:
        raise RuntimeError("BoolQ loaded but returned 0 items.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_rte(n_samples=None, split="validation", cache_dir=None):
    """
    RTE (Recognizing Textual Entailment) from SuperGLUE.

    Dataset: super_glue / rte
    Returns: list of {"premise": str, "hypothesis": str, "label": int}
             label -- 0: entailment, 1: not_entailment

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset("super_glue", "rte", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load RTE (super_glue/rte, split='{split}'): {e}"
        ) from e

    items = [
        {
            "premise":    item["premise"],
            "hypothesis": item["hypothesis"],
            "label":      int(item["label"]),
        }
        for item in dataset
    ]
    if not items:
        raise RuntimeError("RTE loaded but returned 0 items.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_arc_challenge(n_samples=None, split="test", cache_dir=None):
    """
    ARC-Challenge: 4-way science multiple-choice task.

    Dataset: ai2_arc / ARC-Challenge
    Returns: list of {"question": str, "choices": [str, ...], "answer_idx": int}

    Raises RuntimeError if the dataset cannot be loaded.
    Skips malformed entries where answerKey is not found in the label list.
    """
    try:
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load ARC-Challenge (ai2_arc/ARC-Challenge, split='{split}'): {e}"
        ) from e

    items = []
    for item in dataset:
        labels     = item["choices"]["label"]
        texts      = item["choices"]["text"]
        answer_key = item["answerKey"]
        try:
            answer_idx = labels.index(answer_key)
        except ValueError:
            continue  # skip rare malformed entries (numeric vs letter label mismatch)
        items.append({"question": item["question"], "choices": texts, "answer_idx": answer_idx})

    if not items:
        raise RuntimeError("ARC-Challenge loaded but returned 0 valid items.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_winogrande(n_samples=None, split="validation", cache_dir=None):
    """
    WinoGrande: commonsense fill-in-the-blank with two options.

    Dataset: winogrande / winogrande_xl
    Returns: list of {"sentence": str, "options": [str, str], "answer_idx": int}
             answer_idx -- 0 for option1, 1 for option2

    Raises RuntimeError if the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset("winogrande", "winogrande_xl", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load WinoGrande (winogrande/winogrande_xl, split='{split}'): {e}"
        ) from e

    items = []
    for item in dataset:
        items.append({
            "sentence":   item["sentence"],
            "options":    [item["option1"], item["option2"]],
            "answer_idx": int(item["answer"]) - 1,  # "1"/"2" -> 0/1
        })

    if not items:
        raise RuntimeError("WinoGrande loaded but returned 0 items.")

    if n_samples:
        items = items[:n_samples]
    return items


# -- XSTest: over-refusal benchmark (Rottger et al. 2024) ---------------------

def load_xstest(n_samples=None, cache_dir=None):
    """
    XSTest: 250 prompts (200 'safe' designed-to-trick benign + 50 'unsafe' genuinely
    harmful) used to measure exaggerated safety behavior (over-refusal).

    Dataset: natolambert/xstest-v2-copy
    Returns list of dicts: {"prompt": str, "type": "safe"|"unsafe", "category": str}
      - "safe":   should ANSWER (refusing these counts as over-refusal)
      - "unsafe": should REFUSE (refusing these is correct)

    Raises RuntimeError if dataset cannot be loaded.
    """
    try:
        ds = load_dataset("natolambert/xstest-v2-copy", split="prompts",
                          cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load XSTest (natolambert/xstest-v2-copy, split='prompts'): {e}\n"
            f"Fix: ensure HF_HOME points to Ocean and the job has network access."
        ) from e

    items = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        t = (row.get("type") or "").lower()
        # XSTest v2 has 450 prompts: 200 genuinely-unsafe prompts whose 'type' is
        # prefixed 'contrast_' (contrast_homonyms, contrast_definitions, ...), and
        # 250 safe (designed-to-trick-but-benign) prompts with plain category types
        # (homonyms, figurative_language, safe_targets, ...). Safe == NOT contrast.
        is_safe = not t.startswith("contrast")
        items.append({
            "prompt": prompt,
            "type": "safe" if is_safe else "unsafe",
            "category": row.get("category") or row.get("focus") or "",
        })
    if not items:
        raise RuntimeError("XSTest loaded but returned 0 items -- check schema.")
    if n_samples:
        items = items[:n_samples]
    return items


# -- C4: generic web text for the perplexity-only calibration baseline --------

def load_c4_calibration(n_samples=64, max_chars=1000, cache_dir=None):
    """
    Random C4 (Colossal Clean Crawled Corpus) passages for use as
    perplexity-style (no harmful/harmless contrast) calibration data.

    Returns a list of strings, each <= max_chars characters, suitable as the
    'user turn' in a chat template. The teacher will generate continuations.

    Streams allenai/c4 English split to avoid downloading the full corpus.
    """
    try:
        ds = load_dataset("allenai/c4", "en", split="train",
                          streaming=True, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load C4 (allenai/c4, 'en'): {e}\n"
            f"Fix: ensure HF_HOME points to Ocean and the job has network access."
        ) from e

    prompts = []
    for doc in ds:
        text = (doc.get("text") or "").strip()
        if len(text) < 100:
            continue  # skip very short documents
        prompts.append(text[:max_chars])
        if len(prompts) >= n_samples:
            break
    if not prompts:
        raise RuntimeError("C4 streamed but returned 0 usable documents.")
    return prompts


# -- CLI: prefetch all datasets ------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser(description="Pre-download and spot-check all benchmark datasets")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples to preview per dataset")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SafeQuant Dataset Pre-fetch")
    print("=" * 70 + "\n")

    checks = [
        ("JBB-Behaviors (harmful)", lambda: load_harmful_prompts(args.n_samples, cache_dir=args.cache_dir)),
        ("HarmBench (harmful)",   lambda: load_harmbench_prompts(args.n_samples, cache_dir=args.cache_dir)),
        ("WildJailbreak (adv)",   lambda: load_wildjailbreak_prompts(args.n_samples, cache_dir=args.cache_dir)),
        ("StrongREJECT",          lambda: load_strongreject_prompts(args.n_samples, cache_dir=args.cache_dir)),
        ("Alpaca-cleaned",        lambda: load_harmless_prompts(args.n_samples, cache_dir=args.cache_dir)),
        ("HellaSwag",             lambda: load_hellaswag(args.n_samples, cache_dir=args.cache_dir)),
        ("BoolQ",                 lambda: load_boolq(args.n_samples, cache_dir=args.cache_dir)),
        ("RTE",                   lambda: load_rte(args.n_samples, cache_dir=args.cache_dir)),
        ("ARC-Challenge",         lambda: load_arc_challenge(args.n_samples, cache_dir=args.cache_dir)),
        ("WinoGrande",            lambda: load_winogrande(args.n_samples, cache_dir=args.cache_dir)),
    ]

    errors = 0
    for name, loader in checks:
        try:
            data = loader()
            first = data[0] if data else None
            print(f"  OK {name}: {len(data)} items loaded")
            if first:
                preview = str(first)[:120]
                print(f"    Preview: {preview}")
        except RuntimeError as e:
            print(f"  FAIL {name}: FAILED\n    {e}")
            errors += 1
        print()

    print("=" * 70)
    if errors == 0:
        print("OK All datasets accessible.")
    else:
        print(f"FAIL {errors} dataset(s) failed -- fix before submitting jobs.")
    print("=" * 70)
