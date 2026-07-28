#Front Matter
import random

import pandas as pd
from datasets import load_dataset


def load_harmbench_prompts(n_samples = None, cache_dir = None):
    """Load harmful behaviors from HarmBench (secondary safety benchmark).
    Dataset: centerforaisafety/HarmBench (CSV pinned by revision SHA).
    Reference: Mazeika et al., HarmBench (ICML 2024)."""
    try:
        sha = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
        url = (
            f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{sha}"
            "/data/behavior_datasets/harmbench_behaviors_text_all.csv"
        )
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Cannot load HarmBench CSV from GitHub: {e}") from e

    prompts = []
    for behavior in df["Behavior"].dropna():
        behavior = str(behavior).strip()
        if not behavior:
            continue
        prompts.append(behavior)
    if not prompts:
        raise RuntimeError(
            "HarmBench loaded but 'Behavior' field is empty, please check the CSV schema."
        )

    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


def load_wildjailbreak_prompts(n_samples = None, config = "train", cache_dir = None):
    """Load adversarial jailbreak prompts from WildJailbreak (NeurIPS 2024).
    Uses the adversarial_harmful subset (jailbreak tactics composed in)."""
    try:
        dataset = load_dataset(
            "allenai/wildjailbreak",
            config,
            revision = "5ddc12a7894f842b0619b8e1c7ee496b198af009",
            delimiter = "\t",
            keep_default_na = False,
            cache_dir = cache_dir,
        )
        dataset = dataset["train"]
    except Exception as e:
        raise RuntimeError(
            f"Cannot load WildJailbreak (allenai/wildjailbreak, config = '{config}'): {e}"
        ) from e

    items = []
    for row in dataset:
        dtype = row.get("data_type", "")
        if "adversarial_harmful" not in dtype:
            continue
        adv = row.get("adversarial", "").strip()
        if not adv:
            continue
        vanilla = row.get("vanilla", "").strip()
        items.append({"prompt": adv, "vanilla": vanilla, "data_type": dtype})

    if not items:
        raise RuntimeError(
            "WildJailbreak loaded but no adversarial_harmful items found."
        )

    # Fixed seed keeps prompt order stable across pipeline phases
    random.Random(42).shuffle(items)
    if n_samples:
        items = items[:n_samples]
    return items


def load_harmless_prompts(n_samples = None, cache_dir = None):
    """Load harmless prompts from Alpaca-cleaned (GitHub JSON, pinned revision)."""
    try:
        sha = "d03c782bffd50ceeb1f4ef3c020129229ec4698c"
        url = (
            f"https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/{sha}"
            "/alpaca_data_cleaned.json"
        )
        df = pd.read_json(url)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load Alpaca-cleaned JSON from GitHub (gururise/AlpacaDataCleaned): {e}"
        ) from e

    prompts = []
    for _, item in df.iterrows():
        instr = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        if not instr:
            continue
        if inp:
            prompts.append(f"{instr}\n{inp}")
        else:
            prompts.append(instr)

    if not prompts:
        raise RuntimeError(
            "Alpaca-cleaned loaded but no instruction fields found, please check the schema."
        )

    random.Random(42).shuffle(prompts)
    if n_samples:
        prompts = prompts[:n_samples]
    return prompts


def load_arc_challenge(n_samples = None, split = "test", cache_dir = None):
    """Load ARC-Challenge multiple-choice items (Clark et al., 2018)."""
    try:
        dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split = split,
            revision = "210d026faf9955653af8916fad021475a3f00453",
            cache_dir = cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load ARC-Challenge (allenai/ai2_arc/ARC-Challenge, split = '{split}'): {e}"
        ) from e

    items = []
    for item in dataset:
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        try:
            answer_idx = labels.index(answer_key)
        except ValueError:
            continue
        items.append(
            {"question": item["question"], "choices": texts, "answer_idx": answer_idx}
        )

    if not items:
        raise RuntimeError("ARC-Challenge loaded but returned 0 valid items.")

    if n_samples:
        items = items[:n_samples]
    return items


def load_xstest(n_samples = None, cache_dir = None):
    """Load XSTest prompts for over-refusal measurement (Rottger et al., NAACL 2024)."""
    try:
        sha = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
        url = f"https://raw.githubusercontent.com/paul-rottger/xstest/{sha}/xstest_prompts.csv"
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load XSTest CSV from GitHub (paul-rottger/xstest): {e}"
        ) from e

    items = []
    for _, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        label = str(row.get("label", "")).strip().lower()
        items.append(
            {
                "prompt": prompt,
                "type": "safe" if label == "safe" else "unsafe",
                "category": str(row.get("type") or row.get("focus") or "").strip(),
            }
        )
    if not items:
        raise RuntimeError("XSTest loaded but returned 0 items, check CSV schema.")
    if n_samples:
        items = items[:n_samples]
    return items


def load_c4_calibration(n_samples = 64, max_chars = 1000, cache_dir = None):
    """Stream random C4 passages for generic calibration / distillation text."""
    try:
        ds = load_dataset(
            "allenai/c4",
            "en",
            split = "train",
            revision = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            streaming = True,
            cache_dir = cache_dir,
        )
    except Exception as e:
        raise RuntimeError(f"Cannot load C4 (allenai/c4, 'en'): {e}") from e

    prompts = []
    for doc in ds:
        text = (doc.get("text") or "").strip()
        if len(text) < 100:
            continue
        prompts.append(text[:max_chars])
        if len(prompts) >= n_samples:
            break
    if not prompts:
        raise RuntimeError("C4 streamed but returned 0 usable documents.")
    return prompts


def load_wikitext2(split = "test", cache_dir = None):
    """Load WikiText-2 (raw) as one concatenated string for perplexity evaluation."""
    try:
        ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split = split, cache_dir = cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load WikiText-2 (wikitext-2-raw-v1, split = {split}): {e}"
        ) from e
    return "\n\n".join(ds["text"])
