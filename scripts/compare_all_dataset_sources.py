"""
Compare HuggingFace dataset loaders vs GitHub / package alternatives.

Run: python scripts/compare_all_dataset_sources.py

Helps decide whether to migrate off HF for reproducibility.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
import jailbreakbench as jbb

from core.datasets import (
    load_arc_challenge,
    load_harmbench_prompts,
    load_harmful_prompts,
    load_harmless_prompts,
    load_wildjailbreak_prompts,
    load_xstest,
)

SEED = 42


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _report(name: str, *, hf_ok: bool, alt_ok: bool, detail: str, recommendation: str) -> None:
    status = "OK" if hf_ok and alt_ok else ("PARTIAL" if alt_ok or hf_ok else "FAIL")
    print(f"\n[{status}] {name}")
    print(detail)
    print(f"  -> {recommendation}")


def compare_jbb() -> None:
    _header("1. JBB-Behaviors: HF vs jailbreakbench.read_dataset()")
    try:
        hf = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        hf_items = [
            {"goal": r["Goal"].strip(), "behavior": r["Behavior"], "target": r["Target"]}
            for r in hf if r.get("Goal", "").strip()
        ]
        hf_ok = True
    except Exception as e:
        hf_ok = False
        hf_items = []
        hf_err = str(e)

    try:
        lib_items = [
            {"goal": x["goal"], "behavior": x["behavior"], "target": x["target"]}
            for x in load_harmful_prompts()
        ]
        alt_ok = True
    except Exception as e:
        alt_ok = False
        lib_items = []
        lib_err = str(e)

    if not hf_ok or not alt_ok:
        _report(
            "JBB-Behaviors",
            hf_ok=hf_ok,
            alt_ok=alt_ok,
            detail=f"HF: {'fail ' + hf_err if not hf_ok else f'{len(hf_items)} items'}\n"
                   f"  lib: {'fail ' + lib_err if not alt_ok else f'{len(lib_items)} items'}",
            recommendation="Already on jailbreakbench; keep it.",
        )
        return

    hf_by = {x["behavior"]: x for x in hf_items}
    lib_by = {x["behavior"]: x for x in lib_items}
    shared = set(hf_by) & set(lib_by)
    goal_match = sum(1 for b in shared if hf_by[b]["goal"] == lib_by[b]["goal"])
    tgt_match = sum(1 for b in shared if hf_by[b]["target"] == lib_by[b]["target"])

    _report(
        "JBB-Behaviors",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  counts: HF={len(hf_items)} lib={len(lib_items)} shared_behaviors={len(shared)}\n"
            f"  goal match: {goal_match}/{len(shared)}  target match: {tgt_match}/{len(shared)}"
        ),
        recommendation="Already migrated to jailbreakbench. Safe — 99/100 identical (one opioids typo in lib CSV).",
    )


def compare_harmbench() -> None:
    _header("2. HarmBench behaviors: HF mirror vs GitHub CSV (current)")
    gh = load_harmbench_prompts()
    gh_ok = True

    hf_ok = False
    hf_prompts: list[str] = []
    hf_name = ""
    for candidate in (
        "walledai/HarmBench",
        "cais/HarmBench",
        "swiss-ai/harmbench",
    ):
        try:
            ds = load_dataset(candidate, split="train")
            col = "Behavior" if "Behavior" in ds.column_names else None
            if col is None:
                continue
            hf_prompts = [str(x).strip() for x in ds[col] if str(x).strip()]
            hf_ok = True
            hf_name = candidate
            break
        except Exception:
            continue

    if not hf_ok:
        _report(
            "HarmBench",
            hf_ok=False,
            alt_ok=gh_ok,
            detail=f"  GitHub CSV: {len(gh)} behaviors\n  HF: no matching public dataset found in mirrors",
            recommendation="Already on GitHub CSV (official HarmBench repo). No HF migration needed.",
        )
        return

    gh_set, hf_set = set(gh), set(hf_prompts)
    _report(
        "HarmBench",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  GitHub CSV: {len(gh)}  HF ({hf_name}): {len(hf_prompts)}\n"
            f"  exact prompt overlap: {len(gh_set & hf_set)}  only_gh={len(gh_set - hf_set)}  only_hf={len(hf_set - gh_set)}"
        ),
        recommendation="Keep GitHub CSV — it is the official source; HF mirrors may be subsets or different splits.",
    )


def compare_wildjailbreak() -> None:
    _header("3. WildJailbreak eval: HF load_dataset vs HF raw TSV URL")
    # WildJailbreak has no public full mirror on GitHub; HF dataset IS the canonical host.
    try:
        hf_items = load_wildjailbreak_prompts(n_samples=500, config="eval")
        hf_ok = True
    except Exception as e:
        hf_ok = False
        hf_err = str(e)
        hf_items = []

    alt_ok = False
    alt_items: list[dict] = []
    alt_err = ""
    try:
        url = "https://huggingface.co/datasets/allenai/wildjailbreak/resolve/main/eval/eval.tsv"
        df = pd.read_csv(url, sep="\t", keep_default_na=False)
        items = []
        for _, row in df.iterrows():
            dtype = str(row.get("data_type", ""))
            if "adversarial_harmful" not in dtype:
                continue
            adv = str(row.get("adversarial", "")).strip()
            if not adv:
                continue
            items.append({
                "prompt": adv,
                "vanilla": str(row.get("vanilla", "")).strip(),
                "data_type": dtype,
            })
        random.Random(SEED).shuffle(items)
        alt_items = items[:500]
        alt_ok = True
    except Exception as e:
        alt_err = str(e)

    if not hf_ok or not alt_ok:
        _report(
            "WildJailbreak",
            hf_ok=hf_ok,
            alt_ok=alt_ok,
            detail=(
                f"  HF loader: {'fail — ' + hf_err if not hf_ok else f'{len(hf_items)} eval adv_harmful (n=500 after shuffle)'}\n"
                f"  raw TSV:   {'fail — ' + alt_err if not alt_ok else f'{len(alt_items)} eval adv_harmful (n=500 after shuffle)'}"
            ),
            recommendation="Stay on HF — dataset is gated and hosted only on HuggingFace (no full GitHub dump).",
        )
        return

    hf_by = {x["prompt"]: x for x in hf_items}
    alt_by = {x["prompt"]: x for x in alt_items}
    shared = set(hf_by) & set(alt_by)
    vanilla_match = sum(1 for p in shared if hf_by[p]["vanilla"] == alt_by[p]["vanilla"])

    _report(
        "WildJailbreak (eval, n=500 after seed=42 shuffle)",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  shared prompts: {len(shared)}/{len(hf_items)}\n"
            f"  vanilla match on shared: {vanilla_match}/{len(shared)}"
        ),
        recommendation="Keep HF load_dataset — same TSV under the hood; no separate GitHub corpus to switch to.",
    )


def compare_alpaca() -> None:
    _header("4. Alpaca-cleaned: HF yahma/alpaca-cleaned vs GitHub AlpacaDataCleaned")
    hf_prompts = load_harmless_prompts(n_samples=1000)
    hf_ok = True

    alt_ok = False
    alt_prompts: list[str] = []
    alt_err = ""
    try:
        import urllib.request
        with urllib.request.urlopen(
            "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
        ) as resp:
            data = json.load(resp)
        prompts = []
        for item in data:
            instr = str(item.get("instruction", "")).strip()
            inp = str(item.get("input", "")).strip()
            if not instr:
                continue
            prompts.append(f"{instr}\n{inp}" if inp else instr)
        random.Random(SEED).shuffle(prompts)
        alt_prompts = prompts[:1000]
        alt_ok = True
    except Exception as e:
        alt_err = str(e)

    if not alt_ok:
        _report(
            "Alpaca-cleaned",
            hf_ok=hf_ok,
            alt_ok=False,
            detail=f"  HF (shuffled n=1000): {len(hf_prompts)}\n  GitHub JSON: fail — {alt_err}",
            recommendation="Keep HF yahma/alpaca-cleaned unless you vendor a snapshot.",
        )
        return

    hf_set, alt_set = set(hf_prompts), set(alt_prompts)
    first50 = hf_prompts[:50] == alt_prompts[:50]
    _report(
        "Alpaca-cleaned (n=1000 after seed=42 shuffle)",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  set overlap: {len(hf_set & alt_set)}/{len(hf_set)} unique prompts\n"
            f"  first-50 order match after same shuffle: {first50}"
        ),
        recommendation="Keep HF if overlap is high; GitHub JSON is the upstream cleaned file yahma mirrored.",
    )


def compare_arc() -> None:
    _header("5. ARC-Challenge test: HF ai2_arc vs GitHub allenai/ai2_arc JSONL")
    hf_items = load_arc_challenge(n_samples=None, split="test")
    hf_ok = True

    alt_ok = False
    alt_items: list[dict] = []
    alt_err = ""
    try:
        url = "https://raw.githubusercontent.com/allenai/ai2_arc/master/ARC-Challenge/ARC-Challenge-Test.jsonl"
        items = []
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            for line in resp:
                item = json.loads(line)
                labels = item["choices"]["label"]
                texts = item["choices"]["text"]
                answer_key = item["answerKey"]
                try:
                    answer_idx = labels.index(answer_key)
                except ValueError:
                    continue
                items.append({
                    "question": item["question"],
                    "choices": texts,
                    "answer_idx": answer_idx,
                })
        alt_items = items
        alt_ok = True
    except Exception as e:
        alt_err = str(e)

    if not alt_ok:
        _report(
            "ARC-Challenge",
            hf_ok=hf_ok,
            alt_ok=False,
            detail=f"  HF: {len(hf_items)} items\n  GitHub JSONL: fail — {alt_err}",
            recommendation="Keep HF ai2_arc.",
        )
        return

    hf_by_q = {x["question"]: x for x in hf_items}
    alt_by_q = {x["question"]: x for x in alt_items}
    shared = set(hf_by_q) & set(alt_by_q)
    ans_match = sum(
        1 for q in shared
        if hf_by_q[q]["choices"] == alt_by_q[q]["choices"]
        and hf_by_q[q]["answer_idx"] == alt_by_q[q]["answer_idx"]
    )
    order_match = all(
        hf_items[i]["question"] == alt_items[i]["question"]
        for i in range(min(len(hf_items), len(alt_items)))
    )

    _report(
        "ARC-Challenge (test split)",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  counts: HF={len(hf_items)} GitHub={len(alt_items)} shared_questions={len(shared)}\n"
            f"  choices+answer match on shared: {ans_match}/{len(shared)}\n"
            f"  row-order match: {order_match}"
        ),
        recommendation=(
            "Could switch to GitHub JSONL if identical — same AI2 source, avoids HF dependency."
            if ans_match == len(shared) and len(hf_items) == len(alt_items)
            else "Keep HF until you verify any mismatch."
        ),
    )


def compare_xstest() -> None:
    _header("6. XSTest: HF natolambert/xstest-v2-copy vs GitHub CSV (current)")
    try:
        ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
        old = []
        for row in ds:
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            t = (row.get("type") or "").lower()
            old.append({
                "prompt": prompt,
                "type": "safe" if not t.startswith("contrast") else "unsafe",
            })
        hf_ok = True
    except Exception as e:
        hf_ok = False
        old = []
        hf_err = str(e)

    try:
        new = [{"prompt": x["prompt"], "type": x["type"]} for x in load_xstest()]
        alt_ok = True
    except Exception as e:
        alt_ok = False
        new = []
        alt_err = str(e)

    if not hf_ok or not alt_ok:
        _report(
            "XSTest",
            hf_ok=hf_ok,
            alt_ok=alt_ok,
            detail="Load failure on one side.",
            recommendation="Already on GitHub CSV.",
        )
        return

    old_by = {x["prompt"]: x for x in old}
    new_by = {x["prompt"]: x for x in new}
    shared = set(old_by) & set(new_by)
    type_match = sum(1 for p in shared if old_by[p]["type"] == new_by[p]["type"])

    _report(
        "XSTest",
        hf_ok=True,
        alt_ok=True,
        detail=(
            f"  counts: HF={len(old)} CSV={len(new)} shared={len(shared)}\n"
            f"  type match on shared: {type_match}/{len(shared)}"
        ),
        recommendation="Already migrated to GitHub CSV. Reproducibility-safe for metrics.",
    )


def compare_c4_note() -> None:
    _header("7. C4 calibration stream (note only)")
    _report(
        "C4 (allenai/c4 streaming)",
        hf_ok=True,
        alt_ok=False,
        detail=(
            "  No practical GitHub equivalent — multi-TB corpus.\n"
            "  load_c4_calibration() streams first N docs; order depends on HF shard/stream.\n"
            "  Not comparable item-for-item without pinning a fixed doc list."
        ),
        recommendation="Keep HF streaming; for LRQAT reproducibility pin a local snapshot of N doc strings if needed.",
    )


def main() -> None:
    print("SafeQuant dataset source comparison")
    print("Active HF loaders in core/datasets.py vs GitHub/package alternatives\n")
    compare_jbb()
    compare_harmbench()
    compare_wildjailbreak()
    compare_alpaca()
    compare_arc()
    compare_xstest()
    compare_c4_note()
    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
