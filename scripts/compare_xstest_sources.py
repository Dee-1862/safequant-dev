"""Compare old HF XSTest loader vs current GitHub CSV loader."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import load_dataset
from core.datasets import load_xstest


def load_xstest_hf_old(n_samples=None, cache_dir=None):
    """Previous loader: natolambert/xstest-v2-copy + contrast-prefix heuristic."""
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts", cache_dir=cache_dir)
    items = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        t = (row.get("type") or "").lower()
        is_safe = not t.startswith("contrast")
        items.append({
            "prompt": prompt,
            "type": "safe" if is_safe else "unsafe",
            "category": row.get("category") or row.get("focus") or "",
        })
    if n_samples:
        items = items[:n_samples]
    return items


def main():
    print("=" * 70)
    print("XSTest: HF (old) vs GitHub CSV (current)")
    print("=" * 70)

    try:
        old = load_xstest_hf_old()
        hf_ok = True
    except Exception as e:
        print(f"HF load failed: {e}")
        hf_ok = False
        old = []

    try:
        new = load_xstest()
        csv_ok = True
    except Exception as e:
        print(f"CSV load failed: {e}")
        csv_ok = False
        new = []

    if not hf_ok or not csv_ok:
        return

    print(f"\nCounts: HF={len(old)}  CSV={len(new)}")
    print(f"Keys: HF={list(old[0].keys())}  CSV={list(new[0].keys())}")

    old_safe = sum(1 for x in old if x["type"] == "safe")
    old_unsafe = sum(1 for x in old if x["type"] == "unsafe")
    new_safe = sum(1 for x in new if x["type"] == "safe")
    new_unsafe = sum(1 for x in new if x["type"] == "unsafe")
    print(f"Safe/unsafe: HF={old_safe}/{old_unsafe}  CSV={new_safe}/{new_unsafe}")

    old_by_prompt = {x["prompt"]: x for x in old}
    new_by_prompt = {x["prompt"]: x for x in new}
    shared = set(old_by_prompt) & set(new_by_prompt)
    only_old = set(old_by_prompt) - set(new_by_prompt)
    only_new = set(new_by_prompt) - set(old_by_prompt)
    print(f"\nPrompt overlap: shared={len(shared)} only_hf={len(only_old)} only_csv={len(only_new)}")

    type_mismatch = []
    for p in shared:
        if old_by_prompt[p]["type"] != new_by_prompt[p]["type"]:
            type_mismatch.append((p, old_by_prompt[p]["type"], new_by_prompt[p]["type"]))

    print(f"Type mismatches on shared prompts: {len(type_mismatch)}")
    if type_mismatch:
        p, o, n = type_mismatch[0]
        print(f"  Example: {p[:80]}...")
        print(f"    HF type={o}  CSV type={n}")

    # Order check for [:n_samples] reproducibility
    n = min(50, len(old), len(new))
    order_match = all(old[i]["prompt"] == new[i]["prompt"] for i in range(n))
    print(f"\nFirst-{n} row order match: {order_match}")

    if only_old:
        print(f"\nSample only in HF ({len(only_old)}):")
        for p in list(only_old)[:2]:
            print(f"  {p[:90]}...")
    if only_new:
        print(f"\nSample only in CSV ({len(only_new)}):")
        for p in list(only_new)[:2]:
            print(f"  {p[:90]}...")

    # Direct CSV schema peek
    url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    df = pd.read_csv(url)
    print(f"\nGitHub CSV columns: {list(df.columns)}")
    print(f"GitHub CSV label counts:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
