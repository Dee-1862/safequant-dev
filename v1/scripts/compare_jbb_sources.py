"""Compare HuggingFace JBB-Behaviors vs jailbreakbench.read_dataset()."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
import jailbreakbench as jbb
from core.datasets import load_harmful_prompts


def main():
    print("=" * 70)
    print("1. RAW SOURCE COMPARISON: HF vs jailbreakbench.read_dataset()")
    print("=" * 70)

    try:
        hf = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
        )
        hf_ok = True
    except Exception as e:
        print(f"HF load FAILED: {e}")
        hf_ok = False
        hf = []

    jbb_ds = jbb.read_dataset()
    jbb_df = jbb_ds.as_dataframe()

    if hf_ok:
        print(f"HF harmful count: {len(hf)}")
        print(f"HF columns: {hf.column_names}")
        hf0 = hf[0]
        print(f"HF row0 Goal: {hf0['Goal'][:70]}...")
        print(f"HF row0 Behavior: {hf0['Behavior']}")

    print(f"JBB lib count: {len(jbb_ds.goals)}")
    print(f"JBB lib columns: {list(jbb_df.columns)}")
    print(f"JBB row0 Goal: {jbb_ds.goals[0][:70]}...")
    print(f"JBB row0 Behavior: {jbb_ds.behaviors[0]}")

    if not hf_ok:
        print("\nSkipping HF comparison (dataset unavailable).")
    else:
        hf_items = [
            {
                "goal": r["Goal"].strip(),
                "behavior": r["Behavior"],
                "target": r["Target"],
                "category": r.get("Category", ""),
            }
            for r in hf
            if r.get("Goal", "").strip()
        ]
        lib_items = load_harmful_prompts()
        print(f"\nNormalized item counts: HF={len(hf_items)} lib={len(lib_items)}")

        hf_by_beh = {x["behavior"]: x for x in hf_items}
        lib_by_beh = {x["behavior"]: x for x in lib_items}
        print(f"Unique behaviors: HF={len(hf_by_beh)} lib={len(lib_by_beh)}")

        only_hf = set(hf_by_beh) - set(lib_by_beh)
        only_lib = set(lib_by_beh) - set(hf_by_beh)
        print(f"Behaviors only in HF: {len(only_hf)}")
        print(f"Behaviors only in lib: {len(only_lib)}")

        shared = set(hf_by_beh) & set(lib_by_beh)
        goal_match = tgt_match = cat_match = 0
        mismatches = []
        for b in shared:
            h, l = hf_by_beh[b], lib_by_beh[b]
            g = h["goal"] == l["goal"]
            t = h["target"] == l["target"]
            c = h.get("category", "") == l.get("category", "")
            goal_match += int(g)
            tgt_match += int(t)
            cat_match += int(c)
            if not (g and t):
                mismatches.append((b, h, l))

        print(f"\nOn {len(shared)} shared behaviors:")
        print(f"  goal exact match:    {goal_match}/{len(shared)}")
        print(f"  target exact match:  {tgt_match}/{len(shared)}")
        print(f"  category exact match:{cat_match}/{len(shared)}")

        n = min(len(hf_items), len(lib_items))
        order_match = all(
            hf_items[i]["goal"] == lib_items[i]["goal"]
            and hf_items[i]["behavior"] == lib_items[i]["behavior"]
            for i in range(n)
        )
        print(f"  row-order match:     {order_match} ({n} rows)")

        if mismatches:
            print(f"\nFirst mismatch ({mismatches[0][0]}):")
            h, l = mismatches[0][1], mismatches[0][2]
            if h["goal"] != l["goal"]:
                print("  GOAL HF: ", h["goal"][:120])
                print("  GOAL lib:", l["goal"][:120])
            if h["target"] != l["target"]:
                print("  TARGET HF: ", h["target"][:120])
                print("  TARGET lib:", l["target"][:120])
        else:
            print("\nAll shared behaviors: goal + target identical.")

    print("\n" + "=" * 70)
    print("2. PIPELINE STRUCTURE: what evaluation.py expects")
    print("=" * 70)

    beh = load_harmful_prompts(3)
    print("load_harmful_prompts() keys:", list(beh[0].keys()))
    gen_prompt = beh[0].get("prompt") or beh[0].get("goal", "")
    print("generation uses:", gen_prompt[:80] + "...")
    print("judge behavior:", beh[0]["behavior"])


if __name__ == "__main__":
    main()
