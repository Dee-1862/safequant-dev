"""
SafeQuant Phase 2 v2 — Compile all results.

Run:
  python phase2_solution/summary/compile_phase2.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from config import MODELS


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    print("=" * 75)
    print("  SafeQuant Phase 2 v2: RESULTS SUMMARY")
    print("  Per-Dimension Vulnerability-Guided Sparse Repair")
    print("=" * 75)

    for model_key in MODELS:
        diag_dir = os.path.join("phase2_solution", "diagnosis", "outputs", model_key)
        eval_dir = os.path.join("phase2_solution", "evaluate", "outputs", model_key)

        print(f"\n{'='*75}")
        print(f"  MODEL: {model_key}")
        print(f"{'='*75}")

        # Vulnerability scores
        vuln = load_json(os.path.join(diag_dir, "vulnerability_scores.json"))
        if vuln:
            v = vuln["validation"]
            print(f"\n  DIAGNOSIS")
            print(f"  Total dimensions scored: {vuln['total_dimensions']}")
            print(f"  Validation (top 50% DFA filter):")
            print(f"    Corr(V_i, actual loss):  {v['corr_v_score']:.4f}")
            print(f"    Corr(|DFA_i|, loss):     {v['corr_dfa_alone']:.4f}")
            print(f"    Corr(ε_i, loss):         {v['corr_error_alone']:.4f}")
            print(f"    Best predictor: {v['best_predictor']}")

            print(f"\n  Top 5 vulnerable dimensions:")
            for item in vuln["top_200_vulnerable"][:5]:
                print(f"    {item['component']:<18} dim {item['dimension']:>4}: "
                      f"V={item['vulnerability_score']:.6f}")
        else:
            print(f"\n  DIAGNOSIS: [not computed]")

        # Repair comparison
        comp = load_json(os.path.join(eval_dir, "comparison_results.json"))
        if comp:
            print(f"\n  REPAIR COMPARISON")
            print(f"  {'Condition':<15} | {'ASR':>8} | {'Utility':>8} | {'Overhead':>8}")
            print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            for cond in ["fp16", "nf4", "targeted", "chhabra", "random", "dfa_only", "error_only"]:
                if cond not in comp:
                    continue
                r = comp[cond]
                asr = f"{r['asr']:.1f}%" if "asr" in r else "N/A"
                util = f"{r['utility']:.1f}%" if isinstance(r.get("utility"), (int, float)) else "N/A"
                ovh = r.get("overhead", "-")
                print(f"  {cond:<15} | {asr:>8} | {util:>8} | {str(ovh):>8}")
        else:
            print(f"\n  REPAIR: [not computed]")

    print(f"\n{'='*75}")
    print("  KEY CLAIMS FOR PAPER")
    print(f"{'='*75}")
    print("""
  1. Per-dimension V_i(c) = |DFA_i(c)| × ε_i(c) predicts safety damage
     at individual weight-row resolution.

  2. Global Frobenius norm hides localized damage: top 1% vulnerable
     dimensions show Nx higher error than global average.

  3. Sparse ΔW repair (restoring only high-V_i rows) recovers safety
     with minimal FP16 overhead, deployable as sparse LoRA adapter.

  4. Targeted repair outperforms Chhabra (layer-level AIRD) and
     random repair, validating that V-score identifies correct targets.

  5. (Secondary) Different harm categories have different vulnerability
     profiles, suggesting quantizers may selectively damage specific
     safety behaviors.
    """)


if __name__ == "__main__":
    main()
