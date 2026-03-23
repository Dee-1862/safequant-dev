"""
SafeQuant Phase 1: Compile all results into a summary.

Reads outputs from step1_direction, step2_behavior, and step3_circuit and
prints Findings 1–3 (direction preservation, behavioral safety, circuit analysis).

Run after all steps are complete:
  python phase1_problem/summary/compile_phase1.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from config import MODELS


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    print("=" * 70)
    print("  SafeQuant Phase 1: RESULTS SUMMARY")
    print("  Proving that quantization degrades safety despite")
    print("  preserving the refusal direction.")
    print("=" * 70)
    
    for model_key in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_key} ({MODELS[model_key]})")
        print(f"{'='*70}")
        
        dir_out = os.path.join("phase1_problem", "step1_direction", "outputs", model_key)
        beh_out = os.path.join("phase1_problem", "step2_behavior", "outputs", model_key)
        cir_out = os.path.join("phase1_problem", "step3_circuit", "outputs", model_key)
        
        # --- Finding 1: Direction Preservation ---
        dir_comp = load_json_safe(os.path.join(dir_out, "direction_comparison.json"))
        if dir_comp:
            print(f"\n  FINDING 1: Direction Preservation")
            print(f"  ---------------------------------")
            print(f"  Best direction cosine (FP16 vs NF4): {dir_comp['best_cosine_similarity']:.4f}")
            print(f"  Absolute cosine:                     {dir_comp['best_abs_cosine']:.4f}")
            print(f"  Overall mean abs cosine (all layers): {dir_comp['overall_abs_mean_cosine']:.4f}")
            print(f"  FP16 best at layer {dir_comp['fp16_best_layer']}, "
                  f"NF4 best at layer {dir_comp['nf4_best_layer']}")
            
            if dir_comp['best_abs_cosine'] > 0.80:
                print(f"  → CONFIRMED: Refusal direction is well-preserved (>{0.80})")
            else:
                print(f"  → NOTE: Direction preservation is moderate")
        else:
            print(f"\n  FINDING 1: [Not yet computed - run step1_direction scripts]")
        
        # --- Finding 2: Behavioral Degradation ---
        asr_fp16 = load_json_safe(os.path.join(beh_out, "asr_fp16.json"))
        asr_nf4 = load_json_safe(os.path.join(beh_out, "asr_nf4.json"))
        
        if asr_fp16 and asr_nf4:
            print(f"\n  FINDING 2: Behavioral Safety")
            print(f"  ----------------------------")
            print(f"  FP16 ASR: {asr_fp16['asr_percent']:.1f}% "
                  f"({asr_fp16['complied']}/{asr_fp16['total_prompts']} complied)")
            print(f"  NF4  ASR: {asr_nf4['asr_percent']:.1f}% "
                  f"({asr_nf4['complied']}/{asr_nf4['total_prompts']} complied)")
            
            delta = asr_nf4['asr_percent'] - asr_fp16['asr_percent']
            if delta > 0:
                print(f"  → CONFIRMED: Safety degraded by {delta:.1f} percentage points")
            elif delta == 0:
                print(f"  → NOTE: No measurable safety degradation on this model size")
                print(f"    (Expected — small models may not show behavioral change)")
                print(f"    (Circuit analysis below may still show signal weakening)")
            else:
                print(f"  → UNEXPECTED: NF4 appears safer than FP16")
        else:
            print(f"\n  FINDING 2: [Not yet computed - run step2_behavior scripts]")
        
        # --- Finding 2b: Utility ---
        util_fp16 = load_json_safe(os.path.join(beh_out, "utility_fp16.json"))
        util_nf4 = load_json_safe(os.path.join(beh_out, "utility_nf4.json"))
        
        if util_fp16 and util_nf4:
            for b_fp16, b_nf4 in zip(util_fp16["benchmarks"], util_nf4["benchmarks"]):
                print(f"\n  Utility ({b_fp16['benchmark']}):")
                print(f"    FP16: {b_fp16['accuracy']:.1f}%")
                print(f"    NF4:  {b_nf4['accuracy']:.1f}%")
                delta = b_nf4['accuracy'] - b_fp16['accuracy']
                print(f"    Delta: {delta:+.1f}%")
        
        # --- Finding 3: Circuit Analysis ---
        circ_comp = load_json_safe(os.path.join(cir_out, "circuit_comparison.json"))
        
        if circ_comp:
            print(f"\n  FINDING 3: Circuit-Level Explanation")
            print(f"  ------------------------------------")
            print(f"  Signal retention: {circ_comp['signal_retention_percent']:.1f}%")
            print(f"  Net signal FP16: {circ_comp['net_signal_fp16']:.4f}")
            print(f"  Net signal NF4:  {circ_comp['net_signal_nf4']:.4f}")
            
            print(f"\n  Top 5 most affected components:")
            for item in circ_comp["top_degraded"][:5]:
                print(f"    {item['component']:<20}: "
                      f"FP16={item['fp16_dfa']:.4f} → NF4={item['nf4_dfa']:.4f} "
                      f"(Δ={item['absolute_change']:+.4f})")
            
            print(f"\n  → These components are the candidates for targeted repair in Phase 2.")
        else:
            print(f"\n  FINDING 3: [Not yet computed - run step3_circuit scripts]")
    
    # Summary
    print(f"\n{'='*70}")
    print("  PHASE 1 CONCLUSION")
    print(f"{'='*70}")
    print("""
  If Findings 1-3 are confirmed:
    - The refusal direction survives NF4 quantization (Finding 1)
    - But actual safety behavior degrades (Finding 2)  
    - Because specific circuit components lost refusal signal (Finding 3)
    
  This sets up Phase 2: Can we PREDICT which components will fail
  (vulnerability score = DFA importance × quantization error)
  and PERMANENTLY FIX them (targeted AIRD weight repair)?
    """)


if __name__ == "__main__":
    main()
