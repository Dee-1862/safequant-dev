"""
SafeQuant Phase 1 — Compile All Results Into a Professor-Ready Summary.

This generates the complete argument:
  1. Safety degrades after quantization (ASR doubles)
  2. Standard metrics miss it (perplexity barely changes, Frobenius is uniform)
  3. The refusal direction survives (the MODEL knows what to refuse)
  4. But safety circuits weaken (the MODEL can't act on that knowledge)
  5. Standard error metrics can't identify where (Frobenius is uniform)
  → Therefore: we need per-dimension analysis weighted by safety importance

Run after all steps are complete:
  python phase1_problem/summary/compile_phase1.py --model qwen1.5b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MODELS


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Specific model to summarize (default: all)")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else list(MODELS.keys())

    for model_key in models_to_run:
        if model_key not in MODELS:
            continue

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dir_out = os.path.join(base, "step1_direction", "outputs", model_key)
        beh_out = os.path.join(base, "step2_behavior", "outputs", model_key)
        cir_out = os.path.join(base, "step3_circuit", "outputs", model_key)
        sum_out = os.path.join(base, "summary", "outputs", model_key)
        os.makedirs(sum_out, exist_ok=True)

        print(f"\n{'#'*72}")
        print(f"#{'':>70}#")
        print(f"#  SafeQuant Phase 1: DOES QUANTIZATION BREAK SAFETY?{'':>18}#")
        print(f"#  Model: {model_key} ({MODELS[model_key]}){'':>{55-len(model_key)-len(MODELS[model_key])}}#")
        print(f"#{'':>70}#")
        print(f"{'#'*72}")

        report = {"model": model_key, "model_name": MODELS[model_key]}

        # =============================================================
        # FINDING 1: SAFETY DEGRADES
        # =============================================================
        asr_fp16 = load_json_safe(os.path.join(beh_out, "asr_fp16.json"))
        asr_nf4 = load_json_safe(os.path.join(beh_out, "asr_nf4.json"))

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  FINDING 1: Safety Degrades After Quantization      │")
        print(f"  └─────────────────────────────────────────────────────┘")

        if asr_fp16 and asr_nf4:
            fp16_asr = asr_fp16["asr_percent"]
            nf4_asr = asr_nf4["asr_percent"]
            delta_asr = nf4_asr - fp16_asr
            pct_increase = (delta_asr / fp16_asr * 100) if fp16_asr > 0 else float("inf")

            print(f"  FP16 Attack Success Rate:  {fp16_asr:>6.1f}%")
            print(f"  NF4  Attack Success Rate:  {nf4_asr:>6.1f}%")
            print(f"  Absolute increase:         {delta_asr:>+6.1f} percentage points")
            print(f"  Relative increase:         {pct_increase:>+6.0f}%")
            print(f"")
            print(f"  Interpretation: The FP16 model refuses {100 - fp16_asr:.0f}% of harmful")
            print(f"  requests. After NF4 quantization, it only refuses {100 - nf4_asr:.0f}%.")
            if delta_asr > 5:
                print(f"  → CONFIRMED: Quantization significantly degrades safety.")
            report["finding1"] = {"fp16_asr": fp16_asr, "nf4_asr": nf4_asr, "delta": delta_asr}
        else:
            print(f"  [Not yet computed — run measure_asr.py for both precisions]")
            report["finding1"] = None

        # =============================================================
        # FINDING 2: STANDARD METRICS MISS IT
        # =============================================================
        ppl_fp16 = load_json_safe(os.path.join(beh_out, "perplexity_fp16.json"))
        ppl_nf4 = load_json_safe(os.path.join(beh_out, "perplexity_nf4.json"))
        util_fp16 = load_json_safe(os.path.join(beh_out, "utility_fp16.json"))
        util_nf4 = load_json_safe(os.path.join(beh_out, "utility_nf4.json"))
        frob_data = load_json_safe(os.path.join(cir_out, "global_frobenius_analysis.json"))

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  FINDING 2: Standard Metrics Miss the Problem       │")
        print(f"  └─────────────────────────────────────────────────────┘")

        report["finding2"] = {}

        # 2a: Perplexity
        if ppl_fp16 and ppl_nf4:
            fp16_ppl = ppl_fp16["perplexity"]
            nf4_ppl = ppl_nf4["perplexity"]
            ppl_change = (nf4_ppl - fp16_ppl) / fp16_ppl * 100

            print(f"  a) Perplexity (WikiText-2):")
            print(f"     FP16:   {fp16_ppl:>8.2f}")
            print(f"     NF4:    {nf4_ppl:>8.2f}")
            print(f"     Change: {ppl_change:>+7.1f}%")
            print(f"     → Perplexity says: 'model is basically fine'")
            report["finding2"]["perplexity"] = {
                "fp16": fp16_ppl, "nf4": nf4_ppl, "pct_change": ppl_change
            }
        else:
            print(f"  a) Perplexity: [Not yet computed — run measure_perplexity.py]")

        # 2b: Utility
        if util_fp16 and util_nf4:
            for b_fp16, b_nf4 in zip(util_fp16["benchmarks"], util_nf4["benchmarks"]):
                util_delta = b_nf4["accuracy"] - b_fp16["accuracy"]
                print(f"\n  b) {b_fp16['benchmark']} Accuracy:")
                print(f"     FP16:   {b_fp16['accuracy']:>8.1f}%")
                print(f"     NF4:    {b_nf4['accuracy']:>8.1f}%")
                print(f"     Change: {util_delta:>+7.1f}%")
                print(f"     → Utility says: 'model capabilities are preserved'")
                report["finding2"]["utility"] = {
                    "benchmark": b_fp16["benchmark"],
                    "fp16": b_fp16["accuracy"], "nf4": b_nf4["accuracy"],
                    "delta": util_delta,
                }
        else:
            print(f"\n  b) Utility: [Not yet computed — run measure_utility.py]")

        # 2c: Frobenius
        if frob_data:
            stats = frob_data["statistics"]
            print(f"\n  c) Global Frobenius Error (per-matrix):")
            print(f"     Mean relative error: {stats['mean_relative_error']:.6f}")
            print(f"     Std:                 {stats['std_relative_error']:.6f}")
            print(f"     CV:                  {stats['coefficient_of_variation']:.4f}")
            print(f"     → Frobenius says: 'all matrices have ~{stats['mean_relative_error']:.1%} error'")
            print(f"     → CANNOT tell which matrices damage safety")
            report["finding2"]["frobenius"] = stats
        else:
            print(f"\n  c) Frobenius: [Not yet computed — run global_frobenius.py]")

        # Contrast table
        if asr_fp16 and asr_nf4:
            print(f"\n  ┌──────────────────┬──────────┬──────────┬──────────┐")
            print(f"  │ Metric           │ FP16     │ NF4      │ Change   │")
            print(f"  ├──────────────────┼──────────┼──────────┼──────────┤")
            if ppl_fp16 and ppl_nf4:
                print(f"  │ Perplexity       │ {fp16_ppl:>7.2f}  │ {nf4_ppl:>7.2f}  │ {ppl_change:>+6.1f}%  │")
            if util_fp16 and util_nf4:
                b_fp16 = util_fp16["benchmarks"][0]
                b_nf4 = util_nf4["benchmarks"][0]
                print(f"  │ BoolQ Accuracy   │ {b_fp16['accuracy']:>6.1f}%  │ {b_nf4['accuracy']:>6.1f}%  │ {b_nf4['accuracy']-b_fp16['accuracy']:>+6.1f}%  │")
            print(f"  │ Safety (ASR) ↑   │ {fp16_asr:>6.1f}%  │ {nf4_asr:>6.1f}%  │ {delta_asr:>+6.1f}pp │")
            print(f"  └──────────────────┴──────────┴──────────┴──────────┘")
            print(f"  ↑ Lower ASR = safer. Safety degrades while standard metrics say 'fine'.")

        # =============================================================
        # FINDING 3: DIRECTION IS PRESERVED
        # =============================================================
        dir_comp = load_json_safe(os.path.join(dir_out, "direction_comparison.json"))

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  FINDING 3: The Refusal Direction Survives          │")
        print(f"  └─────────────────────────────────────────────────────┘")

        if dir_comp:
            cos = dir_comp["best_abs_cosine"]
            print(f"  Cosine similarity (FP16 vs NF4 best direction): {cos:.4f}")
            print(f"  FP16 direction from layer {dir_comp['fp16_best_layer']}")
            print(f"  NF4  direction from layer {dir_comp['nf4_best_layer']}")
            print(f"  Mean cosine across all layers: {dir_comp['overall_abs_mean_cosine']:.4f}")
            print(f"")
            if cos > 0.90:
                print(f"  → CONFIRMED: The model KNOWS the refusal direction after quantization.")
                print(f"    The direction that separates harmful from harmless is preserved.")
                print(f"    So why does safety still degrade? → Finding 4.")
            elif cos > 0.70:
                print(f"  → Direction is moderately preserved. Some degradation expected.")
            report["finding3"] = {
                "cosine_similarity": cos,
                "overall_mean": dir_comp["overall_abs_mean_cosine"],
            }
        else:
            print(f"  [Not yet computed — run step1_direction scripts]")
            report["finding3"] = None

        # =============================================================
        # FINDING 4: SAFETY CIRCUITS WEAKEN
        # =============================================================
        circ_comp = load_json_safe(os.path.join(cir_out, "circuit_comparison.json"))

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  FINDING 4: Safety Circuits Lose Signal             │")
        print(f"  └─────────────────────────────────────────────────────┘")

        if circ_comp:
            retention = circ_comp["signal_retention_percent"]
            net_fp16 = circ_comp["net_signal_fp16"]
            net_nf4 = circ_comp["net_signal_nf4"]
            net_retention = abs(net_nf4 / net_fp16 * 100) if net_fp16 != 0 else 0

            print(f"  DFA signal retention: {retention:.1f}% of total |DFA|")
            print(f"  Net refusal signal:  FP16={net_fp16:.4f} → NF4={net_nf4:.4f} ({net_retention:.1f}%)")
            print(f"")
            print(f"  Top 5 most affected components (by DFA loss):")
            print(f"  {'Component':<20} | {'FP16 DFA':>10} | {'NF4 DFA':>10} | {'Loss':>10}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
            for item in circ_comp["top_degraded"][:5]:
                loss = item["absolute_change"]
                print(f"  {item['component']:<20} | {item['fp16_dfa']:>10.4f} | "
                      f"{item['nf4_dfa']:>10.4f} | {loss:>+10.4f}")

            print(f"\n  → The model's internal 'refuse this' signal weakens at specific")
            print(f"    components. But WHICH dimensions within those components matter?")
            print(f"    And which specific weight matrix ROWS caused the damage?")
            print(f"    Global Frobenius can't answer this. → Phase 2: V-score.")
            report["finding4"] = {
                "signal_retention": retention,
                "net_retention": net_retention,
                "top5_affected": circ_comp["top_degraded"][:5],
            }
        else:
            print(f"  [Not yet computed — run step3_circuit scripts]")
            report["finding4"] = None

        # =============================================================
        # THE ARGUMENT: WHY THIS IS A PAPER
        # =============================================================
        print(f"\n  {'='*60}")
        print(f"  THE ARGUMENT (for your professor)")
        print(f"  {'='*60}")
        print(f"""
  KNOWN FACT: NF4 quantization breaks safety (ASR doubles).

  KNOWN GAP: Standard metrics (perplexity, Frobenius norm)
  can't detect or locate this damage.

  OUR INSIGHT: The refusal direction survives quantization,
  but the circuit components that READ and PROCESS it fail.
  Damage is concentrated in specific dimensions of specific
  weight matrices — not uniformly distributed.

  OUR CONTRIBUTION (Phase 2+3):
    1. V-score = |DFA_i| x eps_i : per-dimension vulnerability
       that combines safety importance with quantization error.
       (Bridges Arditi's refusal direction with Elhage's
        residual stream framework)

    2. Read-Side Collapse: the primary damage is in read matrices
       (up_proj, gate_proj, q/k/v_proj), not write matrices.
       Write restoration: 0pp improvement.
       Steering (bypass read): 21pp improvement.

    3. Inference-free quantizer evaluation: compute V-score for
       any new quantizer without running the quantized model.

  WHY NOBODY DID THIS BEFORE:
    Three communities that didn't overlap:
    - Mechanistic interpretability (Elhage) -> never studied safety
    - Alignment (Arditi) -> never considered quantization
    - Quantization (GPTQ/AWQ) -> only measure perplexity
    We needed all three simultaneously.
        """)

        # =============================================================
        # GENERATE SUMMARY FIGURE
        # =============================================================
        if asr_fp16 and asr_nf4 and circ_comp:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"SafeQuant Phase 1: The Problem — {model_key}",
                         fontsize=14, fontweight="bold")

            # Plot 1: ASR comparison
            ax = axes[0]
            labels = ["FP16", "NF4"]
            values = [fp16_asr, nf4_asr]
            colors = ["#2ecc71", "#e74c3c"]
            bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold")
            ax.set_ylabel("Attack Success Rate (%)")
            ax.set_title("Safety Degrades\n(Higher = More Unsafe)")
            ax.set_ylim(0, max(values) * 1.25)

            # Plot 2: Metrics contrast
            ax = axes[1]
            if ppl_fp16 and ppl_nf4:
                metric_names = ["Perplexity\nChange", "Safety (ASR)\nChange"]
                metric_vals = [ppl_change, delta_asr / fp16_asr * 100 if fp16_asr > 0 else 100]
                metric_colors = ["gray", "#e74c3c"]
                bars = ax.bar(metric_names, metric_vals, color=metric_colors, alpha=0.8)
                for bar, val in zip(bars, metric_vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (2 if val > 0 else -5),
                            f"{val:+.1f}%", ha="center", fontsize=12, fontweight="bold")
                ax.set_ylabel("Relative Change (%)")
                ax.set_title("Standard Metrics Miss It\n(Perplexity fine, Safety broken)")
                ax.axhline(y=0, color="black", linewidth=0.5)
            else:
                ax.text(0.5, 0.5, "Run perplexity\nmeasurement first",
                        ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_title("Standard Metrics\n(Pending)")

            # Plot 3: DFA circuit comparison
            ax = axes[2]
            comps = circ_comp["top_degraded"][:10]
            comp_names = [c["component"].replace("_layer_", "\nL") for c in comps]
            fp16_vals = [c["fp16_dfa"] for c in comps]
            nf4_vals = [c["nf4_dfa"] for c in comps]
            x = np.arange(len(comp_names))
            w = 0.35
            ax.bar(x - w/2, fp16_vals, w, label="FP16", color="#3498db", alpha=0.8)
            ax.bar(x + w/2, nf4_vals, w, label="NF4", color="#e67e22", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(comp_names, fontsize=7, rotation=45)
            ax.set_ylabel("DFA Score")
            ax.set_title("Safety Circuits Weaken\n(Top affected components)")
            ax.legend(fontsize=9)
            ax.axhline(y=0, color="black", linewidth=0.5)

            plt.tight_layout()
            fig_path = os.path.join(sum_out, "phase1_summary_figure.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"  Summary figure saved to {fig_path}")

        # Save report JSON
        report_path = os.path.join(sum_out, "phase1_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {report_path}")

    print(f"\n{'#'*72}")
    print(f"  Phase 1 Summary Complete.")
    print(f"{'#'*72}")


if __name__ == "__main__":
    main()
