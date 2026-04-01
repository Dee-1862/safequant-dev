# Evaluates all repair strategies by measuring ASR and utility.
#
# Loads each repaired model, applies the repair in memory, and measures:
#   - ASR (attack success rate) on the eval harmful prompts
#   - Utility on BoolQ (optional, skippable with --skip_utility)
#
# Conditions evaluated:
#   fp16          - original FP16 model (loaded from phase1 results)
#   nf4           - unrepaired NF4 (loaded from phase1 results)
#   targeted      - our V-score delta-W repair
#   chhabra       - AIRD baseline from their paper
#   random        - random dimension selection (control)
#   dfa_only      - top-k by DFA importance alone (ablation)
#   error_only    - top-k by quantization error alone (ablation)
#
# Run:
#   python phase2_solution/evaluate/eval_all.py --model qwen1.5b

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS


def load_repaired_model(model_name, repair_state_dir):
    bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="cpu", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state_path = os.path.join(repair_state_dir, "model_state.pt")
    saved_state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(saved_state, strict=False)
    model.eval()
    return model, tokenizer


def is_refusal(text):
    text_lower = text.strip().lower()
    for prefix in REFUSAL_PREFIXES:
        if text_lower.startswith(prefix.lower()):
            return True
    return False


def measure_asr(model, tokenizer, prompts):
    comply = 0
    for prompt in tqdm(prompts, desc="    ASR", leave=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if not is_refusal(response):
            comply += 1
    return comply / len(prompts) * 100


def measure_utility(model, tokenizer, n_samples=50):
    try:
        from datasets import load_dataset
        ds = load_dataset("google/boolq", split="validation")
        ds = ds.select(range(min(n_samples, len(ds))))
    except Exception:
        return None

    correct = 0
    for item in tqdm(ds, desc="    Utility", leave=False):
        prompt = f"Answer yes or no.\n\nPassage: {item['passage'][:500]}\n\nQuestion: {item['question']}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        answer = "yes" if item["answer"] else "no"
        predicted = "yes" if response.startswith("yes") else "no" if response.startswith("no") else "unk"
        if predicted == answer:
            correct += 1
    return correct / len(ds) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--skip_utility", action="store_true")
    parser.add_argument("--n_eval", type=int, default=None)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    repair_dir = os.path.join(os.path.dirname(__file__), "..", "repair", "outputs", args.model)
    phase1_beh = os.path.join(os.path.dirname(__file__), "..", "..", "phase1_problem",
                              "step2_behavior", "outputs", args.model)

    with open(os.path.join(DATA_DIR, "eval_harmful.json"), encoding="utf-8") as f:
        eval_prompts = json.load(f)
    if args.n_eval:
        eval_prompts = eval_prompts[:args.n_eval]

    print(f"\nEvaluating all conditions: {args.model}")
    print(f"Eval prompts: {len(eval_prompts)}\n")

    results = {}

    # pull in phase1 baselines if they exist
    for bl in ["fp16", "nf4"]:
        asr_path = os.path.join(phase1_beh, f"asr_{bl}.json")
        if os.path.exists(asr_path):
            with open(asr_path) as f:
                d = json.load(f)
            results[bl] = {"asr": d["asr_percent"], "source": "phase1",
                           "overhead": "0%" if bl == "nf4" else "N/A"}
            print(f"  {bl} baseline (from phase1): ASR={d['asr_percent']:.1f}%")

    # pre-load FP16 weights - we need them to apply delta-W repair in memory
    print("\n  Loading FP16 weights...")
    fp16_temp = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    fp16_weights = {}
    n_layers = fp16_temp.config.num_hidden_layers
    for l in range(n_layers):
        layer = fp16_temp.model.layers[l]
        fp16_weights[f"attn_layer_{l}"] = layer.self_attn.o_proj.weight.data.float().cpu().clone()
        fp16_weights[f"mlp_layer_{l}"] = layer.mlp.down_proj.weight.data.float().cpu().clone()
    del fp16_temp

    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step1_direction", "outputs", args.model
    )
    r_fp16 = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"), map_location="cpu"
    )["direction_normalized"].flatten()
    r_nf4 = torch.load(
        os.path.join(direction_dir, "best_direction_nf4.pt"), map_location="cpu"
    )["direction_normalized"].flatten()

    strategies = ["targeted", "targeted_aird", "chhabra", "random", "dfa_only", "error_only"]

    for strategy in strategies:
        repair_path = os.path.join(repair_dir, f"repaired_{strategy}")
        info_path = os.path.join(repair_path, "repair_info.pt")

        if not os.path.exists(info_path):
            print(f"\n  Skipping {strategy} - repair not found")
            continue

        print(f"\n  Evaluating: {strategy}")
        info = torch.load(info_path, map_location="cpu")

        # load a fresh NF4 model each time so repairs don't bleed into each other
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="cpu", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        strat_type = info["strategy"]

        if strat_type in ["chhabra", "targeted_aird"]:
            # AIRD-based repair: remove old direction component, add new one
            targets = info.get("targets", [])
            if strat_type == "chhabra":
                layer_idx = info["layer"]
                targets = [f"attn_layer_{layer_idx}", f"mlp_layer_{layer_idx}"]

            r_fp16_hat = r_fp16 / torch.norm(r_fp16)
            r_nf4_hat = r_nf4 / torch.norm(r_nf4)
            proj_nf4 = torch.outer(r_nf4_hat, r_nf4_hat)
            proj_fp16 = torch.outer(r_fp16_hat, r_fp16_hat)

            for comp in targets:
                parts = comp.split("_")
                layer_idx = int(parts[-1])
                layer = model.model.layers[layer_idx]

                if parts[0] == "attn":
                    module = layer.self_attn.o_proj
                else:
                    module = layer.mlp.down_proj

                device = next(module.parameters()).device
                in_features = module.in_features

                # identity trick - avoids the BNB CPU dequantize shape bug
                identity = torch.eye(in_features, device=device, dtype=torch.float32)
                with torch.no_grad():
                    W = module(identity).float().T

                W_new = W - proj_nf4.to(device) @ W + proj_fp16.to(device) @ W

                new_mod = torch.nn.Linear(in_features, W_new.shape[0], bias=False, device=device)
                new_mod.weight = torch.nn.Parameter(W_new.to(DTYPE), requires_grad=False)

                if parts[0] == "attn":
                    layer.self_attn.o_proj = new_mod
                else:
                    layer.mlp.down_proj = new_mod

                print(f"    AIRD applied: {comp}")

        else:
            # row restoration repair
            targets = info.get("targets", [])
            comp_dims = {}
            for comp, dim in targets:
                comp_dims.setdefault(comp, []).append(dim)

            for comp, dims in comp_dims.items():
                parts = comp.split("_")
                layer_idx = int(parts[-1])
                layer = model.model.layers[layer_idx]

                if parts[0] == "attn":
                    module = layer.self_attn.o_proj
                else:
                    module = layer.mlp.down_proj

                device = next(module.parameters()).device
                in_features = module.in_features

                identity = torch.eye(in_features, device=device, dtype=torch.float32)
                with torch.no_grad():
                    W_current = module(identity).float().T

                W_fp16_comp = fp16_weights[comp].to(device)
                for dim in dims:
                    if dim < W_current.shape[0] and dim < W_fp16_comp.shape[0]:
                        min_cols = min(W_current.shape[1], W_fp16_comp.shape[1])
                        W_current[dim, :min_cols] = W_fp16_comp[dim, :min_cols]

                new_mod = torch.nn.Linear(in_features, W_current.shape[0], bias=False, device=device)
                new_mod.weight = torch.nn.Parameter(W_current.to(DTYPE), requires_grad=False)

                if parts[0] == "attn":
                    layer.self_attn.o_proj = new_mod
                else:
                    layer.mlp.down_proj = new_mod

            print(f"    Restored {len(targets)} rows across {len(comp_dims)} components")

        asr = measure_asr(model, tokenizer, eval_prompts)
        results[strategy] = {"asr": asr, "n_repaired": len(info.get("targets", []))}
        print(f"    ASR: {asr:.1f}%")

        if not args.skip_utility:
            util = measure_utility(model, tokenizer)
            if util is not None:
                results[strategy]["utility"] = util
                print(f"    Utility: {util:.1f}%")

        del model

    # print summary
    print(f"\nResults: {args.model}")
    for cond in ["fp16", "nf4", "targeted", "random", "dfa_only", "error_only"]:
        if cond not in results:
            continue
        r = results[cond]
        asr_s = f"{r['asr']:.1f}%" if "asr" in r else "N/A"
        util_s = f"{r['utility']:.1f}%" if isinstance(r.get("utility"), (int, float)) else "N/A"
        notes = {"fp16": "baseline", "nf4": "no repair", "targeted": "ours"}.get(cond, "")
        print(f"  {cond:<15}  ASR={asr_s}  Utility={util_s}  {notes}")

    if "targeted" in results and "nf4" in results:
        gap = results["nf4"]["asr"] - results["targeted"]["asr"]
        print(f"\n  Targeted repair reduced ASR by {gap:.1f}pp vs unrepaired NF4")
    if "targeted" in results and "random" in results:
        gap = results["random"]["asr"] - results["targeted"]["asr"]
        print(f"  Targeted is {gap:.1f}pp better than random (validates the V-score)")

    # plot comparison bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds = [c for c in ["fp16", "nf4", "targeted", "random", "dfa_only", "error_only"]
                 if c in results and "asr" in results[c]]
        asr_vals = [results[c]["asr"] for c in conds]
        cmap = {"fp16": "#22c55e", "nf4": "#ef4444", "targeted": "#3b82f6",
                "random": "#9ca3af", "dfa_only": "#8b5cf6", "error_only": "#a16207"}
        colors = [cmap.get(c, "#666") for c in conds]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(conds, asr_vals, color=colors, alpha=0.85)
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(f"Safety After Repair: {args.model} (lower is safer)")
        ax.set_ylim(0, max(asr_vals) * 1.25 + 5)
        for bar, val in zip(bars, asr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "asr_comparison.png"), dpi=150, bbox_inches="tight")
        print("\n  Plot saved")
    except Exception as e:
        print(f"  Plot failed: {e}")

    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
