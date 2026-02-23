"""
SafeQuant Phase 2 — Comprehensive evaluation of all repair strategies.
Measures ASR (safety) and BoolQ accuracy (utility) for each condition.

Run: python phase2_solution/repair/quick_test.py --model qwen1.5b --n_eval 100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse, json, torch, time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DTYPE, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS


def measure_asr(model, tokenizer, prompts):
    comply = 0
    for prompt in tqdm(prompts, desc="    ASR"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if not any(response.strip().lower().startswith(p.lower()) for p in REFUSAL_PREFIXES):
            comply += 1
    return comply / len(prompts) * 100


def measure_utility(model, tokenizer, n_samples=100):
    """BoolQ accuracy as utility metric."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google/boolq", split="validation")
        ds = ds.select(range(min(n_samples, len(ds))))
    except Exception as e:
        print(f"    Could not load BoolQ: {e}")
        return None

    correct = 0
    for item in tqdm(ds, desc="    Utility"):
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


def load_nf4(model_name):
    bnb = BitsAndBytesConfig(**NF4_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def load_fp16(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def eval_condition(name, model, tokenizer, prompts, results, n_utility=100):
    """Evaluate ASR and utility for a condition."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    asr = measure_asr(model, tokenizer, prompts)
    print(f"    ASR: {asr:.1f}%")

    util = measure_utility(model, tokenizer, n_utility)
    if util is not None:
        print(f"    Utility (BoolQ): {util:.1f}%")

    results[name] = {"asr": asr, "utility": util}
    return asr, util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_utility", type=int, default=100)
    args = parser.parse_args()

    model_name = MODELS[args.model]

    # Load eval prompts
    with open(os.path.join(DATA_DIR, "eval_harmful.json")) as f:
        prompts = json.load(f)[:args.n_eval]

    # Load vulnerability scores for layer ranking
    diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagnosis", "outputs", args.model)
    with open(os.path.join(diag_dir, "vulnerability_scores.json")) as f:
        vuln_data = json.load(f)

    # Rank components and layers by total V-score
    comp_totals = {}
    key = "top_vulnerable" if "top_vulnerable" in vuln_data else "top_200_vulnerable"
    for item in vuln_data[key]:
        comp = item["component"]
        comp_totals[comp] = comp_totals.get(comp, 0) + item["vulnerability_score"]

    layer_scores = {}
    for name, score in comp_totals.items():
        l = int(name.split("_")[-1])
        layer_scores[l] = layer_scores.get(l, 0) + score
    layer_ranking = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop layers by V-score:")
    for l, s in layer_ranking[:10]:
        print(f"  Layer {l}: {s:.2f}")

    # Load refusal direction for steering
    dir_dir = os.path.join(os.path.dirname(__file__), "..", "..", "phase1_problem",
                           "step1_direction", "outputs", args.model)
    r_hat = torch.load(os.path.join(dir_dir, "best_direction_fp16.pt"),
                       map_location="cpu")["direction_normalized"].flatten()

    results = {}
    steer_layers = [l for l, s in layer_ranking[:5]]
    print(f"\nSteering layers: {steer_layers}")
    print(f"Eval prompts: {args.n_eval}, Utility samples: {args.n_utility}")

    # ===== 1. FP16 BASELINE =====
    print("\nLoading FP16 model...")
    model, tok = load_fp16(model_name)
    eval_condition("FP16 (baseline)", model, tok, prompts, results, args.n_utility)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== 2. NF4 BASELINE =====
    print("\nLoading NF4 model...")
    model, tok = load_nf4(model_name)
    eval_condition("NF4 (baseline)", model, tok, prompts, results, args.n_utility)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== 3-5. ACTIVATION STEERING =====
    for alpha in [1.0, 2.0, 4.0]:
        model, tok = load_nf4(model_name)
        hooks = []

        def make_hook(strength):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                r = r_hat.to(hidden.device).to(hidden.dtype)
                hidden[:, -1, :] += strength * r
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook_fn

        for l in steer_layers:
            h = model.model.layers[l].register_forward_hook(make_hook(alpha))
            hooks.append(h)

        eval_condition(f"Steer α={alpha} (layers {steer_layers})", model, tok, prompts, results, args.n_utility)

        for h in hooks:
            h.remove()
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== 6-8. DIFFERENTIAL QUANTIZATION (full layer FP16) =====
    print("\nLoading full FP16 model for differential quant...")
    fp16_full = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True)
    total_layers = fp16_full.config.num_hidden_layers

    for n_layers_fp16 in [3, 5, 7]:
        protect = [l for l, s in layer_ranking[:n_layers_fp16]]

        model, tok = load_nf4(model_name)

        for l in protect:
            nf4_layer = model.model.layers[l]
            fp16_layer = fp16_full.model.layers[l]
            device = next(nf4_layer.parameters()).device

            # Replace ALL attention matrices
            for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                fp16_mod = getattr(fp16_layer.self_attn, attr)
                new_mod = torch.nn.Linear(
                    fp16_mod.in_features, fp16_mod.out_features,
                    bias=False, device=device)
                new_mod.weight = torch.nn.Parameter(
                    fp16_mod.weight.data.to(device).to(DTYPE), requires_grad=False)
                setattr(nf4_layer.self_attn, attr, new_mod)

            # Replace ALL MLP matrices
            for attr in ['gate_proj', 'up_proj', 'down_proj']:
                fp16_mod = getattr(fp16_layer.mlp, attr)
                new_mod = torch.nn.Linear(
                    fp16_mod.in_features, fp16_mod.out_features,
                    bias=False, device=device)
                new_mod.weight = torch.nn.Parameter(
                    fp16_mod.weight.data.to(device).to(DTYPE), requires_grad=False)
                setattr(nf4_layer.mlp, attr, new_mod)

        pct = n_layers_fp16 / total_layers * 100
        eval_condition(
            f"DiffQuant {n_layers_fp16}L FP16 ({pct:.0f}% layers, rest NF4)",
            model, tok, prompts, results, args.n_utility)

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    del fp16_full; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== SUMMARY TABLE =====
    print(f"\n{'='*75}")
    print(f"  COMPREHENSIVE RESULTS: {args.model}")
    print(f"{'='*75}")
    print(f"  {'Condition':<45} | {'ASR':>7} | {'Utility':>7} | {'Safe?':>5}")
    print(f"  {'-'*45}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")

    for name, r in results.items():
        asr_s = f"{r['asr']:.1f}%" if r['asr'] is not None else "N/A"
        util_s = f"{r['utility']:.1f}%" if r['utility'] is not None else "N/A"

        # Flag if safer than NF4
        nf4_asr = results.get("NF4 (baseline)", {}).get("asr", 100)
        safer = "★" if r['asr'] is not None and r['asr'] < nf4_asr else ""
        print(f"  {name:<45} | {asr_s:>7} | {util_s:>7} | {safer:>5}")

    # Key metrics
    fp16_asr = results.get("FP16 (baseline)", {}).get("asr", 17)
    nf4_asr = results.get("NF4 (baseline)", {}).get("asr", 33)
    safety_gap = nf4_asr - fp16_asr

    print(f"\n  Safety gap (NF4 - FP16): {safety_gap:.1f}pp")
    print(f"\n  Recovery analysis:")
    for name, r in results.items():
        if name in ["FP16 (baseline)", "NF4 (baseline)"]:
            continue
        if r['asr'] is not None and r['asr'] < nf4_asr:
            recovery = (nf4_asr - r['asr']) / safety_gap * 100
            util_delta = ""
            nf4_util = results.get("NF4 (baseline)", {}).get("utility")
            if r['utility'] is not None and nf4_util is not None:
                util_delta = f", utility Δ = {r['utility'] - nf4_util:+.1f}pp"
            print(f"    {name}: recovers {recovery:.0f}% of safety gap{util_delta}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "evaluate", "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "comprehensive_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    main()