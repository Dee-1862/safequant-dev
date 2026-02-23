"""
Quick diagnostic: test multiple repair strategies in-memory.
Run: python phase2_solution/repair/quick_test.py --model qwen1.5b --n_eval 50
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DTYPE, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS


def measure_asr(model, tokenizer, prompts):
    comply = 0
    from tqdm import tqdm
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


def load_nf4(model_name):
    bnb = BitsAndBytesConfig(**NF4_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n_eval", type=int, default=50)
    args = parser.parse_args()

    model_name = MODELS[args.model]

    with open(os.path.join(DATA_DIR, "eval_harmful.json")) as f:
        prompts = json.load(f)[:args.n_eval]

    # Load vulnerability scores
    diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagnosis", "outputs", args.model)
    with open(os.path.join(diag_dir, "vulnerability_scores.json")) as f:
        vuln_data = json.load(f)

    # Rank components by total V-score
    comp_totals = {}
    for item in vuln_data["top_200_vulnerable"]:
        comp = item["component"]
        comp_totals[comp] = comp_totals.get(comp, 0) + item["vulnerability_score"]
    comp_ranking = sorted(comp_totals.items(), key=lambda x: x[1], reverse=True)

    print(f"\nComponent ranking by total V-score:")
    for name, score in comp_ranking[:10]:
        print(f"  {name:<20}: {score:.2f}")

    # Load FP16 weights
    print("\nLoading FP16 weights...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True)
    fp16_weights = {}
    for l in range(fp16_model.config.num_hidden_layers):
        layer = fp16_model.model.layers[l]
        fp16_weights[f"attn_layer_{l}"] = layer.self_attn.o_proj.weight.data.float().cpu().clone()
        fp16_weights[f"mlp_layer_{l}"] = layer.mlp.down_proj.weight.data.float().cpu().clone()
    del fp16_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load refusal direction
    dir_dir = os.path.join(os.path.dirname(__file__), "..", "..", "phase1_problem",
                           "step1_direction", "outputs", args.model)
    r_hat = torch.load(os.path.join(dir_dir, "best_direction_fp16.pt"),
                       map_location="cpu")["direction_normalized"].flatten()

    results = {}

    # ===== TEST 1: NF4 baseline =====
    print(f"\n--- NF4 baseline ---")
    model, tok = load_nf4(model_name)
    asr = measure_asr(model, tok, prompts)
    results["nf4"] = asr
    print(f"    ASR: {asr:.1f}%")
    del model; torch.cuda.empty_cache()

    # ===== TEST 2: Full component replacement (top 5 by V-score) =====
    for n_comps in [3, 5, 10]:
        print(f"\n--- Full component replace: top {n_comps} by V-score ---")
        target_comps = [name for name, score in comp_ranking[:n_comps]]
        print(f"    Targets: {target_comps}")

        model, tok = load_nf4(model_name)
        for comp in target_comps:
            parts = comp.split("_")
            comp_type, layer_idx = parts[0], int(parts[-1])
            layer = model.model.layers[layer_idx]

            if comp_type == "attn":
                module = layer.self_attn.o_proj
            else:
                module = layer.mlp.down_proj

            device = next(module.parameters()).device
            in_f, out_f = module.in_features, module.out_features

            new_mod = torch.nn.Linear(in_f, out_f, bias=False, device=device)
            new_mod.weight = torch.nn.Parameter(
                fp16_weights[comp][:out_f, :in_f].to(device).to(DTYPE),
                requires_grad=False)

            if comp_type == "attn":
                layer.self_attn.o_proj = new_mod
            else:
                layer.mlp.down_proj = new_mod

        asr = measure_asr(model, tok, prompts)
        results[f"full_comp_{n_comps}"] = asr
        print(f"    ASR: {asr:.1f}%")
        del model; torch.cuda.empty_cache()

    # ===== TEST 3: Activation steering (AUSteer-style) =====
    for alpha in [1.0, 2.0, 4.0]:
        print(f"\n--- Activation steering: alpha={alpha}, top 5 layers ---")
        # Pick top layers by V-score
        layer_totals = {}
        for name, score in comp_totals.items():
            l = int(name.split("_")[-1])
            layer_totals[l] = layer_totals.get(l, 0) + score
        top_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        target_layers = [l for l, s in top_layers]
        print(f"    Steering at layers: {target_layers}")

        model, tok = load_nf4(model_name)
        hooks = []

        def make_hook(strength):
            def hook_fn(module, input, output):
                # output is tuple (hidden_states, ...)
                hidden = output[0] if isinstance(output, tuple) else output
                r = r_hat.to(hidden.device).to(hidden.dtype)
                # Add refusal direction at last token
                hidden[:, -1, :] += strength * r
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook_fn

        for l in target_layers:
            h = model.model.layers[l].register_forward_hook(make_hook(alpha))
            hooks.append(h)

        asr = measure_asr(model, tok, prompts)
        results[f"steer_a{alpha}"] = asr
        print(f"    ASR: {asr:.1f}%")

        for h in hooks:
            h.remove()
        del model; torch.cuda.empty_cache()

    # ===== SUMMARY =====
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    for name, asr in sorted(results.items(), key=lambda x: x[1]):
        marker = " ★" if asr < results.get("nf4", 100) else ""
        print(f"  {name:<25}: ASR = {asr:.1f}%{marker}")
    print(f"\n  FP16 baseline: 17.0%")
    print(f"  NF4 baseline:  {results.get('nf4', '?')}%")


if __name__ == "__main__":
    main()