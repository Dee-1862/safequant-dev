import argparse, gc, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.seeds import set_global_seed, DEFAULT_SEED
    SEED = set_global_seed(DEFAULT_SEED)
except ModuleNotFoundError:
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

from diagnostics.layer_routing import _get_inner
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts
from core.evaluation import classify_batch, _extract_behavior


def _format_prompt(p, tok):
    text = p.get("prompt") or p.get("vanilla") or p.get("behavior") or "" if isinstance(p, dict) else str(p)
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return text


def capture_quant_activations(q_model, q_tok, prompts, layer_idx):
    """Forward each prompt through the quant model; capture the residual-stream
    activation at the end of `layer_idx` (full sequence, not last-token)."""
    inner = _get_inner(q_model)
    device = next(q_model.parameters()).device
    captured = []
    captured_attn_masks = []
    buf = {}
    def hook(module, inputs, output):
        r = output[0] if isinstance(output, (tuple, list)) else output
        buf['x'] = r.detach()
    h = inner.layers[layer_idx].register_forward_hook(hook)
    try:
        for p in prompts:
            text = _format_prompt(p, q_tok)
            inputs = q_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            buf.clear()
            with torch.no_grad():
                _ = q_model(**inputs)
            captured.append(buf['x'].cpu())
            captured_attn_masks.append(inputs.input_ids.cpu())
    finally:
        h.remove()
    return captured, captured_attn_masks


def generate_with_layer_swap(fp32_model, fp32_tok, prompts, layer_idx, quant_acts, quant_input_ids, max_new_tokens=256):
    """For each prompt, generate from the FP32 model BUT replace the residual
    stream at layer_idx with the quantized model's captured activation.

    Implementation: install a hook that, on its first call (the prefill pass),
    overrides the output with the pre-captured activation. Subsequent
    calls (per-token decode) pass through unchanged.
    """
    inner = _get_inner(fp32_model)
    device = next(fp32_model.parameters()).device
    fp_responses = []
    for i, p in enumerate(prompts):
        text = _format_prompt(p, fp32_tok)
        inputs = fp32_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        q_act = quant_acts[i].to(device).to(next(fp32_model.parameters()).dtype)
        # Truncate or pad q_act to match prompt_len if tokenization differs
        # (shouldn't, but be defensive)
        if q_act.shape[1] != prompt_len:
            min_len = min(q_act.shape[1], prompt_len)
            q_act = q_act[:, :min_len, :]
        first_call = [True]
        def hook(module, inputs_, output):
            # Only replace on the prefill pass (when output seq_len == prompt_len)
            r = output[0] if isinstance(output, (tuple, list)) else output
            if first_call[0] and r.shape[1] == prompt_len:
                first_call[0] = False
                new_r = q_act.to(r.dtype)
                # Make sure shapes match exactly (B=1, T, D)
                if new_r.shape == r.shape:
                    if isinstance(output, tuple):
                        return (new_r,) + output[1:]
                    return new_r
            return output
        h = inner.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                gen = fp32_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=fp32_tok.eos_token_id,
                )
            response = fp32_tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()
            fp_responses.append(response)
        finally:
            h.remove()
    return fp_responses


def baseline_generate(model, tok, prompts, max_new_tokens=256):
    out = []
    device = next(model.parameters()).device
    for p in prompts:
        text = _format_prompt(p, tok)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        out.append(tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--swap_layers", default="4,11,16,22,29", help="Comma-sep layers to swap at")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.output is None:
        args.output = f"outputs/activation_swap_{args.model}_{args.quantizer}.json"

    print(f"=== Activation swap | {args.model} / {args.quantizer} | seed={SEED}")
    swap_layers = [int(x) for x in args.swap_layers.split(",")]
    print(f"  swap layers: {swap_layers}")

    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]

    # Capture quant activations at each swap layer (sequential, one layer per pass)
    print(f"\n[{args.quantizer}] capturing per-layer activations on eval prompts ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    quant_acts_by_layer = {}
    for L in swap_layers:
        print(f"  capturing at L={L} ...")
        acts, _ = capture_quant_activations(q_model, q_tok, items, L)
        quant_acts_by_layer[L] = acts
    # Also generate quant baseline ASR
    print(f"  generating quant baseline responses ...")
    quant_responses = baseline_generate(q_model, q_tok, items, max_new_tokens=args.max_new_tokens)
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # FP32 baseline + swap variants
    print(f"\n[FP32] loading ...")
    fp32_model, fp32_tok = load_fp32(args.model)

    print(f"  generating FP32 baseline responses ...")
    fp32_responses = baseline_generate(fp32_model, fp32_tok, items, max_new_tokens=args.max_new_tokens)

    all_responses = {"fp32_baseline": fp32_responses, "quant_baseline": quant_responses}
    for L in swap_layers:
        print(f"\n  generating with quant-L{L} activation swapped INTO FP32 ...")
        swap_responses = generate_with_layer_swap(fp32_model, fp32_tok, items, L, quant_acts_by_layer[L], None, max_new_tokens=args.max_new_tokens)
        all_responses[f"swap_q_into_fp32_L{L}"] = swap_responses
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # Classify all
    print(f"\n  classifying {len(all_responses)*args.n_eval} responses ...")
    keys = list(all_responses.keys())
    flat_b, flat_r = [], []
    for k in keys:
        flat_b.extend(behaviors); flat_r.extend(all_responses[k])
    flags = classify_batch(flat_b, flat_r, device=device)
    n = args.n_eval
    asr = {k: 100.0 * sum(flags[i*n:(i+1)*n]) / n for i, k in enumerate(keys)}

    fp_b = asr["fp32_baseline"]
    q_b  = asr["quant_baseline"]
    print()
    print("="*70)
    print(f"  Activation swap -- {args.model} / {args.quantizer}")
    print(f"  FP32 baseline ASR: {fp_b:.1f}%   Quant baseline ASR: {q_b:.1f}%")
    print(f"  Damage: {q_b - fp_b:+.1f}pp")
    print("="*70)
    print(f"  {'Condition':<36} {'ASR':>7}  {'Delta from FP32':>13}")
    print("  " + "-"*60)
    for L in swap_layers:
        a = asr[f"swap_q_into_fp32_L{L}"]
        d = a - fp_b
        if d > 5:
            interp = "consumption-damage"
        elif d < -3:
            interp = "activation-damage"
        else:
            interp = "neutral"
        print(f"  swap L{L} q->FP32                   {a:>6.1f}% {d:>+11.1f}pp   ({interp})")

    out = {
        "model": args.model, "quantizer": args.quantizer, "seed": SEED,
        "swap_layers": swap_layers, "n_eval": args.n_eval,
        "fp32_baseline_asr": fp_b, "quant_baseline_asr": q_b,
        "swap_asr": {f"L{L}": asr[f"swap_q_into_fp32_L{L}"] for L in swap_layers},
        "swap_delta_from_fp32": {f"L{L}": asr[f"swap_q_into_fp32_L{L}"] - fp_b for L in swap_layers},
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
