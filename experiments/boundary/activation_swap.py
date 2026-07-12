"""Activation swap control: weight vs activation damage."""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datasets import load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.residuals import get_inner, prompt_text
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def capture_quant_activations(q_model, q_tok, prompts, layer_idx):
    """
    Capture the quantized activations for a given layer.
    Inputs:
        q_model (torch.nn.Module): The quantized model to use for capturing activations.
        q_tok (transformers.PreTrainedTokenizer): The tokenizer to use for capturing activations.
        prompts (list): A list of prompts to capture activations for.
        layer_idx (int): The index of the layer to capture activations for.

    Outputs:
        captured (list): A list of captured activations.
    """
    inner = get_inner(q_model)
    device = next(q_model.parameters()).device
    captured = []
    buf = {}

    def hook(module, inputs, output):
        r = output[0] if isinstance(output, (tuple, list)) else output
        buf["x"] = r.detach()

    h = inner.layers[layer_idx].register_forward_hook(hook)
    try:
        for p in prompts:
            text = prompt_text(p, q_tok)
            inputs = q_tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
            buf.clear()
            with torch.no_grad():
                _ = q_model(**inputs)
            captured.append(buf["x"].cpu())
    finally:
        h.remove()
    return captured


def generate_with_layer_swap(fp32_model, fp32_tok, prompts, layer_idx, quant_acts, max_new_tokens = 256):
    """
    Generate responses for a list of prompts using a model and tokenizer with a layer swap.

    Inputs:
        fp32_model (torch.nn.Module): The model to use for generation.
        fp32_tok (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
        prompts (list): A list of prompts to generate responses for.
        layer_idx (int): The index of the layer to swap.
        quant_acts (list): A list of quantized activations to use for the layer swap.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Outputs:
        fp_responses (list): A list of generated responses.
    """
    # Get the inner model
    inner = get_inner(fp32_model)
    device = next(fp32_model.parameters()).device
    fp_responses = []

    # Generate responses for each prompt
    for i, p in enumerate(prompts):
        text = prompt_text(p, fp32_tok)
        inputs = fp32_tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        q_act = quant_acts[i].to(device).to(next(fp32_model.parameters()).dtype)
        if q_act.shape[1] != prompt_len:
            q_act = q_act[:, : min(q_act.shape[1], prompt_len), :]
        first_call = [True]

        # Hook to swap the quantized activations into the fp32 model
        def hook(module, inputs_, output):
            r = output[0] if isinstance(output, (tuple, list)) else output
            if first_call[0] and r.shape[1] == prompt_len:
                first_call[0] = False
                new_r = q_act.to(r.dtype)
                if new_r.shape == r.shape:
                    if isinstance(output, tuple):
                        return (new_r,) + output[1:]
                    return new_r
            return output

        # Register the hook to the layer
        h = inner.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                gen = fp32_model.generate(
                    **inputs,
                    max_new_tokens = max_new_tokens,
                    do_sample = False,
                    pad_token_id = fp32_tok.eos_token_id,
                )
            fp_responses.append(fp32_tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip())
        finally:
            h.remove()

    # Return the generated responses
    return fp_responses


def baseline_generate(model, tok, prompts, max_new_tokens = 256):
    """
    Generate responses for a list of prompts using a model and tokenizer.

    Inputs:
        model (torch.nn.Module): The model to use for generation.
        tok (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
        prompts (list): A list of prompts to generate responses for.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Outputs:
        out (list): A list of generated responses.
    """
    out = []
    device = next(model.parameters()).device

    # Generate responses for each prompt
    for p in prompts:
        text = prompt_text(p, tok)
        inputs = tok(text, return_tensors = "pt", truncation = True, max_length = 512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = tok.eos_token_id,
            )
        out.append(tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip())
    return out


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required = True)
    ap.add_argument("--swap_layers", default = "4,11,16,22,29")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    swap_layers = [int(x) for x in args.swap_layers.split(",")]

    # Load prompts
    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = []
    for item in items:
        behaviors.append(_extract_behavior(item))

    # Load quantized model and capture activations
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    quant_acts_by_layer = {}
    for layer_idx in swap_layers:
        quant_acts_by_layer[layer_idx] = capture_quant_activations(q_model, q_tok, items, layer_idx)
    quant_responses = baseline_generate(q_model, q_tok, items, max_new_tokens = args.max_new_tokens)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load fp32 model and generate responses
    fp32_model, fp32_tok = load_fp32(args.model)
    fp32_responses = baseline_generate(
        model = fp32_model,
        tok = fp32_tok,
        prompts = items,
        max_new_tokens = args.max_new_tokens,
    )

    # Generate responses with layer swap
    all_responses = {"fp32_baseline": fp32_responses, "quant_baseline": quant_responses}
    for layer_idx in swap_layers:
        all_responses[f"swap_q_into_fp32_L{layer_idx}"] = generate_with_layer_swap(
            fp32_model = fp32_model,
            fp32_tok = fp32_tok,
            prompts = items,
            layer_idx = layer_idx,
            quant_acts = quant_acts_by_layer[layer_idx],
            max_new_tokens = args.max_new_tokens,
        )
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Classify responses
    keys = list(all_responses.keys())
    flat_b, flat_r = [], []
    for k in keys:
        flat_b.extend(behaviors)
        flat_r.extend(all_responses[k])
    flags = classify_batch(flat_b, flat_r, device = device)
    n = args.n_eval

    # Calculate the ASR for each key
    asr = {k: 100.0 * sum(flags[i * n : (i + 1) * n]) / n for i, k in enumerate(keys)}

    # Save results
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        extra = {"n_eval": args.n_eval, "swap_layers": swap_layers, "seed": SEED},
    )
    data = {
        "fp32_baseline_asr": asr["fp32_baseline"],
        "quant_baseline_asr": asr["quant_baseline"],
        "swap_asr": {f"L{L}": asr[f"swap_q_into_fp32_L{L}"] for L in swap_layers},
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
