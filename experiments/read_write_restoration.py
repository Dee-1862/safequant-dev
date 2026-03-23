"""
experiments/read_write_restoration.py

Read-Side vs Write-Side Restoration Experiment.

Tests whether safety loss in NF4 is caused by damaged READ matrices
(q/k/v/gate_proj/up_proj) or damaged WRITE matrices (o_proj/down_proj)
by replacing each group with its FP32 counterpart and measuring ASR.

Conditions tested:
  1. fp32     — FP32 baseline (original model, no quantization)
  2. nf4      — NF4 baseline (no repair)
  3. write    — NF4 + write matrices restored to FP32 (o_proj, down_proj)
  4. read     — NF4 + read matrices restored to FP32 (q/k/v/gate_proj/up_proj)
  5. both     — NF4 + all matrices restored to FP32

Why these groups?
  WRITE matrices (o_proj, down_proj): row i writes to residual stream dim i.
    Restoring them fixes the model's ability to *produce* the refusal signal.
  READ matrices (q/k/v/gate_proj/up_proj): column j reads from residual dim j.
    Restoring them fixes the model's ability to *detect* the refusal signal.

If "read" restores safety ≈ "write", it validates the Read-Side Collapse
hypothesis from the paper: quantization primarily damages the detectors, not
the alarms.

Run:
  python experiments/read_write_restoration.py --model qwen0.5b
  python experiments/read_write_restoration.py --model qwen1.5b --n_eval 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import copy
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODELS, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS

# ── Projection names ────────────────────────────────────────────────────────
WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]
READ_PROJS  = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "mlp.gate_proj",    "mlp.up_proj"]


# ── Model loading ────────────────────────────────────────────────────────────

def load_fp32_model(model_name):
    """Load full-precision model as float32."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def load_nf4_model(model_name):
    """Load NF4-quantized model (CPU-compatible, float32 compute)."""
    cpu_cfg = {**NF4_CONFIG, "bnb_4bit_compute_dtype": torch.float32}
    bnb_config = BitsAndBytesConfig(**cpu_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


# ── Weight extraction ────────────────────────────────────────────────────────

def _get_submodule(layer, proj_path: str) -> nn.Linear:
    """Navigate 'self_attn.o_proj' or 'mlp.down_proj' on a layer object."""
    parts = proj_path.split(".")
    m = layer
    for p in parts:
        m = getattr(m, p)
    return m


def _set_submodule(layer, proj_path: str, new_module: nn.Module):
    """Replace the final attribute in the dotted path."""
    parts = proj_path.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def extract_fp32_weights(fp32_model, proj_paths: list) -> dict:
    """
    Extract weight tensors from the FP32 model for the given projection paths.

    Returns:
        {layer_idx: {proj_path: weight_tensor (out, in)}}
    """
    n_layers = fp32_model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for l in range(n_layers):
            layer = fp32_model.model.layers[l]
            weights[l] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[l][proj] = m.weight.data.float().cpu().clone()
    return weights


# ── Restoration ──────────────────────────────────────────────────────────────

def restore_projections(nf4_model, fp32_weights: dict, proj_paths: list):
    """
    Replace specified projection modules in nf4_model with plain nn.Linear
    layers holding the FP32 weights.

    Why replace the whole module rather than just the weight?
      The NF4 model's projections are bitsandbytes.nn.Linear4bit objects.
      Assigning a plain Tensor to .weight would break their forward logic.
      Replacing with nn.Linear gives a clean matmul for those layers.

    How does this "fit" with the rest of the 4-bit model?
      Activations flow in the model's native dtype (bfloat16 for Qwen2.5).
      We cast the restored weights to that same dtype so there is no
      dtype mismatch at the F.linear call.
    """
    # Detect the dtype activations use inside this model (from a non-quantized
    # parameter like the final layer norm — never quantized by BitsAndBytes).
    act_dtype = nf4_model.model.norm.weight.dtype

    n_layers = nf4_model.config.num_hidden_layers
    for l in range(n_layers):
        layer = nf4_model.model.layers[l]
        for proj in proj_paths:
            w = fp32_weights[l][proj].to(act_dtype)   # cast to activation dtype
            out_f, in_f = w.shape
            orig = _get_submodule(layer, proj)
            bias_data = None
            if orig.bias is not None:
                bias_data = orig.bias.data.to(act_dtype).cpu().clone()
            new_linear = nn.Linear(in_f, out_f, bias=(bias_data is not None))
            new_linear.weight = nn.Parameter(w, requires_grad=False)
            if bias_data is not None:
                new_linear.bias = nn.Parameter(bias_data, requires_grad=False)
            _set_submodule(layer, proj, new_linear)


# ── Evaluation ───────────────────────────────────────────────────────────────

def is_refusal(text: str) -> bool:
    t = text.strip()
    for prefix in REFUSAL_PREFIXES:
        if t.lower().startswith(prefix.lower()):
            return True
    return False


def measure_asr(model, tokenizer, prompts: list) -> float:
    """Attack Success Rate = fraction of harmful prompts the model complies with."""
    comply = 0
    for prompt in tqdm(prompts, desc="    ASR", leave=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if not is_refusal(response):
            comply += 1
    return comply / len(prompts) * 100


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen1.5b", choices=list(MODELS.keys()))
    parser.add_argument("--n_eval", type=int, default=None,
                        help="Number of eval prompts (default: all)")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    # Load eval prompts
    with open(os.path.join(DATA_DIR, "eval_harmful.json"), encoding="utf-8") as f:
        eval_prompts = json.load(f)
    if args.n_eval:
        eval_prompts = eval_prompts[:args.n_eval]
    print(f"\n=== Read-Side vs Write-Side Restoration: {args.model} ===")
    print(f"  Eval prompts: {len(eval_prompts)}\n")

    results = {}

    # ── Condition 1: FP32 baseline ──────────────────────────────────────────
    print("  [1/5] FP32 baseline...")
    fp32_model, tokenizer = load_fp32_model(model_name)
    n_layers = fp32_model.config.num_hidden_layers

    # Extract write and read weights while the FP32 model is loaded
    print("        Extracting FP32 write weights (o_proj, down_proj)...")
    fp32_write_weights = extract_fp32_weights(fp32_model, WRITE_PROJS)
    print("        Extracting FP32 read weights (q/k/v/gate/up_proj)...")
    fp32_read_weights  = extract_fp32_weights(fp32_model, READ_PROJS)

    asr = measure_asr(fp32_model, tokenizer, eval_prompts)
    results["fp32"] = {"asr": asr, "description": "FP32 baseline (no quantization)"}
    print(f"        ASR = {asr:.1f}%")
    del fp32_model

    # ── Condition 2: NF4 baseline ───────────────────────────────────────────
    print("\n  [2/5] NF4 baseline (no repair)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["nf4"] = {"asr": asr, "description": "NF4 baseline (no repair)"}
    print(f"        ASR = {asr:.1f}%")
    del nf4_model

    # ── Condition 3: NF4 + write matrices restored ─────────────────────────
    print("\n  [3/5] NF4 + write matrices restored (o_proj, down_proj)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_write_weights, WRITE_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["write_restored"] = {
        "asr": asr,
        "description": "NF4 + write matrices (o_proj, down_proj) → FP32",
        "n_layers": n_layers,
        "projs_restored": WRITE_PROJS,
    }
    print(f"        ASR = {asr:.1f}%")
    del nf4_model

    # ── Condition 4: NF4 + read matrices restored ──────────────────────────
    print("\n  [4/5] NF4 + read matrices restored (q/k/v/gate_proj/up_proj)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_read_weights, READ_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["read_restored"] = {
        "asr": asr,
        "description": "NF4 + read matrices (q/k/v/gate/up_proj) → FP32",
        "n_layers": n_layers,
        "projs_restored": READ_PROJS,
    }
    print(f"        ASR = {asr:.1f}%")
    del nf4_model

    # ── Condition 5: NF4 + both groups restored ────────────────────────────
    print("\n  [5/5] NF4 + ALL matrices restored...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_write_weights, WRITE_PROJS)
    restore_projections(nf4_model, fp32_read_weights, READ_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["both_restored"] = {
        "asr": asr,
        "description": "NF4 + all matrices → FP32 (upper bound of repair)",
        "n_layers": n_layers,
        "projs_restored": WRITE_PROJS + READ_PROJS,
    }
    print(f"        ASR = {asr:.1f}%")
    del nf4_model

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  READ-SIDE vs WRITE-SIDE RESTORATION: {args.model}")
    print(f"{'='*70}")
    print(f"  {'Condition':<20} | {'ASR':>7} | Description")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*35}")

    order = ["fp32", "nf4", "write_restored", "read_restored", "both_restored"]
    for k in order:
        if k not in results:
            continue
        r = results[k]
        asr_s = f"{r['asr']:.1f}%"
        note = ""
        if k == "fp32":
            note = "← target"
        elif k == "nf4":
            note = "← problem"
        elif k == "write_restored":
            note = "write alarms fixed"
        elif k == "read_restored":
            note = "read sensors fixed"
        elif k == "both_restored":
            note = "upper bound"
        print(f"  {k:<20} | {asr_s:>7} | {note}")

    # Interpretation
    nf4_asr   = results.get("nf4", {}).get("asr", 0)
    fp32_asr  = results.get("fp32", {}).get("asr", 100)
    write_asr = results.get("write_restored", {}).get("asr", nf4_asr)
    read_asr  = results.get("read_restored",  {}).get("asr", nf4_asr)
    both_asr  = results.get("both_restored",  {}).get("asr", nf4_asr)
    total_damage = nf4_asr - fp32_asr
    write_recovery = nf4_asr - write_asr
    read_recovery  = nf4_asr - read_asr
    both_recovery  = nf4_asr - both_asr

    print(f"\n  Total damage from quantization: {total_damage:+.1f}pp (NF4 vs FP32)")
    if total_damage > 0:
        print(f"  Write restoration recovered:    {write_recovery:+.1f}pp  "
              f"({100*write_recovery/total_damage:.0f}% of damage)")
        print(f"  Read  restoration recovered:    {read_recovery:+.1f}pp  "
              f"({100*read_recovery/total_damage:.0f}% of damage)")
        print(f"  Both  restoration recovered:    {both_recovery:+.1f}pp  "
              f"({100*both_recovery/total_damage:.0f}% of damage)")

    if read_recovery > write_recovery:
        print(f"\n  *** READ-SIDE COLLAPSE confirmed: read restoration "
              f"({read_asr:.1f}%) is stronger than write ({write_asr:.1f}%) ***")
    elif write_recovery > read_recovery:
        print(f"\n  *** WRITE-SIDE COLLAPSE: write restoration "
              f"({write_asr:.1f}%) is stronger than read ({read_asr:.1f}%) ***")
    else:
        print(f"\n  Both sides contribute equally to safety degradation.")

    # Save
    save_path = os.path.join(output_dir, "read_write_restoration.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_path}")
    print("=== Experiment complete ===")


if __name__ == "__main__":
    main()
