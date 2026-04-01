# Phase 2 experiment: does safety loss come from read matrices or write matrices?
#
# The idea is simple - load the NF4 model, then swap in FP32 weights for one group
# at a time and see which swap actually brings safety back.
#
# Read matrices (q, k, v, gate_proj, up_proj): column j reads from residual dim j
# Write matrices (o_proj, down_proj): row i writes to residual dim i
#
# If swapping read matrices fixes most of the damage, that confirms the
# Read-Side Collapse hypothesis - quantization breaks the detectors, not the alarms.
#
# Run:
#   python phase2_solution/experiments/read_write_restoration.py --model qwen1.5b
#   python phase2_solution/experiments/read_write_restoration.py --model qwen1.5b --n_eval 50

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODELS, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS

WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]
READ_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.gate_proj", "mlp.up_proj"]


def load_fp32_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def load_nf4_model(model_name):
    # float32 compute dtype because float16 matmul fails on CPU
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


def _get_submodule(layer, proj_path):
    # walk dotted path like "self_attn.o_proj" to get the actual module
    parts = proj_path.split(".")
    m = layer
    for p in parts:
        m = getattr(m, p)
    return m


def _set_submodule(layer, proj_path, new_module):
    parts = proj_path.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def extract_fp32_weights(fp32_model, proj_paths):
    # grab a plain float32 copy of each weight before we delete the fp32 model
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


def restore_projections(nf4_model, fp32_weights, proj_paths):
    # we replace the whole module (not just the weight tensor) because the NF4
    # model has Linear4bit objects - assigning a plain tensor to .weight breaks them
    #
    # the key thing: cast the restored weights to match the model's activation dtype
    # (bfloat16 for Qwen2.5) so there's no dtype mismatch at the matmul
    act_dtype = nf4_model.model.norm.weight.dtype

    n_layers = nf4_model.config.num_hidden_layers
    for l in range(n_layers):
        layer = nf4_model.model.layers[l]
        for proj in proj_paths:
            w = fp32_weights[l][proj].to(act_dtype)
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


def is_refusal(text):
    t = text.strip()
    for prefix in REFUSAL_PREFIXES:
        if t.lower().startswith(prefix.lower()):
            return True
    return False


def measure_asr(model, tokenizer, prompts):
    comply = 0
    for prompt in tqdm(prompts, desc="    ASR", leave=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if not is_refusal(response):
            comply += 1
    return comply / len(prompts) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen1.5b", choices=list(MODELS.keys()))
    parser.add_argument("--n_eval", type=int, default=None,
                        help="number of eval prompts to use (default: all)")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, "eval_harmful.json"), encoding="utf-8") as f:
        eval_prompts = json.load(f)
    if args.n_eval:
        eval_prompts = eval_prompts[:args.n_eval]

    print(f"\nRead-Side vs Write-Side Restoration: {args.model}")
    print(f"Eval prompts: {len(eval_prompts)}\n")

    results = {}

    # condition 1: fp32 baseline - also use this to extract the weights we need
    print("[1/5] FP32 baseline...")
    fp32_model, tokenizer = load_fp32_model(model_name)
    n_layers = fp32_model.config.num_hidden_layers

    print("      extracting write weights (o_proj, down_proj)...")
    fp32_write_weights = extract_fp32_weights(fp32_model, WRITE_PROJS)
    print("      extracting read weights (q/k/v/gate/up_proj)...")
    fp32_read_weights = extract_fp32_weights(fp32_model, READ_PROJS)

    asr = measure_asr(fp32_model, tokenizer, eval_prompts)
    results["fp32"] = {"asr": asr, "description": "FP32 baseline"}
    print(f"      ASR = {asr:.1f}%")
    del fp32_model

    # condition 2: nf4 with no repair - this is the damage we want to understand
    print("\n[2/5] NF4 baseline (no repair)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["nf4"] = {"asr": asr, "description": "NF4 baseline"}
    print(f"      ASR = {asr:.1f}%")
    del nf4_model

    # condition 3: restore only the write side
    print("\n[3/5] NF4 + write matrices restored (o_proj, down_proj)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_write_weights, WRITE_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["write_restored"] = {
        "asr": asr,
        "description": "NF4 + write matrices replaced with FP32",
        "n_layers": n_layers,
        "projs_restored": WRITE_PROJS,
    }
    print(f"      ASR = {asr:.1f}%")
    del nf4_model

    # condition 4: restore only the read side - this is the key test
    print("\n[4/5] NF4 + read matrices restored (q/k/v/gate_proj/up_proj)...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_read_weights, READ_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["read_restored"] = {
        "asr": asr,
        "description": "NF4 + read matrices replaced with FP32",
        "n_layers": n_layers,
        "projs_restored": READ_PROJS,
    }
    print(f"      ASR = {asr:.1f}%")
    del nf4_model

    # condition 5: restore everything - should get close to fp32 baseline
    print("\n[5/5] NF4 + all matrices restored...")
    nf4_model, tokenizer = load_nf4_model(model_name)
    restore_projections(nf4_model, fp32_write_weights, WRITE_PROJS)
    restore_projections(nf4_model, fp32_read_weights, READ_PROJS)
    asr = measure_asr(nf4_model, tokenizer, eval_prompts)
    results["both_restored"] = {
        "asr": asr,
        "description": "NF4 + all matrices replaced (upper bound)",
        "n_layers": n_layers,
        "projs_restored": WRITE_PROJS + READ_PROJS,
    }
    print(f"      ASR = {asr:.1f}%")
    del nf4_model

    # print a simple summary table
    print(f"\nResults for {args.model}:")
    order = ["fp32", "nf4", "write_restored", "read_restored", "both_restored"]
    for k in order:
        if k not in results:
            continue
        r = results[k]
        notes = {
            "fp32": "target",
            "nf4": "problem",
            "write_restored": "write alarms fixed",
            "read_restored": "read sensors fixed",
            "both_restored": "upper bound",
        }.get(k, "")
        print(f"  {k:<20}  ASR={r['asr']:.1f}%  ({notes})")

    nf4_asr = results.get("nf4", {}).get("asr", 0)
    fp32_asr = results.get("fp32", {}).get("asr", 100)
    write_asr = results.get("write_restored", {}).get("asr", nf4_asr)
    read_asr = results.get("read_restored", {}).get("asr", nf4_asr)
    both_asr = results.get("both_restored", {}).get("asr", nf4_asr)
    total_damage = nf4_asr - fp32_asr

    print(f"\nTotal damage from quantization: {total_damage:+.1f}pp (NF4 vs FP32)")
    if total_damage > 0:
        write_rec = nf4_asr - write_asr
        read_rec = nf4_asr - read_asr
        both_rec = nf4_asr - both_asr
        print(f"Write restoration recovered:    {write_rec:+.1f}pp  "
              f"({100*write_rec/total_damage:.0f}% of damage)")
        print(f"Read  restoration recovered:    {read_rec:+.1f}pp  "
              f"({100*read_rec/total_damage:.0f}% of damage)")
        print(f"Both  restoration recovered:    {both_rec:+.1f}pp  "
              f"({100*both_rec/total_damage:.0f}% of damage)")

    if (nf4_asr - read_asr) > (nf4_asr - write_asr):
        print(f"\nRead-Side Collapse confirmed: read restoration ({read_asr:.1f}%) "
              f"is stronger than write ({write_asr:.1f}%)")
    elif (nf4_asr - write_asr) > (nf4_asr - read_asr):
        print(f"\nWrite-side dominates: write restoration ({write_asr:.1f}%) "
              f"is stronger than read ({read_asr:.1f}%)")
    else:
        print("\nBoth sides contribute roughly equally.")

    save_path = os.path.join(output_dir, "read_write_restoration.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
