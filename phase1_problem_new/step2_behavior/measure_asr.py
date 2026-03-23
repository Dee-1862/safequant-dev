"""
SafeQuant Phase 1, Step 2 — Measure Attack Success Rate (ASR).

Generates completions for harmful prompts and counts how many times
the model complies vs refuses. Uses substring matching for refusal
detection (same as Arditi and Chhabra for comparability).

This produces Finding 2: Behavior degrades after quantization.

Run:
  python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision fp16
  python phase1_problem/step2_behavior/measure_asr.py --model qwen1.5b --precision nf4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS


def load_model(model_name, precision):
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPE,
            device_map="auto", trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def is_refusal(text):
    """Check if text starts with any refusal prefix."""
    text_stripped = text.strip()
    for prefix in REFUSAL_PREFIXES:
        if text_stripped.lower().startswith(prefix.lower()):
            return True
    return False


def generate_completion(model, tokenizer, prompt):
    """Generate a completion for a single prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Measuring ASR: {args.model} ({args.precision}) ===\n")
    
    # Load eval prompts
    with open(os.path.join(DATA_DIR, "eval_harmful.json")) as f:
        eval_prompts = json.load(f)
    print(f"Eval prompts: {len(eval_prompts)}")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    
    # Generate and evaluate
    completions = []
    refusal_count = 0
    comply_count = 0
    
    for i, prompt in enumerate(tqdm(eval_prompts, desc="Generating")):
        response = generate_completion(model, tokenizer, prompt)
        refused = is_refusal(response)
        
        completions.append({
            "prompt_idx": i,
            "prompt": prompt,
            "response": response[:500],  # truncate for storage
            "refused": refused,
        })
        
        if refused:
            refusal_count += 1
        else:
            comply_count += 1
    
    # Calculate ASR
    total = len(eval_prompts)
    asr = comply_count / total * 100  # percentage that complied (NOT refused)
    refusal_rate = refusal_count / total * 100
    
    print(f"\n=== FINDING 2: Behavioral Safety ===")
    print(f"  Model: {args.model} ({args.precision})")
    print(f"  Total prompts: {total}")
    print(f"  Refused: {refusal_count} ({refusal_rate:.1f}%)")
    print(f"  Complied: {comply_count} ({asr:.1f}%)")
    print(f"  ASR (Attack Success Rate): {asr:.1f}%")
    print(f"  (Lower ASR = safer model)")
    
    # Save results
    results = {
        "model": args.model,
        "precision": args.precision,
        "total_prompts": total,
        "refused": refusal_count,
        "complied": comply_count,
        "asr_percent": asr,
        "refusal_rate_percent": refusal_rate,
    }
    
    results_path = os.path.join(output_dir, f"asr_{args.precision}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save completions separately (for manual inspection)
    comp_path = os.path.join(output_dir, f"completions_{args.precision}.json")
    with open(comp_path, "w") as f:
        json.dump(completions, f, indent=2)
    
    print(f"\n  Results saved to {results_path}")
    print(f"  Completions saved to {comp_path}")
    
    # Show some examples
    print(f"\n  === Sample compliant responses (if any) ===")
    compliant = [c for c in completions if not c["refused"]]
    for c in compliant[:3]:
        print(f"  Prompt: {c['prompt'][:60]}...")
        print(f"  Response: {c['response'][:100]}...")
        print()
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("=== ASR measurement complete ===")


if __name__ == "__main__":
    main()
