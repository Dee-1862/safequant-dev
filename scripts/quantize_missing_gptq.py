"""
On-the-fly GPTQ quantization for models that don't have pre-quantized checkpoints.
Run on a GPU node via quantize_missing_gptq.sbatch.

Quantizes and caches:
  - Mistral-7B-v0.2: 3-bit, 2-bit
  - Qwen2.5-7B:      3-bit, 2-bit
"""
import os
import sys
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

CACHE_BASE = os.path.join(HF_HOME, "gptq_cache")

# (model_hf_id, alias, bits)
JOBS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral7b-v02", 3),
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral7b-v02", 2),
    ("Qwen/Qwen2.5-7B-Instruct",          "qwen7b",        3),
    ("Qwen/Qwen2.5-7B-Instruct",          "qwen7b",        2),
]


def quantize_and_cache(model_id, alias, bits):
    cache_path = os.path.join(CACHE_BASE, f"{alias}-gptq-{bits}bit")

    # Skip if already cached
    if os.path.isdir(cache_path) and os.path.exists(os.path.join(cache_path, "config.json")):
        print(f"  ✓ Already cached: {cache_path}")
        return

    print(f"\n  Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_HOME, trust_remote_code=True
    )

    gptq_config = GPTQConfig(
        bits=bits,
        dataset="wikitext2",
        tokenizer=tokenizer,
        group_size=128,
        use_exllama=False,
    )

    print(f"  Quantizing {model_id} to {bits}-bit GPTQ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=gptq_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )
    model.eval()
    print(f"  ✓ Quantized to {bits}-bit")

    # Save to cache
    os.makedirs(cache_path, exist_ok=True)
    model.save_pretrained(cache_path)
    tokenizer.save_pretrained(cache_path)
    print(f"  ✓ Saved to: {cache_path}")

    # Free memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("  GPTQ On-the-fly Quantization")
    print("=" * 60)

    for model_id, alias, bits in JOBS:
        print(f"\n{'='*60}")
        print(f"  {alias} / gptq{bits}")
        print(f"{'='*60}")
        try:
            quantize_and_cache(model_id, alias, bits)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next job
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  All quantization jobs complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
