"""
Quantize Qwen2.5-7B-Instruct to AQLM 2-bit.

AQLM uses learned additive codebooks — this is an iterative optimization
process that takes 8-24 hours on a single H100 for a 7B model.

The quantized model is saved to the HF cache for reuse.
"""
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_HOME

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SAVE_PATH = os.path.join(HF_HOME, "aqlm_cache", "qwen7b-aqlm-2bit")

# AQLM quantization config
NBITS_PER_CODEBOOK = 16  # 2^16 = 65536 codebook entries
NUM_CODEBOOKS = 1        # 1x16 config = 2-bit effective
IN_GROUP_SIZE = 8        # group size for codebook
FINETUNE_LR = 1e-4
FINETUNE_STEPS = 100


def get_calibration_data(tokenizer, nsamples=128, seqlen=2048):
    """Load calibration data from wikitext2."""
    print("  Loading calibration data (wikitext2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    calibration_data = []
    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        if end > len(tokens):
            break
        calibration_data.append(tokens[start:end].unsqueeze(0))

    print(f"  Prepared {len(calibration_data)} calibration sequences of length {seqlen}")
    return calibration_data


def main():
    print("=" * 60)
    print("  AQLM 2-bit Quantization: Qwen2.5-7B-Instruct")
    print("=" * 60)

    if os.path.isdir(SAVE_PATH) and os.path.exists(os.path.join(SAVE_PATH, "config.json")):
        print(f"  ✓ Already quantized at: {SAVE_PATH}")
        return

    print(f"\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_HOME,
                                               trust_remote_code=True)

    print(f"  Loading model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )

    calibration_data = get_calibration_data(tokenizer)

    print(f"\n  Starting AQLM quantization...")
    print(f"  Config: {NUM_CODEBOOKS}x{NBITS_PER_CODEBOOK} = ~2-bit")
    print(f"  Group size: {IN_GROUP_SIZE}")
    print(f"  This will take several hours...")

    try:
        from aqlm import quantize

        quantized_model = quantize(
            model,
            calibration_data=calibration_data,
            nbits_per_codebook=NBITS_PER_CODEBOOK,
            num_codebooks=NUM_CODEBOOKS,
            in_group_size=IN_GROUP_SIZE,
            finetune_lr=FINETUNE_LR,
            finetune_max_steps=FINETUNE_STEPS,
            relative_mse_tolerance=0.01,
        )

        print(f"\n  ✓ Quantization complete!")
        print(f"  Saving to: {SAVE_PATH}")

        os.makedirs(SAVE_PATH, exist_ok=True)
        quantized_model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  ✓ Saved!")

    except ImportError as e:
        print(f"  ✗ AQLM quantization requires 'aqlm' package: {e}")
        print(f"  Run: pip install aqlm[gpu]")
    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
