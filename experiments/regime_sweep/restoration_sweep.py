"""Table 2: 14-cell FP/quant/+W/+R/+Both restoration matrix."""
import argparse
import gc
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch, generate_only
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.restoration import Restorer, measure_arc
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_SCRIPT = Path(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantizer", default="aqlm2")
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--n_reasoning", type=int, default=50)
    args = parser.parse_args()
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(item) for item in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    all_responses, all_arc = {}, {}

    fp32_model, tokenizer = load_fp32(args.model)
    restorer = Restorer(fp32_model)
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    for cond, restore_fn in [
        ("write_restored", "restore_write"),
        ("read_restored", "restore_read"),
        ("both_restored", "restore_both"),
    ]:
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        getattr(restorer, restore_fn)(q_model)
        all_responses[cond] = generate_only(q_model, tokenizer, items, desc=cond)
        all_arc[cond] = measure_arc(q_model, tokenizer, arc_items)
        del q_model
        gc.collect()
        torch.cuda.empty_cache()

    conditions = list(all_responses.keys())
    flat_behaviors, flat_responses = [], []
    for cond in conditions:
        flat_behaviors.extend(behaviors)
        flat_responses.extend(all_responses[cond])
    flat_flags = classify_batch(flat_behaviors, flat_responses, device=device)
    n = args.n_eval
    asr_results = {}
    for i, cond in enumerate(conditions):
        flags = flat_flags[i * n : (i + 1) * n]
        asr_results[cond] = 100.0 * sum(flags) / n

    fp32_asr = asr_results["fp32"]
    q_asr = asr_results[qtz]
    damage = q_asr - fp32_asr
    both_asr = asr_results["both_restored"]
    swap_tol = max(8.0, 2 * math.sqrt(2 * 0.25 / max(args.n_eval, 1)) * 100)
    swap_delta = both_asr - fp32_asr
    arc_floor_ok = (
        min(all_arc["write_restored"], all_arc["read_restored"], all_arc["both_restored"])
        >= all_arc[qtz] - 7.0
    )
    swap_valid = abs(swap_delta) <= swap_tol and arc_floor_ok

    cfg = RunConfig(
        model=args.model,
        quantizer=qtz,
        extra={"n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED},
    )
    data = {
        "fp32_asr": fp32_asr,
        "quantized_asr": q_asr,
        "damage_pp": damage,
        "write_restored_asr": asr_results["write_restored"],
        "read_restored_asr": asr_results["read_restored"],
        "both_restored_asr": both_asr,
        "write_recovery_pp": q_asr - asr_results["write_restored"],
        "read_recovery_pp": q_asr - asr_results["read_restored"],
        "swap_valid": swap_valid,
        "swap_delta_pp": swap_delta,
        "swap_tol_pp": swap_tol,
        "arc": all_arc,
        "asr_per_condition": asr_results,
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
