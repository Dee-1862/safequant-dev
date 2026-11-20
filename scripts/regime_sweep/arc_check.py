# ARC-Challenge accuracy of one quantized cell, next to the paper value and chance.
#
#   python scripts/regime_sweep/arc_check.py --model llama7b --quantizer gptq2 --tag rebuilt_gptqmodel
#
# Same scorer and items as the paper: core.restoration.measure_arc (summed choice
# log-probs) on the first --n_reasoning items of load_arc_challenge (paper: 100, from
# restoration_sweep_<m>_<q>.json). Writes outputs/regime_sweep/arc_check_<m>_<q>_<tag>.json
# (new name; refuses to overwrite) and prints accuracy with a Wilson 95% interval,
# the paper's quantized and FP32 ARC, and chance (25%, 4-way).
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.datasets import load_arc_challenge  # noqa: E402
from core.manifest import RunConfig, assert_new_result, result_path, save_result  # noqa: E402
from core.model_loader import load_quantized  # noqa: E402
from core.restoration import measure_arc  # noqa: E402
from core.seeds import DEFAULT_SEED, set_global_seed  # noqa: E402

_SCRIPT = Path(__file__)
CHANCE = 25.0


def wilson(p_pct, n, z = 1.96):
    """
    Wilson score interval for an accuracy in percent.

    Inputs:
        - p_pct (float): Accuracy (%)
        - n (int): Items
        - z (float): Normal quantile

    Outputs:
        - low (float): Lower bound (%)
        - high (float): Upper bound (%)
    """
    p = p_pct / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def paper_arc(model, quant):
    """
    Paper ARC values for one cell from the untagged restoration_sweep result.

    Inputs:
        - model (str): Model alias
        - quant (str): Quantizer key

    Outputs:
        - quant_arc (float or None): Quantized ARC (%)
        - fp32_arc (float or None): FP32 ARC (%)
        - path (str): File read
    """
    p = result_path(Path("experiments/regime_sweep/restoration_sweep.py"),
                    RunConfig(model = model, quantizer = quant))
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None, None, str(p)
    a = d.get("arc", {})
    return a.get(quant, a.get("quant")), a.get("fp32"), str(p)


def main():
    ap = argparse.ArgumentParser(description = "ARC-Challenge accuracy for one quantized cell.")
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required = True)
    ap.add_argument("--n_reasoning", type = int, default = 100)
    ap.add_argument("--tag", required = True)
    args = ap.parse_args()

    seed = set_global_seed(DEFAULT_SEED)
    cfg = RunConfig(model = args.model, quantizer = args.quantizer, variant = args.tag,
                    extra = {"n_reasoning": args.n_reasoning})
    assert_new_result(_SCRIPT, cfg)

    items = load_arc_challenge(n_samples = args.n_reasoning)
    model, tok = load_quantized(args.model, quantizer = args.quantizer)
    acc = measure_arc(model, tok, items)
    lo, hi = wilson(acc, len(items))
    q_paper, fp_paper, src = paper_arc(args.model, args.quantizer)

    path = save_result(_SCRIPT, cfg, {
        "model": args.model, "quantizer": args.quantizer, "n_reasoning": len(items),
        "arc": acc, "arc_wilson95": [lo, hi], "chance": CHANCE,
        "paper_arc_quant": q_paper, "paper_arc_fp32": fp_paper, "paper_source": src,
        "seed": seed,
    })
    print(f"\n  {args.model} {args.quantizer}  ARC = {acc:.1f}%  (95% CI {lo:.1f}-{hi:.1f}, n={len(items)})")
    print(f"  paper: quant {q_paper}  fp32 {fp_paper}   chance {CHANCE:.0f}%")
    near = lo <= CHANCE <= hi
    print(f"  chance (25%) inside the 95% CI: {'yes' if near else 'no'}")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
