import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.model_loader import load_fp32, load_quantized  # noqa: E402
from core.evaluation import generate_only, classify_batch, _extract_behavior  # noqa: E402
from core.datasets import load_wildjailbreak_prompts  # noqa: E402
from core.manifest import RunConfig, save_result  # noqa: E402

_SCRIPT = Path(__file__)

ADVBENCH_SPLIT = (
    Path(__file__).resolve().parents[2]
    / "refusal_direction/dataset/splits/harmful_train.json"
)


def _advbench(n_prompts):
    """
    Get the items from the AdvBench split.
    Inputs:
        n_prompts (int): The number of prompts to get.

    Outputs:
        items (list[dict]): The items from the AdvBench split.
    """
    # Open the AdvBench split
    with open(ADVBENCH_SPLIT) as f:
        items = json.load(f)
    prompts = [it["instruction"] for it in items if it.get("instruction")]

    # Get the prompts
    prompts = prompts[:n_prompts] if n_prompts else prompts
    
    # Return the items
    return [{"prompt": p, "vanilla": p} for p in prompts]


def get_items(source, n_prompts, wj_config):
    """
    Get the items from the source.
    Inputs:
        source (str): The source of the items.
        n_prompts (int): The number of prompts to get.
        wj_config (str): The configuration of the wildjailbreak prompts.

    Outputs:
    """

    # Get the items from the source
    if source == "wildjailbreak":
        return load_wildjailbreak_prompts(n_samples=n_prompts, config=wj_config)

    # Get the items from the AdvBench split
    if source == "advbench":
        return _advbench(n_prompts)
    
    # Raise an error if the source is unknown
    raise ValueError(f"unknown prompt_source: {source}")


def _send_prompt(item):
    """
    Send the prompt to the model.
    Inputs:
        item (dict): The item to send the prompt to.

    Outputs:
        prompt (str): The prompt to send to the model.
    """
    # Return the prompt
    return (item.get("prompt") or item.get("goal") or item.get("instruction") or "")


def main():

    # Parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--prompt_source", default="wildjailbreak",
                    choices=["wildjailbreak", "advbench"])
    ap.add_argument("--wj_config", default="eval", choices=["train", "eval"])
    ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output", default=None,
                    help="override output path (default: outputs/trained_repair/qresafe_pairs_<model>_<quant>[_<tag>].json)")
    args = ap.parse_args()

    # Get the pair tag
    pair_tag = args.wj_config if args.prompt_source == "wildjailbreak" and args.wj_config != "train" else None

    # Get the items
    items = get_items(args.prompt_source, args.n_prompts, args.wj_config)

    # Send the prompts
    sends = [_send_prompt(it) for it in items]

    # Extract the behaviors
    behaviors = [_extract_behavior(it) for it in items]
    n = len(items)
    print(f"[data] {n} prompts from {args.prompt_source} (config={args.wj_config})")

    # ---- FP completions (yw candidates) ----
    fp_model, tok = load_fp32(args.model)
    fp_resp = generate_only(fp_model, tok, items, max_new_tokens=args.max_new_tokens,
                            desc="FP")

    # Delete the FP model and collect garbage
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- quantized completions (yl candidates) ----
    q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
    q_resp = generate_only(q_model, tok, items, max_new_tokens=args.max_new_tokens,
                           desc=args.quantizer)

    # Delete the quantized model and collect garbage
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- judge both with HarmBench (one session) ----
    # Classify the behaviors
    flags = classify_batch(behaviors + behaviors, fp_resp + q_resp, device="cuda")

    # Get the harmful flags
    fp_harmful = flags[:n]

    # Get the harmful flags
    q_harmful = flags[n:]

    # Calculate the ASR
    fp_asr = 100.0 * sum(fp_harmful) / n
    q_asr = 100.0 * sum(q_harmful) / n
    print(f"[asr] FP={fp_asr:.1f}%  {args.quantizer}={q_asr:.1f}%  "
          f"({args.prompt_source}, HarmBench judge)")

    # ---- displacement set: FP refused, quantized complied ----

    # Create the pairs
    pairs = []
    for sp, fw, fh, qr, qh in zip(sends, fp_resp, fp_harmful, q_resp, q_harmful):
        if (not fh) and qh:  # FP safe, quantized harmful == displacement
            pairs.append({"instruction": sp, "yw": fw.strip(), "yl": qr.strip()})

    # Print the number of pairs
    print(f"[pairs] {len(pairs)} displacement pairs (FP refused & {args.quantizer} complied)")
    if len(pairs) < 20:
        print(f"  WARNING: only {len(pairs)} displacement pairs -- saliency may be noisy. "
              f"Consider widening n_prompts or relaxing the filter.")

    # Create the run config
    cfg = RunConfig(
        model=args.model,
        quantizer=args.quantizer,
        variant=pair_tag,
        extra={
            "prompt_source": args.prompt_source,
            "wj_config": args.wj_config,
            "n_prompts": n,
            "fp_asr": fp_asr,
            "q_asr": q_asr,
            "max_new_tokens": args.max_new_tokens,
            "filter": "fp_refused AND quant_complied",
        },
    )

    # Create the data
    data = {"n_pairs": len(pairs), "pairs": pairs}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        out_path.write_text(_json.dumps({"meta": cfg.to_dict(), **data}, indent=2))
    else:
        out_path = save_result(_SCRIPT, cfg, data)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
