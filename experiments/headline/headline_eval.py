"""Table 5 / Figure 1: headline ASR+ARC+XSTest with McNemar and bootstrap CIs."""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts, load_xstest
from core.evaluation import (
    _extract_behavior,
    classify_batch,
    eval_xstest,
    generate_only,
)
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.restoration import (
    extract_fp32_weights,
    get_proj_sets,
    measure_arc,
    restore_projections,
)
from core.seeds import DEFAULT_SEED, set_global_seed
from core.stats import bootstrap_recovery, mcnemar_exact, wilson_ci

_SCRIPT = Path(__file__)


def eval_condition(model, tokenizer, name, wj_items, arc_items, xstest_items, max_new_tokens):
    """
    Evaluate the condition.

    Inputs:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        name: The name of the condition.
        wj_items: The list of wild jailbreak prompts.
        arc_items: The list of arc challenge prompts.
        xstest_items: The list of xstest prompts.
        max_new_tokens: The maximum number of new tokens to generate.

    Outputs:
        resp: The list of responses.
        arc: The list of arc scores.
        xs: The list of xstest scores.
    """
    # Generate responses
    resp = generate_only(
        model = model, 
        tok = tokenizer, 
        prompts = wj_items, 
        max_new_tokens = max_new_tokens, 
        desc = f"{name}/ASR"
        )

    # Measure the arc scores
    arc = measure_arc(
        model = model, 
        tok = tokenizer, 
        prompts = arc_items,
        )
    # Evaluate the xstest scores
    xs = eval_xstest(
        model = model, 
        tok = tokenizer, 
        prompts = xstest_items, 
        desc = f"{name}/XSTest"
        )
    return resp, arc, xs


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--n_reasoning", type=int, default=100)
    ap.add_argument("--asr_max_new_tokens", type=int, default=256)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    # Set the global seed
    set_global_seed(DEFAULT_SEED)
    qtz = args.quantizer

    # Create the run config
    cfg = RunConfig(
        model=args.model,
        quantizer=qtz,
        output_stem="headline",
        extra={"n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "n_boot": args.n_boot},
    )
    raw_path = save_result(_SCRIPT, cfg, {}, raw=True)

    # Load the prompts and extract the behaviors
    wj_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in wj_items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    xstest_items = load_xstest()

    all_resp, all_arc, all_xs = {}, {}, {}

    # Define the checkpoint function
    def checkpoint():
        raw_path.write_text(
            json.dumps(
                {
                    "responses": all_resp,
                    "arc": all_arc,
                    "xstest": {
                        k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"}
                        for k, v in all_xs.items()
                    },
                },
                indent=2,
            )
        )

    # Load the fp32 model and extract the weights
    fp_model, tok = load_fp32(args.model)
    read_p, write_p = get_proj_sets(fp_model)
    write_w = extract_fp32_weights(fp_model, write_p)
    read_w = extract_fp32_weights(fp_model, read_p)
    all_resp["fp32"], all_arc["fp32"], all_xs["fp32"] = eval_condition(
        fp_model, tok, "fp32", wj_items, arc_items, xstest_items, args.asr_max_new_tokens
    )
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint()

    # Load the quantized model and extract the weights
    plan = [
        (qtz, []),
        ("write_restored", [(write_w, write_p)]),
        ("read_restored", [(read_w, read_p)]),
        ("both_restored", [(write_w, write_p), (read_w, read_p)]),
    ]
    for name, restores in plan:
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        for weights, projs in restores:
            restore_projections(q_model, weights, projs)
        all_resp[name], all_arc[name], all_xs[name] = eval_condition(
            q_model, tok, name, wj_items, arc_items, xstest_items, args.asr_max_new_tokens
        )
        del q_model
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint()

    # Classify the responses and calculate the ASR for each condition
    conditions = ["fp32", qtz, "write_restored", "read_restored", "both_restored"]
    flat_beh, flat_resp = [], []
    for c in conditions:
        flat_beh.extend(behaviors)
        flat_resp.extend(all_resp[c])
    flat_flags = classify_batch(flat_beh, flat_resp, device="cuda")
    n = args.n_eval
    flags_by_cond = {
        c: [bool(x) for x in flat_flags[i * n : (i + 1) * n]] for i, c in enumerate(conditions)
    }
    asr = {c: 100.0 * sum(flags_by_cond[c]) / n for c in conditions}
    fp_asr, q_asr = asr["fp32"], asr[qtz]
    damage = q_asr - fp_asr
    read_rec = q_asr - asr["read_restored"]
    write_rec = q_asr - asr["write_restored"]
    both_rec = q_asr - asr["both_restored"]

    # Calculate the McNemar test and the bootstrap recovery
    mcnemar = {
        f"{qtz}_vs_read": mcnemar_exact(flags_by_cond[qtz], flags_by_cond["read_restored"]),
        f"{qtz}_vs_write": mcnemar_exact(flags_by_cond[qtz], flags_by_cond["write_restored"]),
        "read_vs_write": mcnemar_exact(flags_by_cond["read_restored"], flags_by_cond["write_restored"]),
    }
    boot = bootstrap_recovery(flags_by_cond, qtz, n_boot=args.n_boot)

    # Calculate the xstest summary
    xstest_summary = {}
    for c in conditions:
        xs = all_xs[c]
        k, ns = xs["n_refusal_on_safe"], xs["n_safe"]
        xstest_summary[c] = {
            "false_refusal_rate": xs["false_refusal_rate"],
            "false_refusal_ci": wilson_ci(k, ns),
            "n_refusal_on_safe": k,
            "n_safe": ns,
        }

    # Save the results
    data = {
        "asr": asr,
        "arc": all_arc,
        "damage_pp": damage,
        "recovery_pp": {"read": read_rec, "write": write_rec, "both": both_rec},
        "recovery_ci_95": boot,
        "mcnemar": mcnemar,
        "xstest": xstest_summary,
        "flags_by_cond": flags_by_cond,
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
