# One condition of the headline eval (Table 5 / X1 / held-out) as a standalone piece, so a
# 7-8 h headline run fits <2 h interactive jobs. Same code path as headline_eval.py per
# condition: same prompts, generate_only/_generate_responses (greedy, batches of
# EVAL_BATCH_SIZE=16), measure_arc's scoring, eval_xstest, and the HarmBench judge
# (classify_responses scores one pair at a time, so labels do not depend on which other
# responses are judged in the same session).
#
# Saves per-prompt data: responses, generated token ids, HarmBench labels, per-item ARC
# correctness, per-prompt XSTest records, plus job id / node / commit / env so
# merge_pieces.py can refuse to combine pieces from different commits or environments.
#
# Prompt splitting (--prompt_lo/--prompt_hi) is allowed only on batch boundaries, so every
# generate() batch is identical to the full run's batch over the same prompts.
#
#   python -m experiments.headline.headline_piece --group heldout --model llama3-8b \
#       --quantizer aqlm2 --condition fp32 --parts wj,arc
# Output: outputs/headline_pieces/<group>/<model>_<quant>_<condition>[_p<lo>-<hi>]_<parts>.json
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts, load_xstest
from core.evaluation import _extract_behavior, _logprob_score, classify_responses, eval_xstest, generate_only
from core.manifest import RunConfig, _git_sha
from core.model_loader import LOAD_INFO, load_fp32, load_quantized
from core.restoration import extract_fp32_weights, get_proj_sets, restore_projections
from core.seeds import DEFAULT_SEED, set_global_seed

CONDITIONS = ("fp32", "quant", "write_restored", "read_restored", "both_restored")
PARTS = ("wj", "arc", "xstest")
_PKGS = ("torch", "transformers", "accelerate", "bitsandbytes", "aqlm", "auto_gptq", "optimum",
         "tokenizers", "safetensors", "numpy", "triton")


def _sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def env_fingerprint():
    """Everything that must match across pieces of one merge (node and job id excluded)."""
    vers = {}
    for p in _PKGS:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    return {
        "python": platform.python_version(),
        "python_exe": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": vers,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch": platform.machine(),
        "env": {k: os.environ.get(k) for k in ("EVAL_BATCH_SIZE", "SQ_GPTQ_DENSE",
                                                "CUBLAS_WORKSPACE_CONFIG", "QM_ROOT")},
        "seed": DEFAULT_SEED,
    }


def _git_dirty():
    """True if tracked files differ from HEAD (untracked outputs do not count)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                      stderr = subprocess.DEVNULL).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _arc_items(model, tokenizer, arc_items):
    """measure_arc() with per-item records; accuracy = same formula."""
    device = next(model.parameters()).device
    per = []
    for item in arc_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)
        per.append({"q_sha": _sha(item["question"]), "pred": int(pred),
                    "answer_idx": int(item["answer_idx"]), "correct": bool(pred == item["answer_idx"])})
    return 100.0 * sum(r["correct"] for r in per) / len(arc_items), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required = True, help = "piece set, e.g. heldout, x1_xstest, dettest_run1")
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required = True)
    ap.add_argument("--condition", required = True, choices = CONDITIONS)
    ap.add_argument("--parts", default = "wj,arc,xstest")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--n_reasoning", type = int, default = 100)
    ap.add_argument("--asr_max_new_tokens", type = int, default = 256)
    ap.add_argument("--n_xstest", type = int, default = None)
    ap.add_argument("--prompt_lo", type = int, default = 0)
    ap.add_argument("--prompt_hi", type = int, default = None)
    ap.add_argument("--allow_dirty", action = "store_true", help = "testing only; merge refuses dirty pieces")
    args = ap.parse_args()

    parts = [p for p in args.parts.replace("+", ",").split(",") if p]
    assert all(p in PARTS for p in parts), parts
    bs = int(os.environ.get("EVAL_BATCH_SIZE", 16))
    lo, hi = args.prompt_lo, (args.prompt_hi if args.prompt_hi is not None else args.n_eval)
    split = (lo, hi) != (0, args.n_eval)
    if split:
        if "wj" not in parts or len(parts) != 1:
            raise SystemExit("prompt split pieces must use --parts wj only")
        if lo % bs or (hi % bs and hi != args.n_eval) or not 0 <= lo < hi <= args.n_eval:
            raise SystemExit(f"prompt range {lo}:{hi} is not on batch boundaries (batch {bs})")

    dirty = _git_dirty()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files differ from HEAD; commit first (pieces record the commit)")

    set_global_seed(DEFAULT_SEED)
    qtz = args.quantizer
    cond_key = qtz if args.condition == "quant" else args.condition
    fm = RunConfig(model = args.model, quantizer = qtz).filename_model()
    rng = f"_p{lo}-{hi}" if split else ""
    out = (Path("outputs/headline_pieces") / args.group /
           f"{fm}_{qtz}_{args.condition}{rng}_{'-'.join(parts)}.json")
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents = True, exist_ok = True)

    t0 = time.time()
    timings = {}
    wj_items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = [_extract_behavior(it) for it in wj_items]
    prompt_sha = [_sha((it.get("prompt") or it.get("goal") or "") if isinstance(it, dict) else it)
                  for it in wj_items]
    arc_items = load_arc_challenge(n_samples = args.n_reasoning)
    xstest_items = load_xstest()
    if args.n_xstest:
        half = max(1, args.n_xstest // 2)
        xstest_items = ([it for it in xstest_items if it["type"] == "safe"][:half]
                        + [it for it in xstest_items if it["type"] != "safe"][:half])

    # Model for this condition (restoration: FP32 weights first, as headline_eval does)
    t = time.time()
    if args.condition == "fp32":
        model, tok = load_fp32(args.model)
    else:
        restores = []
        if args.condition != "quant":
            fp_model, tok = load_fp32(args.model)
            read_p, write_p = get_proj_sets(fp_model)
            if args.condition in ("write_restored", "both_restored"):
                restores.append((extract_fp32_weights(fp_model, write_p), write_p))
            if args.condition in ("read_restored", "both_restored"):
                restores.append((extract_fp32_weights(fp_model, read_p), read_p))
            del fp_model
            gc.collect()
            torch.cuda.empty_cache()
        model, tok_q = load_quantized(args.model, quantizer = qtz)
        if args.condition == "quant":
            tok = tok_q
        for weights, projs in restores:
            restore_projections(model, weights, projs)
    load_info = dict(LOAD_INFO)
    timings["load_s"] = round(time.time() - t, 1)

    data = {"prompt_sha": prompt_sha[lo:hi]}
    if "wj" in parts:
        # record generated token ids without changing the generation call
        rec = []
        orig_generate = model.generate

        def _rec_generate(*a, **k):
            o = orig_generate(*a, **k)
            n_in = k["input_ids"].shape[1]
            rec.extend(o[:, n_in:].tolist())
            return o
        model.generate = _rec_generate
        t = time.time()
        resp = generate_only(model, tok, wj_items[lo:hi], max_new_tokens = args.asr_max_new_tokens,
                             desc = f"{cond_key}/ASR")
        timings["generate_s"] = round(time.time() - t, 1)
        model.generate = orig_generate
        data.update({"responses": resp, "gen_token_ids": rec})

    if "arc" in parts:
        t = time.time()
        arc, arc_per = _arc_items(model, tok, arc_items)
        timings["arc_s"] = round(time.time() - t, 1)
        data.update({"arc": arc, "arc_per_item": arc_per})

    if "xstest" in parts:
        t = time.time()
        xs = eval_xstest(model, tok, xstest_items, desc = f"{cond_key}/XSTest")
        timings["xstest_s"] = round(time.time() - t, 1)
        data["xstest"] = xs

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if "wj" in parts:
        t = time.time()
        flags = classify_responses(behaviors[lo:hi], data["responses"], device = "cuda")
        timings["judge_s"] = round(time.time() - t, 1)
        data["flags"] = [bool(x) for x in flags]
        print(f"  {cond_key} ASR on prompts {lo}:{hi} = {100.0 * sum(flags) / len(flags):.1f}%")

    timings["total_s"] = round(time.time() - t0, 1)
    payload = {
        "piece": {"group": args.group, "model": args.model, "quantizer": qtz,
                  "condition": args.condition, "cond_key": cond_key, "parts": parts,
                  "prompt_lo": lo, "prompt_hi": hi, "n_eval": args.n_eval,
                  "n_reasoning": args.n_reasoning, "asr_max_new_tokens": args.asr_max_new_tokens,
                  "n_xstest": args.n_xstest, "batch_size": bs},
        "run": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
                "partition": os.environ.get("SLURM_JOB_PARTITION"), "git_sha": _git_sha(),
                "git_dirty": dirty, "load_info": load_info, "timings": timings,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
        "env": env_fingerprint(),
        **data,
    }
    out.write_text(json.dumps(payload, indent = 1, default = str))
    print(f"timings: {timings}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
