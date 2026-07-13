import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.model_loader import load_quantized  # noqa: E402
from core.datasets import (load_wildjailbreak_prompts, load_arc_challenge,  # noqa: E402
                           load_xstest)
from core.evaluation import (generate_only, classify_batch, _extract_behavior,  # noqa: E402
                             eval_xstest)
from core.manifest import RunConfig, find_result, result_path, save_result  # noqa: E402
from core.restoration import measure_arc  # noqa: E402
from experiments.trained_repair.qresafe_repair import (attach_masked_lora, set_lora_enabled,  # noqa: E402
                                        unwrap_lora)
from experiments.trained_repair.qresafe_fold import (quant_cfg, fold_and_writeback_,  # noqa: E402
                                                dequantize_module)

# --------------------------------------------------------------------------- #
# DPO math
# --------------------------------------------------------------------------- #
def _tokenize_pair(tok, prompt, completion, device):
    msgs = [{"role": "user", "content": prompt}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")[0]
    c_ids = tok(completion, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(device)
    n_prompt = p_ids.shape[0]
    return input_ids, n_prompt


def seq_logp(model, input_ids, n_prompt):
    """Sum of token log-probs of the completion (positions >= n_prompt)."""
    logits = model(input_ids=input_ids).logits[0]            # [T, V]
    logp = F.log_softmax(logits[:-1].float(), dim=-1)        # predict token t+1
    tgt = input_ids[0, 1:]                                   # [T-1]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [T-1]
    # completion tokens are those predicted from position n_prompt-1 onward
    return tok_lp[n_prompt - 1:].sum()


def dpo_loss(lp_w_pol, lp_l_pol, lp_w_ref, lp_l_ref, beta):
    """Standard DPO: -log sigmoid(beta * [(pol_w-ref_w) - (pol_l-ref_l)])."""
    logits = beta * ((lp_w_pol - lp_w_ref) - (lp_l_pol - lp_l_ref))
    return -F.logsigmoid(logits)


@torch.no_grad()
def precompute_ref(model, tok, pairs, device):
    set_lora_enabled(model, False)              # reference = base, no adapter
    ref = []
    for pr in pairs:
        iw, nw = _tokenize_pair(tok, pr["instruction"], pr["yw"], device)
        il, nl = _tokenize_pair(tok, pr["instruction"], pr["yl"], device)
        ref.append((float(seq_logp(model, iw, nw)), float(seq_logp(model, il, nl))))
    set_lora_enabled(model, True)
    return ref


def train_dpo(model, tok, pairs, ref, lora_params, beta, lr, epochs, device, log_every=50,
              opt8bit=False):
    # At large rank (r>=1024) the fp32 AdamW optimizer states (2x params) dominate GPU
    # memory; 8-bit AdamW cuts that 4x so the paper's r=2048 fits on a single 80GB card.
    if opt8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(lora_params, lr=lr)
        print("[opt] using bitsandbytes AdamW8bit")
    else:
        opt = torch.optim.AdamW(lora_params, lr=lr)
    set_lora_enabled(model, True)
    model.train()
    order = list(range(len(pairs)))
    g = torch.Generator().manual_seed(0)
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(pairs), generator=g).tolist()
        for i in perm:
            pr = pairs[i]
            iw, nw = _tokenize_pair(tok, pr["instruction"], pr["yw"], device)
            il, nl = _tokenize_pair(tok, pr["instruction"], pr["yl"], device)
            lp_w = seq_logp(model, iw, nw)
            lp_l = seq_logp(model, il, nl)
            loss = dpo_loss(lp_w, lp_l, ref[i][0], ref[i][1], beta)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % log_every == 0:
                print(f"    [dpo] ep{ep} step{step} loss={float(loss):.4f}")
    model.eval()


# --------------------------------------------------------------------------- #
# fold + eval
# --------------------------------------------------------------------------- #
def fold_all(model, wrapped_names, cfg, beam_size=5):
    """Fold each LoRALinear's delta into its base codes, then unwrap. Returns the
    per-module codes-changed fraction (the fold-survival signal)."""
    mods = dict(model.named_modules())
    changed = {}
    for name in wrapped_names:
        m = mods[name]
        if type(m).__name__ != "LoRALinear":
            continue
        delta = m.delta_weight().detach()                    # scaling * B@A  [out,in]
        frac = fold_and_writeback_(m.base, delta, cfg, beam_size=beam_size)
        changed[name] = frac
    unwrap_lora(model, wrapped_names)                        # now pure 2-bit, no adapter
    return changed


def eval_model(model, tok, n_eval, n_arc, label):
    wj = load_wildjailbreak_prompts(n_samples=n_eval, config="eval")
    beh = [_extract_behavior(it) for it in wj]
    resp = generate_only(model, tok, wj, desc=f"{label}/ASR")
    flags = classify_batch(beh, resp, device="cuda")
    asr = 100.0 * sum(flags) / len(wj)
    arc = measure_arc(model, tok, load_arc_challenge(n_samples=n_arc))
    xs = eval_xstest(model, tok, load_xstest())
    print(f"  [{label}] ASR={asr:.1f}% ARC={arc:.0f}% XSTest-FR={xs['false_refusal_rate']:.1f}%")
    return {"asr": asr, "arc": arc, "xstest_fr": xs["false_refusal_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=["read", "write", "full"])
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    # Official Q-Resafe llama7b recipe (paper_refs/Qresafe/quant-with-ft/configs/
    # llama7b.yaml): r=128, alpha=128, beta=0.01, lr=5e-7, epochs=1. These are the
    # DEFAULTS; override to sweep and show how recovery increases/decreases. Pair a
    # sweep with --tag so each point writes its own output file.
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=128.0,
                    help="LoRA alpha (official 128; our earlier runs used 256 = 2x scaling)")
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=5e-7,
                    help="DPO learning rate (official 5e-7; our earlier runs used 5e-6 = 10x)")
    ap.add_argument("--epochs", type=int, default=1,
                    help="DPO epochs (official 1; our earlier runs used 3)")
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--n_arc", type=int, default=100)
    ap.add_argument("--pairs_file", default=None,
                    help="displacement pairs JSON (default: qresafe_pairs_<model>_<quant>_wj.json)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--tag", default=None,
                    help="suffix appended to the output variant so hyperparameter-sweep "
                         "points do not overwrite each other, e.g. --tag lr5e-6 writes "
                         "outputs/trained_repair/qresafe_dpo_<model>_<quant>_<mask>_lr5e-6.json")
    ap.add_argument("--opt8bit", action="store_true", default=False,
                    help="use 8-bit AdamW (bitsandbytes) to fit large rank (e.g. r=2048)")
    ap.add_argument("--save_adapter", action="store_true", default=False,
                    help="save the large LoRA adapter .pt (off by default; only for re-folding)")
    ap.add_argument("--seed", type=int, default=0,
                    help="global RNG seed for reproducible LoRA init + DPO training")
    args = ap.parse_args()

    # Seed every RNG BEFORE attach_masked_lora (kaiming LoRA init) and DPO, so the
    # whole run is reproducible. Eval is greedy (deterministic) regardless.
    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[seed] global seed = {args.seed}")

    device = "cuda"
    _SCRIPT = Path(__file__)
    pairs_cfg = RunConfig(model=args.model, quantizer=args.quantizer, variant="wj")
    pairs_path = (
        Path(args.pairs_file)
        if args.pairs_file
        else find_result(
            Path("experiments/trained_repair/qresafe_pairs.py"), pairs_cfg
        )
        or result_path(Path("experiments/trained_repair/qresafe_pairs.py"), pairs_cfg)
    )
    pairs = json.loads(pairs_path.read_text())["pairs"]
    print(f"[data] {len(pairs)} displacement pairs from {pairs_path} | mask={args.mask}")

    model, tok = load_quantized(args.model, quantizer=args.quantizer)
    cfg = quant_cfg(model)
    lora_params, wrapped = attach_masked_lora(model, args.mask, r=args.r, alpha=args.alpha)
    n_train = sum(p.numel() for p in lora_params)
    print(f"[lora] mask={args.mask} wrapped={len(wrapped)} trainable={n_train}")

    print("[dpo] precomputing reference log-probs...")
    ref = precompute_ref(model, tok, pairs, device)
    print(f"[dpo] training beta={args.beta} lr={args.lr} epochs={args.epochs}")
    train_dpo(model, tok, pairs, ref, lora_params, args.beta, args.lr, args.epochs, device,
              opt8bit=args.opt8bit)

    # branch (1) needs the FP+LoRA model evaluated BEFORE folding
    set_lora_enabled(model, True)
    fp_lora = eval_model(model, tok, args.n_eval, args.n_arc, f"{args.mask}/FP+LoRA")

    # Checkpoint the trained adapter BEFORE the lossy fold, so re-folding with a
    # stronger (codes+codebooks) variant never needs retraining. Also records the
    # delta magnitude per module -- the quantity that decides fold survival.
    adapter = {}
    for n, m in model.named_modules():
        if type(m).__name__ == "LoRALinear":
            dw = m.delta_weight().detach()
            adapter[n] = {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu(),
                          "scaling": m.scaling, "delta_relnorm": float(dw.norm() /
                          (dequantize_module(m.base, cfg["nbits"]).norm() + 1e-9))}
    # The adapter .pt is large and only needed for a stronger re-fold; never let a
    # disk failure here lose the result JSON. Opt-in, and guarded.
    if args.save_adapter:
        try:
            out_json = args.output or str(
                result_path(_SCRIPT, RunConfig(model=args.model, quantizer=args.quantizer, variant=args.mask))
            )
            torch.save(adapter, out_json.replace(".json", "_adapter.pt"))
        except Exception as e:
            print(f"[warn] adapter save failed ({type(e).__name__}: {e}); continuing")
    rel = [v["delta_relnorm"] for v in adapter.values()]
    print(f"[delta] trained LoRA delta rel-norm: mean={sum(rel)/len(rel):.4f} "
          f"min={min(rel):.4f} max={max(rel):.4f}  (vs ~0.016 snap threshold seen in smoke)")

    # fold the delta into 2-bit codes, unwrap -> pure 2-bit
    changed = fold_all(model, wrapped, cfg)
    mean_changed = sum(changed.values()) / max(len(changed), 1)
    print(f"[fold] mean codes-changed fraction = {mean_changed:.4f} "
          f"({'delta SURVIVES' if mean_changed > 1e-4 else 'delta SNAPPED AWAY'})")

    folded = eval_model(model, tok, args.n_eval, args.n_arc, f"{args.mask}/folded-2bit")

    run_cfg = RunConfig(
        model=args.model,
        quantizer=args.quantizer,
        variant=(f"{args.mask}_{args.tag}" if args.tag else args.mask),
        extra={
            "r": args.r,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_eval": args.n_eval,
            "pairs_file": str(pairs_path),
        },
    )
    data = {
        "n_pairs": len(pairs),
        "wrapped_modules": len(wrapped),
        "mean_codes_changed": mean_changed,
        "fp_lora": fp_lora,
        "folded": folded,
        "baselines": {"fp32_asr": 45.5, "aqlm2_asr": 63.2, "fp32_xstest": 6.0},
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**data, "manifest": run_cfg.to_dict()}, indent=2))
    else:
        out_path = save_result(_SCRIPT, run_cfg, data)
    print(f"\n[saved] {out_path}")
    print(f"  branch signal: FP+LoRA ASR={fp_lora['asr']:.1f}  folded ASR={folded['asr']:.1f}  "
          f"codes-changed={mean_changed:.4f}  XSTest folded={folded['xstest_fr']:.1f}% (FP 6.0%)")


# --------------------------------------------------------------------------- #
# DPO-math self-test (CPU, no model)
# --------------------------------------------------------------------------- #
def _self_test():
    ok = True
    # prefer-yw beyond ref -> loss small; prefer-yl -> loss large
    good = float(dpo_loss(torch.tensor(0.0), torch.tensor(-2.0),
                          torch.tensor(0.0), torch.tensor(0.0), beta=1.0))
    bad = float(dpo_loss(torch.tensor(-2.0), torch.tensor(0.0),
                         torch.tensor(0.0), torch.tensor(0.0), beta=1.0))
    neutral = float(dpo_loss(torch.tensor(0.0), torch.tensor(0.0),
                             torch.tensor(0.0), torch.tensor(0.0), beta=1.0))
    print(f"  dpo_loss good={good:.4f} neutral={neutral:.4f} bad={bad:.4f}")
    if not (good < neutral < bad):
        print("  FAIL: dpo_loss not monotone in the preference margin"); ok = False
    if abs(neutral - 0.6931) > 1e-3:   # -log sigmoid(0) = ln 2
        print("  FAIL: neutral loss != ln2"); ok = False
    # gradient pushes lora toward preferring yw
    print("\n  DPO MATH GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(0 if _self_test() else 1)
    main()
