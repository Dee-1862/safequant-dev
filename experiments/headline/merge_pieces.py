# Merge headline_piece.py outputs into a headline-style result (CPU only, no torch).
# Refuses to combine pieces whose commit, dirty flag, environment fingerprint or eval
# settings differ, or whose prompt ranges / prompt hashes do not line up.
# Statistics use the same functions and settings as headline_eval.py (ASR, McNemar,
# bootstrap_recovery n_boot 2000 seed 0, Wilson CIs) plus paired_screen's paired dASR
# (paired_delta_asr, 2000 resamples, seed 42).
#
#   python -m experiments.headline.merge_pieces --group heldout --model llama3-8b \
#       --quantizer aqlm2 --variant heldout
#   python -m experiments.headline.merge_pieces --self_test outputs/headline/headline_llama_aqlm2.json
# Output: outputs/headline/headline_<model>_<quant>_<variant>.json (never overwrites)
import argparse
import json
import sys
from pathlib import Path

from core.manifest import RunConfig, assert_new_result, save_result
from core.stats import bootstrap_recovery, mcnemar_exact, paired_delta_asr, wilson_ci

_SCRIPT = Path(__file__)
SEED = 42            # core.seeds.DEFAULT_SEED (not imported: it pulls in torch)
RESTORED = ("write_restored", "read_restored", "both_restored")
_SETTINGS = ("n_eval", "n_reasoning", "asr_max_new_tokens", "n_xstest", "batch_size")


def summarize(qtz, flags_by_cond, arc, xs_counts, n_boot = 2000):
    """Headline statistics from per-condition labels. Fields as headline_eval.py, plus paired_dasr."""
    out = {}
    if flags_by_cond:
        n = len(next(iter(flags_by_cond.values())))
        asr = {c: 100.0 * sum(f) / n for c, f in flags_by_cond.items()}
        out["asr"] = asr
        if arc:
            out["arc"] = arc
        if "fp32" in asr and qtz in asr:
            out["damage_pp"] = asr[qtz] - asr["fp32"]
            out["paired_dasr"] = paired_delta_asr([int(bool(x)) for x in flags_by_cond["fp32"]],
                                                  [int(bool(x)) for x in flags_by_cond[qtz]],
                                                  n_boot = 2000, seed = SEED)
        if qtz in asr and all(c in asr for c in RESTORED):
            out["recovery_pp"] = {"read": asr[qtz] - asr["read_restored"],
                                  "write": asr[qtz] - asr["write_restored"],
                                  "both": asr[qtz] - asr["both_restored"]}
        mc = {}
        for name, a, b in ((f"{qtz}_vs_read", qtz, "read_restored"),
                           (f"{qtz}_vs_write", qtz, "write_restored"),
                           ("read_vs_write", "read_restored", "write_restored")):
            if a in flags_by_cond and b in flags_by_cond:
                mc[name] = mcnemar_exact(flags_by_cond[a], flags_by_cond[b])
        if "fp32" in flags_by_cond and qtz in flags_by_cond:
            mc[f"fp32_vs_{qtz}"] = mcnemar_exact(flags_by_cond["fp32"], flags_by_cond[qtz])
        if mc:
            out["mcnemar"] = mc
        if all(c in flags_by_cond for c in ("fp32", qtz) + RESTORED):
            out["recovery_ci_95"] = bootstrap_recovery(flags_by_cond, qtz, n_boot = n_boot)
        out["flags_by_cond"] = flags_by_cond
    elif arc:
        out["arc"] = arc
    if xs_counts:
        out["xstest"] = {c: {"false_refusal_rate": v["false_refusal_rate"],
                             "false_refusal_ci": wilson_ci(v["n_refusal_on_safe"], v["n_safe"]),
                             "n_refusal_on_safe": v["n_refusal_on_safe"], "n_safe": v["n_safe"]}
                         for c, v in xs_counts.items()}
    return out


def _fail(msg):
    raise SystemExit(f"REFUSING TO MERGE: {msg}")


def load_pieces(group, model, quantizer):
    fm = RunConfig(model = model, quantizer = quantizer).filename_model()
    files = sorted((Path("outputs/headline_pieces") / group).glob(f"{fm}_{quantizer}_*.json"))
    if not files:
        _fail(f"no pieces in outputs/headline_pieces/{group} for {fm}_{quantizer}")
    pieces = [(f, json.loads(f.read_text())) for f in files]
    ref_f, ref = pieces[0]
    for f, p in pieces:
        if p["run"]["git_dirty"]:
            _fail(f"{f.name} ran on a dirty tree")
        if p["run"]["git_sha"] != ref["run"]["git_sha"]:
            _fail(f"commit differs: {f.name} {p['run']['git_sha']} vs {ref_f.name} {ref['run']['git_sha']}")
        if p["env"] != ref["env"]:
            diff = {k: (p["env"].get(k), ref["env"].get(k)) for k in set(p["env"]) | set(ref["env"])
                    if p["env"].get(k) != ref["env"].get(k)}
            _fail(f"environment differs between {f.name} and {ref_f.name}: {diff}")
        for k in _SETTINGS:
            if p["piece"][k] != ref["piece"][k]:
                _fail(f"setting {k} differs: {f.name} {p['piece'][k]} vs {ref_f.name} {ref['piece'][k]}")
    return pieces


def assemble(pieces, qtz):
    """Per condition: concatenate WJ prompt ranges, collect ARC and XSTest."""
    n_eval = pieces[0][1]["piece"]["n_eval"]
    wj, arc, xs = {}, {}, {}
    for f, p in pieces:
        c = p["piece"]["cond_key"]
        if "wj" in p["piece"]["parts"]:
            wj.setdefault(c, []).append(p)
        if "arc" in p["piece"]["parts"]:
            if c in arc:
                _fail(f"two ARC pieces for {c}")
            arc[c] = p["arc"]
        if "xstest" in p["piece"]["parts"]:
            if c in xs:
                _fail(f"two XSTest pieces for {c}")
            xs[c] = p["xstest"]
    flags, prompt_sha = {}, None
    for c, ps in wj.items():
        ps.sort(key = lambda p: p["piece"]["prompt_lo"])
        pos = 0
        fl, sh = [], []
        for p in ps:
            if p["piece"]["prompt_lo"] != pos:
                _fail(f"{c}: prompt ranges not contiguous at {pos}")
            pos = p["piece"]["prompt_hi"]
            fl += p["flags"]
            sh += p["prompt_sha"]
        if pos != n_eval or len(fl) != n_eval:
            _fail(f"{c}: prompts cover 0:{pos}, need 0:{n_eval}")
        if prompt_sha is None:
            prompt_sha = sh
        elif sh != prompt_sha:
            _fail(f"{c}: prompt order differs from other conditions")
        flags[c] = fl
    xs_counts = {c: {k: v[k] for k in ("false_refusal_rate", "n_refusal_on_safe", "n_safe")} for c, v in xs.items()}
    return flags, arc, xs_counts


def self_test(path):
    """Rebuild a full headline result from its per-condition labels; all stats must match exactly."""
    d = json.loads(Path(path).read_text())
    qtz = d["manifest"]["quantizer"]
    xs_counts = {c: {"false_refusal_rate": v["false_refusal_rate"], "n_refusal_on_safe": v["n_refusal_on_safe"],
                     "n_safe": v["n_safe"]} for c, v in d["xstest"].items()}
    s = summarize(qtz, d["flags_by_cond"], d["arc"], xs_counts, n_boot = d["manifest"].get("n_boot", 2000))
    ok = True
    for k in ("asr", "arc", "damage_pp", "recovery_pp", "recovery_ci_95", "xstest", "flags_by_cond"):
        same = s.get(k) == d.get(k)
        ok &= same
        print(f"  {k:16s} {'identical' if same else 'DIFFERENT'}")
    for k, v in d["mcnemar"].items():
        same = s["mcnemar"].get(k) == v
        ok &= same
        print(f"  mcnemar {k:14s} {'identical' if same else 'DIFFERENT'}  (b={v['b']} c={v['c']} p={v['p_value']:.3g})")
    # paired dASR as paired_screen computes it from the original file
    from experiments.regime_sweep.paired_screen import load_labels
    fp, q = load_labels(path, qtz)
    ref = paired_delta_asr(fp, q, n_boot = 2000, seed = SEED)
    same = ref == s["paired_dasr"]
    ok &= same
    print(f"  paired dASR      {'identical' if same else 'DIFFERENT'}  (dASR {ref['delta_asr']:+.1f}, "
          f"CI [{ref['ci95'][0]:+.1f}, {ref['ci95'][1]:+.1f}], McNemar p {ref['p_mcnemar']:.3g})")
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    ap.add_argument("--model")
    ap.add_argument("--quantizer")
    ap.add_argument("--variant", help = "output variant, e.g. heldout -> headline_<m>_<q>_heldout.json")
    ap.add_argument("--conditions", default = None, help = "comma list of cond keys that must be present")
    ap.add_argument("--self_test", metavar = "HEADLINE_JSON")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test(args.self_test) else 1)

    pieces = load_pieces(args.group, args.model, args.quantizer)
    flags, arc, xs_counts = assemble(pieces, args.quantizer)
    if args.conditions:
        have = set(flags) | set(arc) | set(xs_counts)
        miss = [c for c in args.conditions.split(",") if c not in have]
        if miss:
            _fail(f"missing conditions: {miss}")
    ref = pieces[0][1]
    cfg = RunConfig(model = args.model, quantizer = args.quantizer, variant = args.variant,
                    output_stem = "headline",
                    extra = {k: ref["piece"][k] for k in _SETTINGS} | {"merged_from_pieces": args.group})
    assert_new_result(_SCRIPT, cfg)
    data = summarize(args.quantizer, flags, arc, xs_counts)
    data["pieces"] = [{"file": str(f), "condition": p["piece"]["cond_key"], "parts": p["piece"]["parts"],
                       "prompts": [p["piece"]["prompt_lo"], p["piece"]["prompt_hi"]],
                       "job_id": p["run"]["job_id"], "node": p["run"]["node"],
                       "timings": p["run"]["timings"]} for f, p in pieces]
    data["piece_commit"] = ref["run"]["git_sha"]
    data["piece_env"] = ref["env"]
    path = save_result(_SCRIPT, cfg, data)
    print(json.dumps({k: data[k] for k in ("asr", "arc", "damage_pp", "mcnemar", "xstest") if k in data},
                     indent = 1, default = str))
    if "paired_dasr" in data:
        r = data["paired_dasr"]
        print(f"paired dASR {r['delta_asr']:+.1f} pp, 95% CI [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}], "
              f"McNemar p {r['p_mcnemar']:.3g} (b={r['b']}, c={r['c']})")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
