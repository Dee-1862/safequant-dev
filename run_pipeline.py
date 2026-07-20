#!/usr/bin/env python
"""SafeQuant reproduction entrypoint — every paper table mapped to its exact
experiment command, in one place.

This is the portable, human-readable companion to scripts/*.sbatch (which are
the Bridges-2-specific SLURM record). Use it to see, run, or submit the exact
pipeline that produces each table.

    python run_pipeline.py --list                 # table -> command map (docs)
    python run_pipeline.py --table 1 --dry-run     # print the commands for a table
    python run_pipeline.py --table 5 --run         # run a table's steps locally (needs GPU)
    python run_pipeline.py --all --sbatch          # emit sbatch submits for the cluster
    python run_pipeline.py --results               # (re)build the aggregate result files

Prerequisites (not run here): quantized checkpoints must already exist under
$QUANTIZED_MODELS (base + AWQ/AQLM downloads via scripts/fetch_checkpoints.sh;
self-generated GPTQ via scripts/quantize_gptq_all.sh in the sq_autogptq env).
Everything below runs in the eval env (conda 'sq_clean').
"""
import argparse
import subprocess
import sys

# Valid (model, quantizer) cells. Qwen has no AQLM-2 by design.
MODELS = ["llama7b", "qwen7b", "mistral7b-v02", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]


def cells():
    for m in MODELS:
        for q in QUANTS:
            if m == "qwen7b" and q == "aqlm2":
                continue
            yield m, q


def M(mod, *args):
    """One command: python -m <mod> <args...>."""
    return [sys.executable, "-u", "-m", mod, *map(str, args)]


def _drift():  # Table 1 signal-geometry sweep, 3 experiments x 19 cells
    cmds = []
    for exp in ("direction_drift", "probe_accuracy", "probe_rotation"):
        for m, q in cells():
            cmds.append(M(f"experiments.signal_geometry.{exp}",
                          "--model", m, "--quantizer", q, "--n_calib", 50))
    return cmds


def _restore():  # Table 2 damaged-coherent screen, 19 cells (ARC n=100 canonical)
    return [M("experiments.regime_sweep.restoration_sweep",
              "--model", m, "--quantizer", q,
              "--n_eval", 200, "--n_reasoning", 100) for m, q in cells()]


def _qresafe():  # Table 7 fold-reverts rank sweep, both-sides, ranks 128..2048
    out = []
    for r in (128, 256, 512, 1024, 2048):
        out.append(M("experiments.trained_repair.qresafe_dpo",
                     "--mask", "full", "--r", r, "--alpha", 2 * r, "--tag", f"r{r}",
                     "--beta", 0.01, "--epochs", 1, "--lr", "5e-7",
                     "--n_eval", 200, "--opt8bit", "--seed", 42))
    return out


def _lrqat():  # Table 8 matched-budget LR-QAT, role x n_bits
    budget = ["--steps", 3000, "--lr", "1e-4", "--rank", 32, "--alpha", 64,
              "--seq_len", 512, "--grad_accum", 8, "--n_train_docs", 4000,
              "--group_size", 128, "--n_eval", 200, "--n_reasoning", 50,
              "--n_xstest", 50]
    return [M("experiments.trained_repair.lrqat", "--model", "llama7b",
              "--quantizer", "aqlm2", "--role", role, "--n_bits", b, *budget)
            for role in ("read", "write") for b in (2, 4)]


# table label -> (title, list-of-command builders). Order = dependency order.
PIPELINE = {
    "1": ("Table 1  tab:geometry (drift, probe acc, probe rotation; 19 cells)", _drift),
    "2": ("Table 2  tab:regime (damaged-coherent screen; 19 cells, ARC n=100)", _restore),
    "3": ("Table 3  tab:f3a_proj (per-projection, Llama middle band)",
          lambda: [M("experiments.localization.per_projection", "--model", "llama7b",
                     "--quantizer", "aqlm2", "--n_eval", 200, "--n_reasoning", 50,
                     "--layer_lo", 11, "--layer_hi", 22)]),
    "4": ("Table 4  tab:f3a_range (depth bands, Llama, read+write)",
          lambda: [M("experiments.localization.depth_bands", "--model", "llama7b",
                     "--quantizer", "aqlm2", "--projections", side,
                     "--n_eval", 200, "--n_reasoning", 50)
                   for side in ("read", "write")]),
    "5": ("Table 5  tab:f3a (static restoration + McNemar/CI/XSTest, Llama)",
          lambda: [M("experiments.headline.headline_eval", "--model", "llama7b",
                     "--quantizer", "aqlm2", "--n_eval", 200, "--n_reasoning", 100,
                     "--n_boot", 2000)]),
    "6": ("Table 6  tab:controls (param-match, refit floor, cosmed; Llama+Phi-3)",
          lambda: [M("experiments.regime_sweep.param_matched_restore", "--model", m,
                     "--quantizer", "aqlm2", "--n_eval", 200, "--n_reasoning", 50,
                     "--seeds", 3) for m in ("llama7b", "phi3")]
                  + [M("experiments.boundary.probe_refit_control", "--model", m,
                       "--n_calib", 200) for m in ("llama7b", "phi3")]
                  + [M("experiments.signal_geometry.cosmed_after_restore", "--model", m,
                       "--quantizer", "aqlm2", "--n_calib", 200)
                     for m in ("llama7b", "phi3")]),
    "7": ("Table 7  tab:trained_repair (Q-Resafe fold-reverts rank sweep)", _qresafe),
    "8": ("Table 8  tab:lrqat (LR-QAT, role x 2/4-bit)", _lrqat),
    "leakage": ("Prose control: probe leakage (TF-IDF surface + XSTest; Llama+Phi-3)",
                lambda: [M("experiments.boundary.probe_leakage_control", "--model", m,
                           "--n_calib", 200, "--n_transfer", 200)
                         for m in ("llama7b", "phi3")]),
    "arditi": ("Prose: refusal-direction ablation (Arditi; Llama AQLM-2 + FP32)",
               lambda: [M("experiments.behavioral_direction.arditi_repro",
                          "--model", "llama7b", "--quantizer", q)
                        for q in ("aqlm2", "fp32")]),
}


def selected(args):
    keys = list(PIPELINE) if args.all else ([args.table] if args.table else [])
    if not keys:
        sys.exit("nothing selected: pass --list, --table <n>, or --all")
    bad = [k for k in keys if k not in PIPELINE]
    if bad:
        sys.exit(f"unknown table(s): {bad}. valid: {list(PIPELINE)}")
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the table->command map")
    ap.add_argument("--table", help="table key (1-8, leakage, arditi)")
    ap.add_argument("--all", action="store_true", help="every table, in order")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print commands, don't run")
    g.add_argument("--run", action="store_true", help="run steps locally (needs GPU)")
    g.add_argument("--sbatch", action="store_true",
                   help="print the matching scripts/*.sbatch submits for the cluster")
    ap.add_argument("--results", action="store_true",
                    help="(re)build aggregate_results.py + paper_values.py outputs")
    args = ap.parse_args()

    if args.results:
        subprocess.run(M("scripts.aggregate_results"), check=False)
        subprocess.run([sys.executable, "scripts/paper_values.py"], check=False)
        return

    if args.list or not (args.table or args.all):
        print("SafeQuant pipeline -- table -> experiments\n")
        for k, (title, build) in PIPELINE.items():
            print(f"[{k}] {title}")
            print(f"      {len(build())} command(s). Run: python run_pipeline.py "
                  f"--table {k} --dry-run\n")
        print("Prereqs: quantized checkpoints under $QUANTIZED_MODELS. Env: sq_clean.")
        print("Results:  python run_pipeline.py --results")
        return

    for k in selected(args):
        title, build = PIPELINE[k]
        cmds = build()
        print(f"\n### [{k}] {title}  ({len(cmds)} step(s))")
        for c in cmds:
            printable = " ".join(c[3:])  # drop the "python -u -m" prefix
            print(f"  python -m {printable}")
            if args.run:
                rc = subprocess.run(c).returncode
                if rc != 0:
                    print(f"  !! exited {rc}; continuing")
    if args.sbatch:
        print("\nCluster: submit the matching scripts/<area>/*.sbatch with "
              "--export=ALL (see scripts/ for the SLURM record).")


if __name__ == "__main__":
    main()
