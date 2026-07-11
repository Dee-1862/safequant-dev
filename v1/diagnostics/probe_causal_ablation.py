"""Backward-compat shim."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/boundary/probe_causal_ablation.py"),
        run_name="__main__",
    )
