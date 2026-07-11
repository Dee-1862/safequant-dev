"""Backward-compat shim — use experiments/signal_geometry/probe_accuracy.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/signal_geometry/probe_accuracy.py"),
        run_name="__main__",
    )
