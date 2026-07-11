"""Backward-compat shim — use experiments/regime_sweep/restoration_sweep.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/regime_sweep/restoration_sweep.py"),
        run_name="__main__",
    )
