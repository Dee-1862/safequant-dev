"""Backward-compat shim — use per_projection.py --layer_lo/--layer_hi."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/localization/per_projection.py"),
        run_name="__main__",
    )
