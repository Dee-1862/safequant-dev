"""Backward-compat shim."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/localization/depth_bands.py"),
        run_name="__main__",
    )
