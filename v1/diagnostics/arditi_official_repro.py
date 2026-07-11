"""Backward-compat shim."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "experiments/behavioral_direction/arditi_repro.py"),
        run_name="__main__",
    )
