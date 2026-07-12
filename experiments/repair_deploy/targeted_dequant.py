"""Discussion: targeted 4-bit dequant repair (stub until experiment runs)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.manifest import RunConfig

_SCRIPT = Path(__file__)


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--dry_run", action="store_true", help="print config and exit without GPU work")
    args = ap.parse_args()

    # Create the run config
    cfg = RunConfig(model=args.model, quantizer=args.quantizer)

    # Print the config and exit if dry run is enabled
    if args.dry_run:
        print(f"targeted_dequant stub: model={cfg.model} quantizer={cfg.quantizer}")
        print("No JSON written — run full implementation when ready.")
        return

    raise NotImplementedError(
        "targeted_dequant is planned; produce a JSON in outputs/repair_deploy/ when implemented."
    )


if __name__ == "__main__":
    main()
