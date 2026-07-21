# Dump key numbers from every completed experiment JSON, grouped by folder.
import glob
import json
import sys
from pathlib import Path

# Top-level fields worth surfacing across the different experiment schemas.
INTERESTING = [
    "fp32_asr", "quantized_asr", "damage_pp", "swap_valid", "swap_delta_pp",
    "asr_per_condition", "arc_per_condition",
    "recovery_pp", "recovery_pp_read", "recovery_pp_write", "recovery_pp_both",
    "recovery_pp_per_range", "recovery_pp_per_projection", "swap_asr",
    "asr", "arc", "xstest", "mcnemar", "recovery_ci_95", "flags_by_cond",
    "baseline_asr", "ablated_asr", "delta_asr_pp",
    "selected_layer", "selected_pos", "operative",
    "recovery_pp_4bit", "recovery_pp_fp32", "frac_of_fp32_recovery",
    "role", "layer_range", "source_quant",
    "mean_codes_changed", "fp_lora", "folded",
]


def main():
    # Optional filters: substrings a filename must contain (e.g. model tokens
    # qwen / mistral / phi3, or an experiment name). No args = dump everything.
    filters = sys.argv[1:]
    files = sorted(
        f for f in glob.glob("outputs/**/*.json", recursive=True)
        if not f.endswith("_raw.json")
        and (not filters or any(x in Path(f).name for x in filters))
    )
    if not files:
        where = f" matching {filters}" if filters else ""
        print(f"no result JSONs under outputs/{where} (are you at the repo root?)")
        return
    if filters:
        print(f"# filtered to: {filters}\n")

    by_dir = {}
    for f in files:
        by_dir.setdefault(str(Path(f).parent), []).append(f)

    for d in sorted(by_dir):
        print("\n" + "=" * 74)
        print(f"## {d}   ({len(by_dir[d])} files)")
        print("=" * 74)
        for f in by_dir[d]:
            try:
                js = json.load(open(f))
            except Exception as e:  # noqa: BLE001
                print(f"  • {Path(f).name}: UNREADABLE ({e})")
                continue
            print(f"\n  • {Path(f).name}")
            if isinstance(js.get("summary"), dict):
                print(f"      summary: {json.dumps(js['summary'], default=str)}")
            top = {k: js[k] for k in INTERESTING if k in js}
            if top:
                print(f"      {json.dumps(top, default=str)}")


if __name__ == "__main__":
    main()
