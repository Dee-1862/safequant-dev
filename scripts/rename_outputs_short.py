#!/usr/bin/env python3
"""Rename mirrored outputs from long model ids / script stems to short spec names."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

MODEL_LONG_TO_SHORT = {
    "llama7b": "llama",
    "mistral7b-v02": "mistral",
    "qwen7b": "qwen",
}

STEM_RENAMES = {
    "headline_eval": "headline",
    "probe_accuracy_mlp": "probe_accuracy_mlp",  # keep mlp in stem position
}

# probe_refit_control_llama7b.json -> probe_refit_control_llama.json (drop quantizer if present)
REFIT_RE = re.compile(r"^probe_refit_control_(.+?)(?:_fp32)?\.json$")


def new_name(old: str) -> str | None:
    if not old.endswith(".json") or old.endswith("_raw.json"):
        return None
    stem = old[:-5]
    m = REFIT_RE.match(old)
    if m:
        model = m.group(1)
        short = MODEL_LONG_TO_SHORT.get(model, model)
        return f"probe_refit_control_{short}.json"

    if stem.startswith("probe_accuracy_mlp_"):
        rest = stem[len("probe_accuracy_mlp_") :]
        for long, short in MODEL_LONG_TO_SHORT.items():
            if rest.startswith(long + "_"):
                return f"probe_accuracy_{short}_" + rest[len(long) + 1 :] + "_mlp.json"
        return None

    if stem.startswith("headline_eval_"):
        rest = stem[len("headline_eval_") :]
        for long, short in MODEL_LONG_TO_SHORT.items():
            if rest.startswith(long + "_"):
                return f"headline_{short}_" + rest[len(long) + 1 :] + ".json"
        return None

    for long, short in MODEL_LONG_TO_SHORT.items():
        token = f"_{long}_"
        if token in stem:
            return stem.replace(token, f"_{short}_") + ".json"
    return None


def main():
    renamed = 0
    for path in sorted(OUT.rglob("*.json")):
        if "_raw" in path.stem and path.name.endswith("_raw.json"):
            # headline_eval_llama7b_aqlm2_raw.json -> headline_llama_aqlm2_raw.json
            target_name = new_name(path.name.replace("_raw", ""))
            if target_name:
                target_name = target_name.replace(".json", "_raw.json")
            else:
                continue
        else:
            target_name = new_name(path.name)
        if not target_name or target_name == path.name:
            continue
        dst = path.parent / target_name
        if dst.exists():
            continue
        path.rename(dst)
        renamed += 1
        print(f"  {path.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    print(f"Renamed {renamed} files.")


if __name__ == "__main__":
    main()
