#!/usr/bin/env python
"""Make AutoGPTQ-format GPTQ checkpoints loadable by transformers/gptqmodel.

AutoGPTQ writes weights as gptq_model-<bits>bit-<group>g.safetensors and the
quant settings in quantize_config.json. transformers' AutoModelForCausalLM
instead looks for model.safetensors + a `quantization_config` block inside
config.json, so it fails with "no model.safetensors found in directory". This
patch, per checkpoint dir:
  1) symlinks gptq_model-*.safetensors -> model.safetensors (weights findable),
  2) injects quantization_config into config.json from quantize_config.json
     (so transformers routes the load through the GPTQ/optimum path).
Idempotent -- safe to re-run; skips anything already correct.

  python scripts/fix_autogptq_checkpoints.py                # all gptq/ dirs
  python scripts/fix_autogptq_checkpoints.py <dir> [<dir>]  # specific dirs
"""
import glob
import json
import os
import sys
from pathlib import Path

DEFAULT_GLOB = "/ocean/projects/cis240116p/bhoomire/quantized_models/gptq/*-gptq*bit-g*"


def fix(d):
    d = Path(d)
    cfgp, qcfgp = d / "config.json", d / "quantize_config.json"
    if not cfgp.exists() or not qcfgp.exists():
        print(f"  skip {d.name} (missing config.json or quantize_config.json)")
        return

    # 1) weights findable as model.safetensors
    if not (d / "model.safetensors").exists():
        st = sorted(glob.glob(str(d / "gptq_model-*.safetensors")))
        if st:
            os.symlink(Path(st[0]).name, d / "model.safetensors")
            print(f"  {d.name}: linked {Path(st[0]).name} -> model.safetensors")
        else:
            print(f"  {d.name}: WARN no gptq_model-*.safetensors found")

    # 2) config.json declares the GPTQ quantization_config
    cfg = json.loads(cfgp.read_text())
    if "quantization_config" not in cfg:
        q = json.loads(qcfgp.read_text())
        cfg["quantization_config"] = {
            "quant_method": "gptq",
            "bits": q.get("bits"),
            "group_size": q.get("group_size"),
            "desc_act": q.get("desc_act"),
            "sym": q.get("sym", True),
            "damp_percent": q.get("damp_percent", 0.1),
            "true_sequential": q.get("true_sequential", True),
            "checkpoint_format": q.get("checkpoint_format", "gptq"),
        }
        cfgp.write_text(json.dumps(cfg, indent=2))
        print(f"  {d.name}: added quantization_config (bits={q.get('bits')}, "
              f"desc_act={q.get('desc_act')})")
    else:
        print(f"  {d.name}: quantization_config already present")


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(glob.glob(DEFAULT_GLOB))
    if not targets:
        print("no checkpoint dirs found")
        sys.exit(1)
    for t in targets:
        fix(t)
    print("done.")
