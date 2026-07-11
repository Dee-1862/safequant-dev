"""Regression anchors: frozen metrics from published runs.

Compares live output JSONs (or core function outputs after refactor) against
tests/anchors/frozen_metrics.json within float tolerance. Run with:
  pytest tests/test_anchors.py -v
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANCHORS = json.loads((Path(__file__).parent / "anchors" / "frozen_metrics.json").read_text())
TOL = 1e-6


def _load_output(rel_path: str) -> dict:
    path = ROOT / rel_path
    if not path.exists():
        # After output mirror migration, try claim-folder layout.
        name = Path(rel_path).name
        for candidate in ROOT.glob(f"outputs/**/{name}"):
            return json.loads(candidate.read_text())
        pytest.skip(f"anchor source missing: {rel_path}")
    return json.loads(path.read_text())


def test_direction_drift_cos_sim_median():
    anchor = ANCHORS["direction_drift"]
    data = _load_output(anchor["source"])
    got = data["summary"]["cos_sim_median"]
    assert abs(got - anchor["cos_sim_median"]) < TOL, (
        f"cos_sim_median drift: got {got}, expected {anchor['cos_sim_median']}"
    )


def test_probe_accuracy_min_acc():
    anchor = ANCHORS["probe_accuracy"]
    data = _load_output(anchor["source"])
    got_q = data["summary"]["quant_min_acc"]
    got_fp = data["summary"]["fp32_min_acc"]
    assert abs(got_q - anchor["quant_min_acc"]) < TOL
    assert abs(got_fp - anchor["fp32_min_acc"]) < TOL


def test_core_directions_compute_r_hat_stable():
    """Smoke test that core directions module imports and runs."""
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location(
        "directions", ROOT / "core" / "directions.py"
    )
    directions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(directions)
    compute_r_hat = directions.compute_r_hat
    per_layer_separation = directions.per_layer_separation

    h = [torch.randn(64) for _ in range(10)]
    s = [torch.randn(64) for _ in range(10)]
    r = compute_r_hat(h, s)
    sep, hm, sm = per_layer_separation(h, s, r)
    assert isinstance(sep, float)
    assert hm != sm or sep == 0.0


def test_manifest_result_path():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "manifest", ROOT / "core" / "manifest.py"
    )
    manifest = importlib.util.module_from_spec(spec)
    sys.modules["manifest"] = manifest
    spec.loader.exec_module(manifest)
    RunConfig = manifest.RunConfig
    result_path = manifest.result_path

    cfg = RunConfig(model="llama7b", quantizer="aqlm2", variant="read")
    p = result_path(Path("experiments/localization/depth_bands.py"), cfg)
    assert p == Path("outputs/localization/depth_bands_llama_aqlm2_read.json")

    cfg_headline = RunConfig(model="llama7b", quantizer="aqlm2", output_stem="headline")
    assert result_path(Path("experiments/headline/headline_eval.py"), cfg_headline) == (
        Path("outputs/headline/headline_llama_aqlm2.json")
    )

    cfg_refit = RunConfig(model="llama7b", omit_quantizer=True)
    assert result_path(Path("experiments/boundary/probe_refit_control.py"), cfg_refit) == (
        Path("outputs/boundary/probe_refit_control_llama.json")
    )
