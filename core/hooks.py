"""Residual-stream forward hooks (directional ablation), factored out of the archived
4-condition directional_ablation.py so the keeper probe_causal_ablation.py depends on core/,
not on an attic'd module. This is the Phase-0 seed of core/hooks.py; Phase 1 adds the
`HookContext` / `ablate(...)` context manager (see PHASE1_CONSOLIDATION.md) so handle removal
is guaranteed by `with`.

Projects a (pre-normalized) direction out of the residual stream at each listed layer:
    h <- h - (h . r) r
"""
import sys
from pathlib import Path

# TEMPORARY backward import: _get_inner still lives in diagnostics/layer_routing.py.
# Phase 1 moves the single canonical _get_inner into core/residuals.py; then this import
# becomes `from core.residuals import _get_inner` and the sys.path hack goes away.
sys.path.insert(0, str(Path(__file__).parent.parent))
from diagnostics.layer_routing import _get_inner  # noqa: E402


def make_ablation_hook(r_hat_cpu):
    """Forward hook that projects r_hat out of the residual stream output
    (h <- h - (h @ r) * r). r_hat_cpu must be pre-normalized; it is moved to the
    activation's device/dtype on the fly."""
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if h.dim() != 3:
            return output
        r = r_hat_cpu.to(h.device).to(h.dtype)
        proj = (h @ r).unsqueeze(-1)
        h_new = h - proj * r
        if is_tuple:
            return (h_new,) + output[1:]
        return h_new
    return hook


def install_ablation(model, r_hat_cpu, layers):
    inner = _get_inner(model)
    handles = []
    for L in layers:
        handles.append(inner.layers[L].register_forward_hook(
            make_ablation_hook(r_hat_cpu)))
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()
