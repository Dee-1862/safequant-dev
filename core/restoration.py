# FP32 weight extraction and projection restoration (read/write split).

#Front Matter
import torch
import torch.nn as nn

from core.evaluation import _logprob_score
from core.model_loader import get_weight
from core.residuals import get_inner

WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]
READ_PROJS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]
PHI3_READ_PROJS = ["self_attn.qkv_proj", "mlp.gate_up_proj"]
PHI3_WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]


def get_proj_sets(model):
    """Return (read_projs, write_projs) matching the model projection layout."""
    # Inspect leaf submodule names for Phi-3 fused layouts
    submods = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if "qkv_proj" in submods or "gate_up_proj" in submods:
        return list(PHI3_READ_PROJS), list(PHI3_WRITE_PROJS)
    return list(READ_PROJS), list(WRITE_PROJS)


def _get_submodule(layer, proj_path):
    """Resolve a dotted projection path under a transformer layer."""
    # Walk attribute path
    m = layer
    for part in proj_path.split("."):
        m = getattr(m, part)
    return m


def _set_submodule(layer, proj_path, new_module):
    """Replace a dotted projection path under a transformer layer."""
    # Walk to parent, then setattr the leaf
    parts = proj_path.split(".")
    parent = layer
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _is_awq_module(module):
    """Detect AWQ WQLinear modules by class name."""
    return "WQLinear" in type(module).__name__


def _transform_fp32_to_awq_space(fp32_weight, awq_module):
    """Rescale an FP32 weight into the AWQ activation/weight scale space."""
    # Expand group scales along the input feature axis
    scales = awq_module.scales.float()
    group_size = awq_module.group_size
    scales_expanded = scales.repeat_interleave(group_size, dim = 0)
    return fp32_weight.float().to(scales_expanded.device) * scales_expanded.T


def extract_fp32_weights(fp32_model, proj_paths):
    """Snapshot FP32 projection weights for every layer and path."""
    # Iterate all hidden layers
    n_layers = fp32_model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer = fp32_model.model.layers[layer_idx]
            weights[layer_idx] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[layer_idx][proj] = (
                    get_weight(m, as_dtype = torch.float32).cpu().clone()
                )
    return weights


def restore_projections(model, fp32_weights, proj_paths, layer_range = None):
    """Restore projections in the model, optionally limited to a layer range."""
    # Resolve inner layers and activation device/dtype
    inner = get_inner(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    if layer_range is None:
        lo = 0
        hi = n_layers
    else:
        lo, hi = layer_range

    # Swap each selected proj for an nn.Linear with restored weights
    for layer_idx in range(lo, hi):
        layer = inner.layers[layer_idx]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, "in_features", None) or orig_module.infeatures
            out_f = getattr(orig_module, "out_features", None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[layer_idx][proj]

            # AWQ modules need weights transformed into scale space
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)
            new_linear = nn.Linear(in_f, out_f, bias = has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(
                w.to(device).to(act_dtype), requires_grad = False
            )

            if has_bias:
                new_linear.bias = nn.Parameter(
                    orig_module.bias.data.clone(), requires_grad = False
                )
            _set_submodule(layer, proj, new_linear)


def measure_arc(model, tokenizer, arc_items):
    """Measure multiple-choice accuracy of the model on ARC items."""
    # Score each item by summed choice log-probs
    device = next(model.parameters()).device
    correct = 0
    for item in arc_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)
        if pred == item["answer_idx"]:
            correct += 1
    accur = 100.0 * correct / len(arc_items)
    return accur


class Restorer:
    def __init__(self, fp32_model):
        """Cache FP32 read/write projection weights from a dense model."""
        # Snapshot both proj sets once for later restores
        self.read_projs, self.write_projs = get_proj_sets(fp32_model)
        self.write_weights = extract_fp32_weights(fp32_model, self.write_projs)
        self.read_weights = extract_fp32_weights(fp32_model, self.read_projs)

    def restore_write(self, model):
        """Restore write projections on a (possibly quantized) model."""
        restore_projections(model, self.write_weights, self.write_projs)

    def restore_read(self, model):
        """Restore read projections on a (possibly quantized) model."""
        restore_projections(model, self.read_weights, self.read_projs)

    def restore_both(self, model):
        """Restore write then read projections on a model."""
        self.restore_write(model)
        self.restore_read(model)
