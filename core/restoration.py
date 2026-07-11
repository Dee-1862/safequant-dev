"""FP32 weight extraction and projection restoration (read/write split)."""
from __future__ import annotations

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
    submods = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if "qkv_proj" in submods or "gate_up_proj" in submods:
        return list(PHI3_READ_PROJS), list(PHI3_WRITE_PROJS)
    return list(READ_PROJS), list(WRITE_PROJS)


def _get_submodule(layer, proj_path):
    m = layer
    for part in proj_path.split("."):
        m = getattr(m, part)
    return m


def _set_submodule(layer, proj_path, new_module):
    parts = proj_path.split(".")
    parent = layer
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _is_awq_module(module):
    return "WQLinear" in type(module).__name__


def _transform_fp32_to_awq_space(fp32_weight, awq_module):
    scales = awq_module.scales.float()
    group_size = awq_module.group_size
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    return fp32_weight.float().to(scales_expanded.device) * scales_expanded.T


def extract_fp32_weights(fp32_model, proj_paths):
    n_layers = fp32_model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer = fp32_model.model.layers[layer_idx]
            weights[layer_idx] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[layer_idx][proj] = get_weight(m, as_dtype=torch.float32).cpu().clone()
    return weights


def restore_projections(model, fp32_weights, proj_paths):
    inner = get_inner(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    for layer_idx in range(n_layers):
        layer = inner.layers[layer_idx]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, "in_features", None) or orig_module.infeatures
            out_f = getattr(orig_module, "out_features", None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[layer_idx][proj]
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)
            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(
                w.to(device).to(act_dtype), requires_grad=False
            )
            if has_bias:
                new_linear.bias = nn.Parameter(
                    orig_module.bias.data.clone(), requires_grad=False
                )
            _set_submodule(layer, proj, new_linear)


def restore_projections_in_range(model, fp32_weights, proj_paths, layer_range):
    """Restore only layers in [layer_range[0], layer_range[1])."""
    inner = get_inner(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    lo, hi = layer_range
    for layer_idx in range(lo, hi):
        layer = inner.layers[layer_idx]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, "in_features", None) or orig_module.infeatures
            out_f = getattr(orig_module, "out_features", None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[layer_idx][proj]
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)
            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(
                w.to(device).to(act_dtype), requires_grad=False
            )
            if has_bias:
                new_linear.bias = nn.Parameter(
                    orig_module.bias.data.clone(), requires_grad=False
                )
            _set_submodule(layer, proj, new_linear)


def measure_arc(model, tokenizer, arc_items):
    device = next(model.parameters()).device
    correct = 0
    for item in arc_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)
        if pred == item["answer_idx"]:
            correct += 1
    return 100.0 * correct / len(arc_items)


class Restorer:
    """Caches FP32 weights and restores read/write projections."""

    def __init__(self, fp32_model):
        self.read_projs, self.write_projs = get_proj_sets(fp32_model)
        self.write_weights = extract_fp32_weights(fp32_model, self.write_projs)
        self.read_weights = extract_fp32_weights(fp32_model, self.read_projs)

    def restore_write(self, model):
        restore_projections(model, self.write_weights, self.write_projs)

    def restore_read(self, model):
        restore_projections(model, self.read_weights, self.read_projs)

    def restore_both(self, model):
        self.restore_write(model)
        self.restore_read(model)
