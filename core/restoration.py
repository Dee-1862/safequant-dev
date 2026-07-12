"""FP32 weight extraction and projection restoration (read/write split)."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.evaluation import _logprob_score
from core.model_loader import get_weight
from core.residuals import get_inner

WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]
READ_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"]
PHI3_READ_PROJS = ["self_attn.qkv_proj", "mlp.gate_up_proj"]
PHI3_WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]


def get_proj_sets(model):
    """
    Return (read_projs, write_projs) matching the model projection layout.

    Input:
    - model (torch.nn.Module): the model to get the projection sets from

    Output:
    - READ_PROJS (list): the read projection sets
    - WRITE_PROJS (list): the write projection sets
    """
    # Get the submodules of the model
    submods = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}

    # Check if the model is a Phi3 model and return the appropriate projection sets
    if "qkv_proj" in submods or "gate_up_proj" in submods:
        return list(PHI3_READ_PROJS), list(PHI3_WRITE_PROJS)
    return list(READ_PROJS), list(WRITE_PROJS)


def _get_submodule(layer, proj_path):
    """
    Get the submodule of the layer.

    Input:
    - layer (torch.nn.Module): the layer to get the submodule from
    - proj_path (str): the path to the submodule

    Output:
    - m (torch.nn.Module): the submodule
    """
    m = layer
    for part in proj_path.split("."):
        m = getattr(m, part)
    return m


def _set_submodule(layer, proj_path, new_module):
    """
    Set the submodule of the layer.

    Input:
    - layer (torch.nn.Module): the layer to set the submodule of
    - proj_path (str): the path to the submodule
    - new_module (torch.nn.Module): the new submodule
    """
    # Get the parts of the project path
    parts = proj_path.split(".")
    parent = layer

    # Set the submodule of the layer
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _is_awq_module(module):
    """
    Check if the module is an AWQ module.

    Input:
    - module (torch.nn.Module): the module to check if it is an AWQ module

    Output:
    - bool: True if the module is an AWQ module, False otherwise
    """
    return "WQLinear" in type(module).__name__


def _transform_fp32_to_awq_space(fp32_weight, awq_module):
    """
    Transform the FP32 weight to the AWQ space.

    Input:
    - fp32_weight (torch.Tensor): the FP32 weight to transform
    - awq_module (torch.nn.Module): the AWQ module to transform the weight to

    Output:
    - torch.Tensor: the transformed weight
    """
    # Get the scales and the group size
    scales = awq_module.scales.float()
    group_size = awq_module.group_size
    scales_expanded = scales.repeat_interleave(group_size, dim = 0)
    return fp32_weight.float().to(scales_expanded.device) * scales_expanded.T


def extract_fp32_weights(fp32_model, proj_paths):
    """
    Extract the FP32 weights from the model.

    Input:
    - fp32_model (torch.nn.Module): the FP32 model to extract the weights from
    - proj_paths (list): the paths to the projection sets

    Output:
    - weights (dict): a dictionary of the weights for each layer and projection set
    """
    # Get the number of layers
    n_layers = fp32_model.config.num_hidden_layers

    # Extract the weights for each layer and projection set
    weights = {}
    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer = fp32_model.model.layers[layer_idx]
            weights[layer_idx] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[layer_idx][proj] = (
                    get_weight(m, as_dtype=torch.float32).cpu().clone()
                )
    return weights


def restore_projections(model, fp32_weights, proj_paths, layer_range = None):
    """
    Restore projections in the model, optionally limited to a layer range.

    Input:
    - model (torch.nn.Module): the model to restore the projections in
    - fp32_weights (dict): a dictionary of the weights for each layer and projection set
    - proj_paths (list): the paths to the projection sets
    - layer_range (tuple | None): optional [lo, hi) layer indices and restores all layers if None
    """

    # Get the inner model, the activation dtype, the device, and the number of layers
    inner = get_inner(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    if layer_range is None:
        lo = 0
        hi = n_layers
    else:
        lo, hi = layer_range

    # Restore the projections for each layer and projection set

    for layer_idx in range(lo, hi):
        layer = inner.layers[layer_idx]
        for proj in proj_paths:

            # Get the original module, the input features, the output features, and the bias
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, "in_features", None) or orig_module.infeatures
            out_f = getattr(orig_module, "out_features", None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[layer_idx][proj]

            # Transform the weight to the AWQ space if the module is an AWQ module
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)
            new_linear = nn.Linear(in_f, out_f, bias = has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(w.to(device).to(act_dtype), requires_grad = False)

            # Set the bias if the module has a bias
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(), requires_grad = False)
            _set_submodule(layer, proj, new_linear)


def measure_arc(model, tokenizer, arc_items):
    """
    Measure the accuracy of the model on the ARC items.

    Input:
    - model (torch.nn.Module): the model to measure the accuracy of
    - tokenizer (Tokenizer): the tokenizer to use
    - arc_items (list): the ARC items to measure the accuracy of

    Output:
    - accur (float): the accuracy of the model on the ARC items
    """
    # Get the device and the number of correct predictions
    device = next(model.parameters()).device
    correct = 0

    # Measure the accuracy of the model on the ARC items
    for item in arc_items:
        # Get the prompt and the prediction
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)

        # Check if the prediction is correct
        if pred == item["answer_idx"]:
            correct += 1

    # Calculate the accuracy
    accur = 100.0 * correct / len(arc_items)
    return accur


class Restorer:
    """
    Caches FP32 weights and restores read/write projections.

    Input:
    - fp32_model (torch.nn.Module): the FP32 model to cache the weights from
    """

    # Initialize the Restorer class
    def __init__(self, fp32_model):
        self.read_projs, self.write_projs = get_proj_sets(fp32_model)
        self.write_weights = extract_fp32_weights(fp32_model, self.write_projs)
        self.read_weights = extract_fp32_weights(fp32_model, self.read_projs)

    # Restore the write projections
    def restore_write(self, model):
        restore_projections(model, self.write_weights, self.write_projs)

    # Restore the read projections
    def restore_read(self, model):
        restore_projections(model, self.read_weights, self.read_projs)


    # Restore both the write and read projections
    def restore_both(self, model):
        self.restore_write(model)
        self.restore_read(model)
