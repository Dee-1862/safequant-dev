#!/usr/bin/env python
"""run_arditi_official.py -- Shim script to run the official Arditi et al. method.

This uses the actual functions from the `refusal_direction` repository downloaded
by the agent to execute their position-by-layer search and generation-based eval,
but feeds it models loaded by our quantized model loader and our datasets.
"""
import argparse
import sys
import torch
import os
import json
from pathlib import Path

# Add project root and refusal_direction repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "refusal_direction"))

from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts

from pipeline.model_utils.llama2_model import Llama2Model
from pipeline.model_utils.llama3_model import Llama3Model
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction

# --- Custom ModelBase Wrapper ------------------------------------------------

class SafeQuantLlama2(Llama2Model):
    def __init__(self, model_name_or_path, quantizer):
        self.quantizer = quantizer
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.float16):
        if self.quantizer == "fp32":
            model, _ = load_fp32(model_path)
        else:
            model, _ = load_quantized(model_path, quantizer=self.quantizer)
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        from core.model_loader import load_fp32
        _, tok = load_fp32(model_path, only_tokenizer=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

class SafeQuantLlama3(Llama3Model):
    def __init__(self, model_name_or_path, quantizer):
        self.quantizer = quantizer
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.bfloat16):
        if self.quantizer == "fp32":
            model, _ = load_fp32(model_path)
        else:
            model, _ = load_quantized(model_path, quantizer=self.quantizer)
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        from core.model_loader import load_fp32
        _, tok = load_fp32(model_path, only_tokenizer=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

def construct_model_base(model_name_or_path, quantizer):
    if "llama-3" in model_name_or_path.lower():
        return SafeQuantLlama3(model_name_or_path, quantizer)
    return SafeQuantLlama2(model_name_or_path, quantizer)

# --- Entry Point ------------------------------------------------------------

def run_pipeline(model_path, quantizer, n_train=128, n_val=32):
    model_base = construct_model_base(model_path, quantizer)
    
    # Load safequant datasets as instructions
    # the official repo expects simple strings for get_refusal_scores and get_mean_diff
    h_all = load_wildjailbreak_prompts(n_train + n_val, config="train")
    harmful_train = [p["prompt"] for p in h_all[:n_train]]
    harmful_val   = [p["prompt"] for p in h_all[n_train:n_train+n_val]]

    hl_all = load_harmless_prompts(n_train + n_val)
    harmless_train = [str(p) for p in hl_all[:n_train]]
    harmless_val   = [str(p) for p in hl_all[n_train:n_train+n_val]]

    artifact_dir = f"outputs/arditi_{quantizer}"
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"[{model_path}/{quantizer}] Generating candidate refusal directions...")
    candidate_directions = generate_directions(
        model_base, 
        harmful_train, 
        harmless_train, 
        artifact_dir=artifact_dir
    )

    print(f"[{model_path}/{quantizer}] Selecting optimal refusal direction...")
    pos, layer, direction = select_direction(
        model_base, 
        harmful_val, 
        harmless_val, 
        candidate_directions, 
        artifact_dir=artifact_dir
    )

    print(f"Done! Selected optimal direction at layer={layer}, position={pos}")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="fp32")
    args = ap.parse_args()
    
    run_pipeline(args.model, args.quantizer)
