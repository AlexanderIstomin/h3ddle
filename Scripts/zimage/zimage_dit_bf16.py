#!/usr/bin/env python3
"""Rewrite the base DiT into the package's naming at bf16, unquantised.

The C harness reads one file whichever weights it is given, so having an
unquantised copy in the same layout lets the full forward be gated twice from
the same code: once against bf16, where any discrepancy is a plumbing fault,
and once against the shipped int8, where the discrepancy is quantisation
compounding over thirty layers. Conflating those two would leave neither
measured.

  zimage_dit_bf16.py --model <base checkout> --out dit-bf16.safetensors
"""

import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# diffusers splits the projections and suffixes the norms differently from the
# original Z-Image checkpoint the int8 repack follows. The engine reads the
# latter, so that is what the port is written against.
RENAME = {
    "attention.to_out.0.weight": "attention.out.weight",
    "attention.norm_q.weight": "attention.q_norm.weight",
    "attention.norm_k.weight": "attention.k_norm.weight",
}
TOP_LEVEL = {
    "all_x_embedder.2-1.weight": "x_embedder.weight",
    "all_x_embedder.2-1.bias": "x_embedder.bias",
    "all_final_layer.2-1.linear.weight": "final_layer.linear.weight",
    "all_final_layer.2-1.linear.bias": "final_layer.linear.bias",
    "all_final_layer.2-1.adaLN_modulation.1.weight": "final_layer.adaLN_modulation.1.weight",
    "all_final_layer.2-1.adaLN_modulation.1.bias": "final_layer.adaLN_modulation.1.bias",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = f"{args.model}/transformer"
    index = json.load(open(f"{root}/diffusion_pytorch_model.safetensors.index.json"))
    index = index["weight_map"]
    handles = {}

    def get(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{root}/{shard}", framework="pt")
        return handles[shard].get_tensor(key)

    out = {}
    prefixes = ([f"layers.{i}." for i in range(30)] +
                [f"noise_refiner.{i}." for i in range(2)] +
                [f"context_refiner.{i}." for i in range(2)])
    for prefix in prefixes:
        # q, k, v in that order — established from the shipped package by row
        # norms, which a rotation along the input dim leaves untouched.
        out[f"{prefix}attention.qkv.weight"] = torch.cat(
            [get(f"{prefix}attention.to_{n}.weight") for n in "qkv"], dim=0)
        for key in index:
            if not key.startswith(prefix):
                continue
            leaf = key[len(prefix):]
            if leaf.startswith("attention.to_") and leaf != "attention.to_out.0.weight":
                continue
            out[prefix + RENAME.get(leaf, leaf)] = get(key)

    for source, target in TOP_LEVEL.items():
        out[target] = get(source)
    for key in ("cap_embedder.0.weight", "cap_embedder.1.weight", "cap_embedder.1.bias",
                "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias",
                "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
                "x_pad_token", "cap_pad_token"):
        out[key] = get(key)

    # The checkpoint stores f32 whose values are already bf16-exact, so this
    # narrows the storage without touching a value.
    total = 0
    for key, value in out.items():
        assert value.dtype in (torch.float32, torch.bfloat16), (key, value.dtype)
        narrowed = value.to(torch.bfloat16)
        assert torch.equal(narrowed.float(), value.float()), f"{key} is not bf16-exact"
        out[key] = narrowed.contiguous()
        total += narrowed.numel()

    save_file(out, args.out)
    print(f"wrote {args.out}: {len(out)} tensors, {total / 1e9:.2f}G parameters")


if __name__ == "__main__":
    main()
