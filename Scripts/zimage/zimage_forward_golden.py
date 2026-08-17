#!/usr/bin/env python3
"""Export a whole S3-DiT forward, for the C port to match.

Stage 4 established that one layer is right. What this adds is the plumbing
around the layers, which is where a model that computes every block correctly
still produces the wrong picture:

  - the refiners do *not* run on the unified sequence. `noise_refiner` sees
    image tokens alone and `context_refiner` sees caption tokens alone, each
    attending only within its own half, and only afterwards are the two
    concatenated for the 30-layer trunk. Running them on the joint sequence is
    numerically plausible and wrong;
  - `context_refiner` takes no conditioning at all, while `noise_refiner` and
    the trunk share one `adaln_input`;
  - the head's output is unpatchified from the *image* tokens only, the
    caption tail being dropped.

One block is instantiated and its weights are swapped per layer, rather than
loading the model, which would want ~25 GB at f32. It costs a re-read of the
checkpoint per layer and keeps the peak at one block.

  zimage_forward_golden.py --model <base checkout> --out golden.safetensors
"""

import argparse
import json
from types import SimpleNamespace

import torch
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import (
    FinalLayer,
    RopeEmbedder,
    TimestepEmbedder,
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from safetensors import safe_open
from safetensors.torch import save_file

DIM = 3840
HEADS = 30
NORM_EPS = 1e-5
CAP_DIM = 2560
PATCH = 2
LATENT_CHANNELS = 16
SEQ_MULTI_OF = 32
LAYERS = 30
REFINERS = 2

LATENT_SIDE = 32
CAPTION_TOKENS = 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timestep", type=float, default=0.375)
    parser.add_argument("--package", help="read the shipped int8 DiT instead of "
                                          "the base checkpoint, dequantised and "
                                          "un-rotated, to price quantisation")
    args = parser.parse_args()

    root = f"{args.model}/transformer"
    index = json.load(open(f"{root}/diffusion_pytorch_model.safetensors.index.json"))
    index = index["weight_map"]
    handles = {}

    def get_base(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{root}/{shard}", framework="pt")
        return handles[shard].get_tensor(key).float()

    if not args.package:
        get = get_base
    else:
        package = safe_open(args.package, framework="pt")

        def convrot(v, group=256):
            """The engine's h3_convrot kernel: radix-4 over each group of 256
            with the regular Hadamard butterfly, scaled 1/16. Symmetric and
            orthogonal, so it is its own inverse and un-rotates by reapplying."""
            shape = v.shape
            x = v.reshape(-1, shape[-1] // group, group).clone()
            stride = 1
            for _ in range(4):
                y = x.reshape(x.shape[0], x.shape[1], group // (4 * stride), 4, stride)
                a, b, c, d = y[..., 0, :], y[..., 1, :], y[..., 2, :], y[..., 3, :]
                x = torch.stack([a + b + c - d, a + b - c + d,
                                 a - b + c + d, -a + b + c + d], dim=-2).reshape(x.shape)
                stride *= 4
            return (x * 0.0625).reshape(shape)

        # The package follows the original Z-Image naming; diffusers splits the
        # projections and suffixes the norms the other way round.
        RENAME = {"attention.to_out.0.weight": "attention.out.weight",
                  "attention.norm_q.weight": "attention.q_norm.weight",
                  "attention.norm_k.weight": "attention.k_norm.weight",
                  "all_x_embedder.2-1.weight": "x_embedder.weight",
                  "all_x_embedder.2-1.bias": "x_embedder.bias"}
        for leaf in ("linear.weight", "linear.bias",
                     "adaLN_modulation.1.weight", "adaLN_modulation.1.bias"):
            RENAME[f"all_final_layer.2-1.{leaf}"] = f"final_layer.{leaf}"

        def fetch(name):
            t = package.get_tensor(name)
            if t.dtype != torch.int8:
                return t.float()
            # Weight-only int8: dequantise by the per-row scale, then undo the
            # rotation. Equivalent to rotating the activation instead, exactly,
            # because the rotation is orthogonal and symmetric.
            return convrot(t.float() * package.get_tensor(name[:-6] + "weight_scale").float())

        def get(key):
            for part, index_in_fused in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
                if key.endswith(f"attention.{part}.weight"):
                    fused = fetch(key[:-len(f"{part}.weight")] + "qkv.weight")
                    return fused[index_in_fused * DIM:(index_in_fused + 1) * DIM]
            for suffix, replacement in RENAME.items():
                if key.endswith(suffix):
                    return fetch(key[:len(key) - len(suffix)] + replacement)
            return fetch(key)

    def build(module, prefix, mapping):
        module.load_state_dict({k: get(prefix + v) for k, v in mapping.items()},
                               strict=True)
        return module.eval()

    torch.manual_seed(11)

    cap_padded = CAPTION_TOKENS + (-CAPTION_TOKENS) % SEQ_MULTI_OF
    tokens_side = LATENT_SIDE // PATCH
    image_tokens = tokens_side * tokens_side
    assert image_tokens % SEQ_MULTI_OF == 0

    cap_ids = torch.tensor([[1 + i, 0, 0] for i in range(cap_padded)], dtype=torch.int32)
    img_ids = torch.tensor(
        [[cap_padded + 1, h, w] for h in range(tokens_side) for w in range(tokens_side)],
        dtype=torch.int32,
    )
    rope = RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])
    img_freqs = rope(img_ids).unsqueeze(0)
    cap_freqs = rope(cap_ids).unsqueeze(0)
    freqs = torch.cat([img_freqs, cap_freqs], dim=1)

    # ---- conditioning and embedding ------------------------------------
    t_embedder = build(TimestepEmbedder(256, mid_size=1024), "t_embedder.",
                       {f"mlp.{i}.{p}": f"mlp.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")})
    timestep = torch.tensor([args.timestep])
    with torch.no_grad():
        adaln_input = t_embedder(timestep * 1000.0)

    x_embedder = build(torch.nn.Linear(PATCH * PATCH * LATENT_CHANNELS, DIM),
                       "all_x_embedder.2-1.", {"weight": "weight", "bias": "bias"})
    cap_embedder = build(
        torch.nn.Sequential(RMSNorm(CAP_DIM, eps=NORM_EPS), torch.nn.Linear(CAP_DIM, DIM)),
        "cap_embedder.", {"0.weight": "0.weight", "1.weight": "1.weight", "1.bias": "1.bias"})
    cap_pad_token = get("cap_pad_token")

    latent = torch.randn(LATENT_CHANNELS, 1, LATENT_SIDE, LATENT_SIDE)
    patches, _, _ = ZImageTransformer2DModel._patchify_image(None, latent, PATCH, 1)
    caption = torch.randn(CAPTION_TOKENS, CAP_DIM)

    with torch.no_grad():
        image = x_embedder(patches).unsqueeze(0)
        cap = cap_embedder(caption)
        cap = torch.cat([cap, cap[-1:].repeat(cap_padded - CAPTION_TOKENS, 1)])
        cap[CAPTION_TOKENS:] = cap_pad_token
        cap = cap.unsqueeze(0)

    tensors = {
        "latent": latent, "caption": caption, "timestep": timestep,
        "adaln_input": adaln_input,
        "image_embedded": image.squeeze(0), "cap_embedded": cap.squeeze(0),
    }

    block_map = {k: k for k in (
        "attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight",
        "attention.to_out.0.weight", "attention.norm_q.weight", "attention.norm_k.weight",
        "attention_norm1.weight", "attention_norm2.weight",
        "ffn_norm1.weight", "ffn_norm2.weight",
        "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight")}
    modulated_map = dict(block_map, **{k: k for k in (
        "adaLN_modulation.0.weight", "adaLN_modulation.0.bias")})

    # One block, reused. Its parameters are overwritten per layer.
    modulated = ZImageTransformerBlock(0, DIM, HEADS, HEADS, NORM_EPS, True, modulation=True)
    plain = ZImageTransformerBlock(0, DIM, HEADS, HEADS, NORM_EPS, True, modulation=False)

    # ---- refiners, each over its own half only -------------------------
    for layer in range(REFINERS):
        build(modulated, f"noise_refiner.{layer}.", modulated_map)
        with torch.no_grad():
            image = modulated(image, attn_mask=None, freqs_cis=img_freqs,
                              adaln_input=adaln_input)
    tensors["after_noise_refiner"] = image.squeeze(0)
    print(f"noise_refiner   {tuple(image.shape)}  (image tokens only)")

    for layer in range(REFINERS):
        build(plain, f"context_refiner.{layer}.", block_map)
        with torch.no_grad():
            cap = plain(cap, attn_mask=None, freqs_cis=cap_freqs)
    tensors["after_context_refiner"] = cap.squeeze(0)
    print(f"context_refiner {tuple(cap.shape)}  (caption tokens only, no adaLN)")

    # ---- the trunk -----------------------------------------------------
    unified = torch.cat([image, cap], dim=1)
    tensors["unified"] = unified.squeeze(0)
    print(f"unified         {tuple(unified.shape)}")

    for layer in range(LAYERS):
        build(modulated, f"layers.{layer}.", modulated_map)
        with torch.no_grad():
            unified = modulated(unified, attn_mask=None, freqs_cis=freqs,
                                adaln_input=adaln_input)
        if layer in (0, 14, LAYERS - 1):
            tensors[f"trunk_{layer:02d}"] = unified.squeeze(0)
        print(f"  layer {layer:02d}  |x| {unified.abs().mean():.4f}")

    # ---- head and unpatchify -------------------------------------------
    final = build(FinalLayer(DIM, PATCH * PATCH * LATENT_CHANNELS),
                  "all_final_layer.2-1.",
                  {"linear.weight": "linear.weight", "linear.bias": "linear.bias",
                   "adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
                   "adaLN_modulation.1.bias": "adaLN_modulation.1.bias"})
    with torch.no_grad():
        head = final(unified, c=adaln_input)
    tensors["head"] = head.squeeze(0)

    # unpatchify reads self.out_channels, so it needs a stand-in where
    # _patchify_image did not. Still the reference's own code rather than a
    # transcription of the inverse permutation, which is the part worth not
    # getting wrong.
    out = ZImageTransformer2DModel.unpatchify(
        SimpleNamespace(out_channels=LATENT_CHANNELS),
        list(head.unbind(dim=0)), [(1, LATENT_SIDE, LATENT_SIDE)], PATCH, 1)[0]
    tensors["output"] = out
    print(f"head            {tuple(head.shape)}")
    print(f"output          {tuple(out.shape)}  "
          f"range [{out.min():.3f}, {out.max():.3f}]")

    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, args.out)
    print(f"\nwrote {args.out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
