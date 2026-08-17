#!/usr/bin/env python3
"""Export what one S3-DiT layer produces, for the C port to match.

This is the stage where the port can actually be wrong. The text encoder was a
config, and the VAE was a convolution stack whose every piece already existed;
the DiT block has four things that are new at once — sandwich norm, per-head
QK-norm inside a *bidirectional* attention, low-rank adaLN driven by a 256-wide
conditioning vector, and a 3D rope split 32/48/48 at an unusually small theta.
Any one of them fails quietly: the shapes agree either way and only the values
move.

So each is exported separately rather than only the layer's output. A single
end-of-layer comparison says something is wrong without saying which of the
four it is, and the four have very different fixes.

Three blocks are captured because there are two code paths, not one:
`layers.0` and `noise_refiner.0` are modulated, `context_refiner.0` is not,
and the unmodulated path is not simply the modulated one with the scales set
to 1 — it drops the gates entirely.

The modules are built piecewise from the base checkpoint rather than by
loading ZImageTransformer2DModel, which would want ~12 GB to hand back three
blocks. The classes are diffusers' own, so the arithmetic is still the
reference's and not a paraphrase of it.

  zimage_dit_golden.py --model <base checkout> --out golden.safetensors \
      --weights weights.safetensors
"""

import argparse
import json

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

# Small enough to check in C in a moment, large enough to be real: a 32x32
# latent is a 256x256 image, and at patch 2 that is 16x16 = 256 image tokens,
# already a multiple of 32 so the image needs no padding. The caption length is
# deliberately *not* a multiple of 32, so the padded tail and the pad token are
# exercised rather than skipped.
LATENT_SIDE = 32
CAPTION_TOKENS = 40


def load_group(shards, index, prefix):
    """Pull every tensor under `prefix` out of the sharded checkpoint."""
    out = {}
    for key, shard in index.items():
        if key.startswith(prefix):
            if shard not in shards:
                shards[shard] = None
            out[key[len(prefix):]] = shard
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    root = f"{args.model}/transformer"
    index = json.load(open(f"{root}/diffusion_pytorch_model.safetensors.index.json"))
    index = index["weight_map"]
    handles = {}

    def get(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{root}/{shard}", framework="pt")
        # The checkpoint stores f32 whose values are all bf16-exact, so f32 is
        # the model's own precision here and not an upcast that flatters it.
        return handles[shard].get_tensor(key).float()

    def build(module, prefix, mapping):
        state = {theirs: get(prefix + mine) for theirs, mine in mapping.items()}
        missing, unexpected = module.load_state_dict(state, strict=True)
        assert not missing and not unexpected
        return module.eval()

    torch.manual_seed(11)

    # ---- rope -----------------------------------------------------------
    # Positions follow patchify_and_embed for the basic (non-omni) path: the
    # caption occupies 1..P on the temporal axis with both spatial axes at 0,
    # and the image starts at P+1 and spreads over the two spatial axes. The
    # unified order is [image, caption] — image first, which is the opposite
    # of most joint-attention models and is easy to assume wrongly.
    cap_padded = CAPTION_TOKENS + (-CAPTION_TOKENS) % SEQ_MULTI_OF
    tokens_side = LATENT_SIDE // PATCH
    image_tokens = tokens_side * tokens_side
    assert image_tokens % SEQ_MULTI_OF == 0, "picked a size that needs image padding"

    cap_ids = torch.tensor([[1 + i, 0, 0] for i in range(cap_padded)], dtype=torch.int32)
    img_ids = torch.tensor(
        [[cap_padded + 1, h, w] for h in range(tokens_side) for w in range(tokens_side)],
        dtype=torch.int32,
    )
    pos_ids = torch.cat([img_ids, cap_ids], dim=0)
    sequence = pos_ids.shape[0]

    rope = RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])
    freqs = rope(pos_ids)
    print(f"sequence {sequence} = {image_tokens} image + {cap_padded} caption")
    print(f"freqs_cis {tuple(freqs.shape)} {freqs.dtype}")

    # ---- conditioning ---------------------------------------------------
    t_embedder = build(
        TimestepEmbedder(256, mid_size=1024), "t_embedder.",
        {"mlp.0.weight": "mlp.0.weight", "mlp.0.bias": "mlp.0.bias",
         "mlp.2.weight": "mlp.2.weight", "mlp.2.bias": "mlp.2.bias"},
    )
    # A mid-schedule value rather than an endpoint, so a sign or scale error in
    # the sinusoid cannot hide behind t=0 or t=1.
    timestep = torch.tensor([0.375])
    with torch.no_grad():
        t_freq = TimestepEmbedder.timestep_embedding(timestep * 1000.0, 256)
        adaln_input = t_embedder(timestep * 1000.0)
    print(f"adaln_input {tuple(adaln_input.shape)} "
          f"range [{adaln_input.min():.3f}, {adaln_input.max():.3f}]")

    # ---- embedders ------------------------------------------------------
    x_embedder = build(
        torch.nn.Linear(PATCH * PATCH * LATENT_CHANNELS, DIM), "all_x_embedder.2-1.",
        {"weight": "weight", "bias": "bias"},
    )
    cap_embedder = build(
        torch.nn.Sequential(RMSNorm(CAP_DIM, eps=NORM_EPS), torch.nn.Linear(CAP_DIM, DIM)),
        "cap_embedder.",
        {"0.weight": "0.weight", "1.weight": "1.weight", "1.bias": "1.bias"},
    )
    x_pad_token = get("x_pad_token")
    cap_pad_token = get("cap_pad_token")

    # The latent is patchified with the channel varying *fastest* — the
    # permute in _patchify_image lands (pF, pH, pW, C) — so a port that lays
    # the patch out channel-major produces a plausible image of the wrong
    # thing. Taken from the reference's own method rather than transcribed:
    # a transcription would be wrong on both sides at once and the C could
    # then match a layout the model does not use. It reads nothing off self,
    # so it runs unbound.
    latent = torch.randn(LATENT_CHANNELS, 1, LATENT_SIDE, LATENT_SIDE)
    patches, size, token_grid = ZImageTransformer2DModel._patchify_image(
        None, latent, PATCH, 1
    )
    assert size == (1, LATENT_SIDE, LATENT_SIDE), size
    assert token_grid == (1, tokens_side, tokens_side), token_grid
    assert patches.shape == (image_tokens, PATCH * PATCH * LATENT_CHANNELS)
    caption = torch.randn(CAPTION_TOKENS, CAP_DIM)

    with torch.no_grad():
        image_embedded = x_embedder(patches)
        cap_embedded = cap_embedder(caption)
        # Pad by repeating the last row, then overwrite the tail with the pad
        # token — the repeat is what _pad_with_ids does and the overwrite is
        # what _prepare_sequence does, and only the second is observable.
        cap_embedded = torch.cat(
            [cap_embedded, cap_embedded[-1:].repeat(cap_padded - CAPTION_TOKENS, 1)]
        )
        cap_embedded[CAPTION_TOKENS:] = cap_pad_token
        unified = torch.cat([image_embedded, cap_embedded], dim=0).unsqueeze(0)

    print(f"unified {tuple(unified.shape)} "
          f"range [{unified.min():.3f}, {unified.max():.3f}]")

    # ---- the blocks -----------------------------------------------------
    block_map = {
        "attention.to_q.weight": "attention.to_q.weight",
        "attention.to_k.weight": "attention.to_k.weight",
        "attention.to_v.weight": "attention.to_v.weight",
        "attention.to_out.0.weight": "attention.to_out.0.weight",
        "attention.norm_q.weight": "attention.norm_q.weight",
        "attention.norm_k.weight": "attention.norm_k.weight",
        "attention_norm1.weight": "attention_norm1.weight",
        "attention_norm2.weight": "attention_norm2.weight",
        "ffn_norm1.weight": "ffn_norm1.weight",
        "ffn_norm2.weight": "ffn_norm2.weight",
        "feed_forward.w1.weight": "feed_forward.w1.weight",
        "feed_forward.w2.weight": "feed_forward.w2.weight",
        "feed_forward.w3.weight": "feed_forward.w3.weight",
    }
    modulated_map = dict(block_map)
    modulated_map.update({
        "adaLN_modulation.0.weight": "adaLN_modulation.0.weight",
        "adaLN_modulation.0.bias": "adaLN_modulation.0.bias",
    })

    def block(prefix, modulation):
        module = ZImageTransformerBlock(
            0, DIM, HEADS, HEADS, NORM_EPS, qk_norm=True, modulation=modulation
        )
        return build(module, prefix, modulated_map if modulation else block_map)

    tensors = {
        "pos_ids": pos_ids.float(),
        "freqs_real": freqs.real.contiguous(),
        "freqs_imag": freqs.imag.contiguous(),
        "timestep": timestep,
        "t_freq": t_freq,
        "adaln_input": adaln_input,
        "latent": latent,
        "patches": patches,
        "caption": caption,
        "image_embedded": image_embedded,
        "cap_embedded": cap_embedded,
        "x_pad_token": x_pad_token,
        "cap_pad_token": cap_pad_token,
        "unified": unified.squeeze(0),
    }

    # Weights go out in the *package's* naming — fused qkv, `out`, `q_norm` —
    # so the C harness is written against the names it will read in
    # production, and the same harness can later be pointed at the int8
    # package to price ConvRot without touching a line of it.
    weights = {}

    for name, prefix, modulation in [
        ("layer0", "layers.0.", True),
        ("context_refiner0", "context_refiner.0.", False),
        ("noise_refiner0", "noise_refiner.0.", True),
    ]:
        module = block(prefix, modulation)
        with torch.no_grad():
            out = module(
                unified, attn_mask=None, freqs_cis=freqs.unsqueeze(0),
                adaln_input=adaln_input if modulation else None,
            )
        tensors[f"{name}_out"] = out.squeeze(0)
        moved = (out.squeeze(0) - unified.squeeze(0)).pow(2).mean().sqrt()
        print(f"{name:18} out {tuple(out.shape)}  moved {moved:.4f} RMS")

        state = module.state_dict()
        weights[f"{name}.attention.qkv.weight"] = torch.cat(
            [state["attention.to_q.weight"], state["attention.to_k.weight"],
             state["attention.to_v.weight"]], dim=0
        ).to(torch.bfloat16)
        weights[f"{name}.attention.out.weight"] = state["attention.to_out.0.weight"].to(torch.bfloat16)
        weights[f"{name}.attention.q_norm.weight"] = state["attention.norm_q.weight"].to(torch.bfloat16)
        weights[f"{name}.attention.k_norm.weight"] = state["attention.norm_k.weight"].to(torch.bfloat16)
        for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
            weights[f"{name}.{norm}.weight"] = state[f"{norm}.weight"].to(torch.bfloat16)
        for w in ("w1", "w2", "w3"):
            weights[f"{name}.feed_forward.{w}.weight"] = state[f"feed_forward.{w}.weight"].to(torch.bfloat16)
        if modulation:
            weights[f"{name}.adaLN_modulation.0.weight"] = state["adaLN_modulation.0.weight"].to(torch.bfloat16)
            weights[f"{name}.adaLN_modulation.0.bias"] = state["adaLN_modulation.0.bias"].to(torch.bfloat16)
        del module

    # ---- the head -------------------------------------------------------
    # Carried here rather than left for stage 5 because it holds the one
    # asymmetry in the whole model: the block's adaLN is a bare Linear, while
    # the head's is SiLU *then* Linear. Reading the block and assuming the head
    # matches it produces a wrong image with no error anywhere.
    final = build(
        FinalLayer(DIM, PATCH * PATCH * LATENT_CHANNELS), "all_final_layer.2-1.",
        {"linear.weight": "linear.weight", "linear.bias": "linear.bias",
         "adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
         "adaLN_modulation.1.bias": "adaLN_modulation.1.bias"},
    )
    with torch.no_grad():
        head = final(tensors["layer0_out"].unsqueeze(0), c=adaln_input)
    tensors["final_out"] = head.squeeze(0)
    print(f"{'final_layer':18} out {tuple(head.shape)}")

    for w in ("linear", "adaLN_modulation.1"):
        state = final.state_dict()
        weights[f"final_layer.{w}.weight"] = state[f"{w}.weight"].to(torch.bfloat16)
        weights[f"final_layer.{w}.bias"] = state[f"{w}.bias"].to(torch.bfloat16)
    for key, value in {
        "x_embedder.weight": x_embedder.weight, "x_embedder.bias": x_embedder.bias,
        "cap_embedder.0.weight": cap_embedder[0].weight,
        "cap_embedder.1.weight": cap_embedder[1].weight,
        "cap_embedder.1.bias": cap_embedder[1].bias,
        "t_embedder.mlp.0.weight": t_embedder.mlp[0].weight,
        "t_embedder.mlp.0.bias": t_embedder.mlp[0].bias,
        "t_embedder.mlp.2.weight": t_embedder.mlp[2].weight,
        "t_embedder.mlp.2.bias": t_embedder.mlp[2].bias,
        "x_pad_token": x_pad_token, "cap_pad_token": cap_pad_token,
    }.items():
        weights[key] = value.detach().to(torch.bfloat16)

    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, args.out)
    save_file({k: v.contiguous().clone() for k, v in weights.items()}, args.weights)
    print(f"\nwrote {args.out} ({len(tensors)} tensors)")
    print(f"wrote {args.weights} ({len(weights)} tensors)")


if __name__ == "__main__":
    main()
