#!/usr/bin/env python3
"""Convert a Z-Image-Turbo release into the safetensors package layout h3.c reads.

Two sources, because the useful weights are in two places. The diffusion
transformer is taken already quantized from martin-rizzo's INT8-ConvRot
repack, whose per-tensor `.comfy_quant` markers are exactly the schema
h3_weight_i8_linear_convrot_group parses; requantizing it here could only lose
fidelity to a conversion that was already tuned. Everything else comes from the
Tongyi-MAI release.

What comes out, one file per subsystem so each can be validated on its own
against the goldens and demand-paged independently:

  transformer.safetensors    30 S3-DiT layers plus 2+2 refiners, int8 ConvRot
  text_encoder.safetensors   Qwen3-4B, the same encoder FLUX.2 klein uses
  vae_decoder.safetensors    AutoencoderKL's decoder half
  tokenizer.json             the released vocabulary, byte for byte

What is deliberately left behind:

  * the VAE *encoder* and `quant_conv`. Text to image never encodes an image,
    so half the autoencoder is dead weight. Editing would need them back, and
    would need SigLIP 2 as well, which this package also omits.
  * nothing else. No tensor value changes, which is what makes a tensor-by-
    tensor check against the reference meaningful.

Constants that live in the reference's Python rather than in any tensor — the
flow-match shift, the latent scaling and shift, the DiT's own hyperparameters —
are recorded in the output metadata so the engine reads them back instead of
carrying its own copy that can drift.

Usage:
  convert-z-image-package.py --base Z-Image-Turbo \\
      --transformer Z-Image-Turbo-INT8-ConvRot-ComfyUI \\
      --out Z-Image-Turbo-INT8-ConvRot
"""

import argparse
import json
import os
import shutil
import struct
import sys

# The engine rejects anything else, so refuse to emit it.
REQUIRED_QUANT_FORMAT = "int8_tensorwise"

NOTICE = """Z-Image-Turbo by Alibaba Tongyi Lab, licensed under the Apache License 2.0.

The diffusion transformer is the INT8-ConvRot quantization published by Martin
Rizzo, also under the Apache License 2.0, and is copied here byte for byte.

Modifications by H3ddle: the weights were repackaged from the released
safetensors into one file per subsystem, and the autoencoder's encoder half was
dropped because generating an image never runs it. No tensor values were
changed. No weights were retrained, merged, pruned, or quantized here.
"""

DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1}


def read_safetensors(path):
    """Returns (dict of name -> (dtype, shape, bytes), metadata).

    Raw bytes throughout: int8 payloads and bf16 both pass through untouched,
    and neither has to survive a round trip through a dtype NumPy cannot spell.
    """
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        blob = handle.read()
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        tensors[name] = (entry["dtype"], entry["shape"], blob[start:end])
    return tensors, metadata


def read_sharded(directory, index_name):
    """Merges a sharded checkpoint, or reads the single file when there is one."""
    index_path = os.path.join(directory, index_name)
    if not os.path.exists(index_path):
        single = index_name.replace(".index.json", "")
        return read_safetensors(os.path.join(directory, single))[0]
    with open(index_path) as handle:
        index = json.load(handle)
    merged = {}
    for shard in sorted(set(index["weight_map"].values())):
        merged.update(read_safetensors(os.path.join(directory, shard))[0])
    missing = set(index["weight_map"]) - set(merged)
    if missing:
        raise SystemExit(f"{index_name}: {len(missing)} tensors the index "
                         f"promised are not in the shards")
    return merged


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        dtype, shape, payload = tensors[name]
        expected = DTYPE_SIZES[dtype]
        for dimension in shape:
            expected *= dimension
        if expected != len(payload):
            raise SystemExit(f"{name}: {len(payload)} bytes, expected {expected}")
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)     # the data section wants 8-byte alignment
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in ordered:
            handle.write(tensors[name][2])


def regular_hadamard_size(size):
    """The engine's own test, mirrored: a power of four, at least four.

    Rejecting a group size here rather than at load time means the package is
    never built in a shape h3_weights.c would refuse.
    """
    if size < 4:
        return False
    while size > 1 and size % 4 == 0:
        size //= 4
    return size == 1


def check_quantization(tensors):
    """Every ConvRot marker parses, and they all agree on one group size."""
    groups = set()
    markers = [name for name in tensors if name.endswith(".comfy_quant")]
    if not markers:
        raise SystemExit("no .comfy_quant markers: this is not a quantized "
                         "transformer, and the engine's int8 path needs them")
    for name in markers:
        dtype, _, payload = tensors[name]
        if dtype != "U8":
            raise SystemExit(f"{name}: markers must be U8, found {dtype}")
        marker = json.loads(payload.decode("utf-8").rstrip("\0"))
        if marker.get("format") != REQUIRED_QUANT_FORMAT:
            raise SystemExit(f"{name}: format is {marker.get('format')!r}, "
                             f"the engine reads {REQUIRED_QUANT_FORMAT!r}")
        if not marker.get("convrot"):
            raise SystemExit(f"{name}: ConvRot is off; the engine's int8 path "
                             f"expects rotated weights")
        group = marker.get("convrot_groupsize")
        if not isinstance(group, int) or not regular_hadamard_size(group):
            raise SystemExit(f"{name}: group size {group!r} is not a power of "
                             f"four, which h3_weights.c requires")
        groups.add(group)

        stem = name[: -len(".comfy_quant")]
        for needed in (f"{stem}.weight", f"{stem}.weight_scale"):
            if needed not in tensors:
                raise SystemExit(f"{stem}: marked quantized but {needed} is missing")
    if len(groups) != 1:
        raise SystemExit(f"mixed ConvRot group sizes {sorted(groups)}; the "
                         f"engine holds one per linear but the loader assumes "
                         f"the package is uniform")
    return len(markers), groups.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True,
                        help="Tongyi-MAI/Z-Image-Turbo checkout")
    parser.add_argument("--transformer", required=True,
                        help="INT8-ConvRot checkout holding the quantized DiT")
    parser.add_argument("--out", required=True, help="package directory to write")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- the transformer, copied rather than rebuilt --------------------
    candidates = [name for name in sorted(os.listdir(args.transformer))
                  if name.endswith(".safetensors")]
    if len(candidates) != 1:
        raise SystemExit(f"expected one safetensors file in {args.transformer}, "
                         f"found {len(candidates)}")
    source = os.path.join(args.transformer, candidates[0])
    tensors, _ = read_safetensors(source)
    quantized, group = check_quantization(tensors)
    print(f"transformer: {len(tensors)} tensors, {quantized} int8 ConvRot "
          f"linears at group {group}")

    with open(os.path.join(args.base, "transformer", "config.json")) as handle:
        dit_config = json.load(handle)
    # Copied whole: the file is already exactly what the engine reads, and
    # rewriting it would only risk reordering or re-encoding what is correct.
    shutil.copyfile(source, os.path.join(args.out, "transformer.safetensors"))

    # --- the text encoder, merged from its shards ----------------------
    encoder = read_sharded(os.path.join(args.base, "text_encoder"),
                           "model.safetensors.index.json")
    with open(os.path.join(args.base, "text_encoder", "config.json")) as handle:
        encoder_config = json.load(handle)
    dropped = [name for name in encoder if name.startswith("lm_head")]
    for name in dropped:
        del encoder[name]
    print(f"text encoder: {len(encoder)} tensors"
          + (f", dropped {len(dropped)} language-head" if dropped else "")
          + f" ({encoder_config['num_hidden_layers']} layers, "
          f"{encoder_config['hidden_size']} wide)")
    write_safetensors(
        os.path.join(args.out, "text_encoder.safetensors"), encoder,
        {"architecture": "qwen3",
         "hidden_size": str(encoder_config["hidden_size"]),
         "layers": str(encoder_config["num_hidden_layers"]),
         "heads": str(encoder_config["num_attention_heads"]),
         "kv_heads": str(encoder_config["num_key_value_heads"]),
         "head_dim": str(encoder_config["head_dim"]),
         "rope_theta": repr(encoder_config["rope_theta"]),
         "rms_eps": repr(encoder_config["rms_norm_eps"])})

    # --- the VAE, decoder half only ------------------------------------
    vae = read_sharded(os.path.join(args.base, "vae"),
                       "diffusion_pytorch_model.safetensors.index.json")
    with open(os.path.join(args.base, "vae", "config.json")) as handle:
        vae_config = json.load(handle)
    decoder = {name: value for name, value in vae.items()
               if name.startswith(("decoder.", "post_quant_conv."))}
    if not decoder:
        raise SystemExit("no decoder tensors in the VAE; the layout is not the "
                         "AutoencoderKL one this expects")
    print(f"vae decoder: {len(decoder)} of {len(vae)} tensors "
          f"({len(vae) - len(decoder)} encoder-side dropped)")
    write_safetensors(
        os.path.join(args.out, "vae_decoder.safetensors"), decoder,
        # Applied around the decoder by the caller, so record them rather than
        # folding them into a weight: a recorded constant can be checked
        # against the reference, a folded one cannot.
        {"latent_channels": str(vae_config["latent_channels"]),
         "scaling_factor": repr(vae_config["scaling_factor"]),
         "shift_factor": repr(vae_config["shift_factor"]),
         "block_out_channels": json.dumps(vae_config["block_out_channels"]),
         "layers_per_block": str(vae_config["layers_per_block"]),
         "norm_num_groups": str(vae_config["norm_num_groups"])})

    # --- tokenizer and the constants that live in Python ---------------
    shutil.copyfile(os.path.join(args.base, "tokenizer", "tokenizer.json"),
                    os.path.join(args.out, "tokenizer.json"))

    with open(os.path.join(args.base, "scheduler",
                           "scheduler_config.json")) as handle:
        scheduler = json.load(handle)
    recipe = {
        "model": "Z-Image-Turbo",
        "transformer": {key: dit_config[key] for key in (
            "n_layers", "n_refiner_layers", "dim", "n_heads", "n_kv_heads",
            "in_channels", "all_patch_size", "axes_dims", "axes_lens",
            "cap_feat_dim", "rope_theta", "norm_eps", "qk_norm", "t_scale")},
        "scheduler": {"kind": scheduler["_class_name"],
                      "shift": scheduler["shift"],
                      "train_timesteps": scheduler["num_train_timesteps"]},
        # Turbo is distilled to eight function evaluations and is trained
        # without classifier-free guidance; a guidance scale above zero makes
        # it worse, not stronger.
        "sampling": {"steps": 8, "guidance": 0.0},
        "quantization": {"format": REQUIRED_QUANT_FORMAT,
                         "convrot": True, "convrot_groupsize": group},
    }
    with open(os.path.join(args.out, "recipe.json"), "w") as handle:
        json.dump(recipe, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(args.out, "NOTICE"), "w") as handle:
        handle.write(NOTICE)

    total = sum(os.path.getsize(os.path.join(args.out, name))
                for name in os.listdir(args.out))
    print(f"wrote {args.out}: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
