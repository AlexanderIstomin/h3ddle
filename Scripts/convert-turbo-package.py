#!/usr/bin/env python3
"""Convert MiniMax H3 pruned BF16 weights (optionally merged with a distilled
adapter) into the INT8 ConvRot single-file layout that h3.c's optimized
package loader reads.

The output copies every tensor from a template file (the official
minimax_h3_fl2va_pruned_int8_convrot.safetensors) verbatim, except:

- the 200 quantized projections (blocks.N.{attn.qkv_proj, attn.out_proj,
  mlp.fc1, mlp.fc2}.weight): re-derived from the BF16 base, with the LoRA
  delta merged at the requested strength, rotated by the grouped Hadamard
  (H256 = H4 kron H4 kron H4 kron H4, /16 — the transform h3_convrot_bf16
  applies to activations), then symmetric per-row absmax INT8 quantization;
- every remaining BF16/F32 parameter addressed by the adapter: low-rank and
  exact ``.diff``/``.diff_b`` deltas merged losslessly.

The historical fused H3 Turbo adapters are supported directly. FastVideo's
``fastvideo-lora-v2`` schema is understood for validation and tests, but its
time/AdaLN updates cannot be folded into the old eight-dimensional compact
template. Use ``convert-fasth3-package.py`` with FastVideo's merged dense
checkpoint; it preserves those updates as exact four-step AdaLN tables. VSA
adapters are rejected rather than partially merged: their
``to_gate_compress.set_weight`` tensors require the versioned VSA format and
runtime.

Self-check: run with --strength 0 and the output must reproduce the
template's INT8 payloads and scales (up to rounding ties); --self-check
compares in memory without writing a file.

Usage:
  convert-turbo-package.py --base pruned_bf16.safetensors \
      --template official_int8_convrot.safetensors \
      [--lora distilled_adapter.safetensors --strength 1.0] \
      [--generation-profile auto|standard|turbo|fasth3] \
      [--out merged_int8_convrot.safetensors | --self-check]
"""

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np

DTYPE_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I8": 1,
    "U8": 1,
    "U32": 4,
    "I64": 8,
}
GROUP = 256
PROFILE_METADATA = {
    "standard": (20, "serving", "conditional"),
    "turbo": (8, "beta", "conditional"),
    "fasth3": (4, "serving", "t2va"),
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {})
    return header, metadata, 8 + n


class TensorFile:
    def __init__(self, path):
        self.path = path
        self.header, self.metadata, self.data_offset = read_header(path)
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.read_names = set()

    def raw(self, name):
        t = self.header[name]
        begin, end = t["data_offsets"]
        return self.map[self.data_offset + begin : self.data_offset + end]

    def f32(self, name):
        self.read_names.add(name)
        t = self.header[name]
        raw = np.frombuffer(self.raw(name), dtype=np.uint8)
        if t["dtype"] == "BF16":
            bits = raw.view(np.uint16).astype(np.uint32) << 16
            values = bits.view(np.float32)
        elif t["dtype"] == "F32":
            values = raw.view(np.float32)
        else:
            raise ValueError(f"{name}: cannot widen {t['dtype']}")
        return values.reshape(t["shape"]).astype(np.float32)


def f32_to_bf16_bytes(values):
    bits = values.astype(np.float32).view(np.uint32)
    # Round to nearest even, matching the engine's conversions.
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).tobytes()


def f32_bytes(values, dtype):
    if dtype == "BF16":
        return f32_to_bf16_bytes(values)
    if dtype == "F32":
        return values.astype(np.float32).tobytes()
    raise ValueError(f"cannot encode merged values as {dtype}")


def hadamard256():
    h4 = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=np.float32,
    )
    h = h4
    for _ in range(3):
        h = np.kron(h, h4)
    return h / 16.0


def fastvideo_stems(template_name):
    """Adapter module names that feed one h3.c parameter.

    The FastVideo adapter is written against MiniMax's Diffusers tree while
    h3.c reads Comfy's compact/fused tree. A QKV target deliberately returns
    three stems in Q, K, V order; every other target returns one.
    """
    suffix = ".weight" if template_name.endswith(".weight") else ".bias"
    stem = template_name.removesuffix(suffix)
    if stem.startswith("token_refiner.blocks."):
        stem = "token_refiner.refiner_blocks." + stem.removeprefix(
            "token_refiner.blocks."
        )
    elif stem.startswith("blocks."):
        stem = "transformer_blocks." + stem.removeprefix("blocks.")

    replacements = {
        "time_embedder.proj_in": "time_embedder.linear_1",
        "time_embedder.proj_out": "time_embedder.linear_2",
        "condition_proj": "context_embedder",
        "video_patch_proj": "proj_in",
        "audio_patch_proj": "audio_proj_in",
        "final_layer.norm": "norm_out.norm",
        "final_layer.adaln_proj.linear": "norm_out.linear",
        "final_layer.video_out": "proj_out",
        "final_layer.audio_out": "audio_proj_out",
    }
    stem = replacements.get(stem, stem)
    stem = stem.replace(".attn.out_proj", ".attn.to_out.0")
    stem = stem.replace(".attn.q_norm", ".attn.norm_q")
    stem = stem.replace(".attn.k_norm", ".attn.norm_k")
    stem = stem.replace(".mlp.fc1", ".ff.net.0.proj")
    stem = stem.replace(".mlp.fc2", ".ff.net.2")
    if stem.endswith(".attn.qkv_proj"):
        prefix = stem.removesuffix("qkv_proj")
        return [prefix + "to_q", prefix + "to_k", prefix + "to_v"]
    return [stem]


def pair_delta(lora, stem, strength):
    a_name = f"{stem}.lora_A.weight"
    b_name = f"{stem}.lora_B.weight"
    if a_name not in lora.header:
        return None
    if b_name not in lora.header:
        raise ValueError(f"{a_name}: matching {b_name} is missing")
    a = lora.f32(a_name)
    b = lora.f32(b_name)
    pair_scale = 1.0
    alpha_names = (f"{stem}.alpha", f"{stem}.lora_alpha")
    alpha_name = next((name for name in alpha_names if name in lora.header), None)
    if alpha_name:
        alpha = lora.f32(alpha_name).reshape(-1)
        if alpha.size != 1 or not np.isfinite(alpha[0]):
            raise ValueError(f"{alpha_name}: expected one finite value")
        pair_scale = float(alpha[0]) / a.shape[0]
    return (strength * pair_scale * (b @ a)).astype(np.float32)


def exact_delta(lora, template_name, strength):
    suffix = ".diff" if template_name.endswith(".weight") else ".diff_b"
    values = []
    for stem in fastvideo_stems(template_name):
        name = stem + suffix
        if name not in lora.header:
            continue
        values.append(strength * lora.f32(name))
    if not values:
        return None
    if len(values) == 1:
        return values[0].astype(np.float32)
    return np.concatenate(values, axis=0).astype(np.float32)


def lora_delta(lora, template_name, strength):
    if lora is None or strength == 0.0:
        return None
    # Historical H3/Comfy adapters address h3.c's already-fused names.
    stem = "diffusion_model." + template_name.removesuffix(".weight")
    low_rank = pair_delta(lora, stem, strength)
    if low_rank is None:
        pieces = [
            value
            for candidate in fastvideo_stems(template_name)
            if (value := pair_delta(lora, candidate, strength)) is not None
        ]
        low_rank = (
            np.concatenate(pieces, axis=0).astype(np.float32)
            if len(pieces) > 1
            else pieces[0] if pieces else None
        )
    dense = exact_delta(lora, template_name, strength)
    if low_rank is None:
        return dense
    if dense is None:
        return low_rank
    if low_rank.shape != dense.shape:
        raise ValueError(
            f"{template_name}: low-rank delta {low_rank.shape} != exact delta {dense.shape}"
        )
    return (low_rank + dense).astype(np.float32)


def generation_profile(requested, lora):
    if requested != "auto":
        return requested
    if lora is None:
        return "standard"
    if getattr(lora, "metadata", {}).get("format") == "fastvideo-lora-v2":
        return "fasth3"
    return "turbo"


def quantize_convrot(weight, hadamard):
    rows, columns = weight.shape
    if columns % GROUP:
        raise ValueError(f"input dimension {columns} is not a multiple of {GROUP}")
    rotated = (weight.reshape(rows, columns // GROUP, GROUP) @ hadamard).reshape(
        rows, columns
    )
    scale = np.abs(rotated).max(axis=1, keepdims=True).astype(np.float32) / 127.0
    scale[scale == 0.0] = 1.0
    q = np.rint(rotated / scale)
    return np.clip(q, -127, 127).astype(np.int8), scale.reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--generation-profile",
        choices=("auto", "standard", "turbo", "fasth3"),
        default="auto",
        help="runtime defaults recorded in safetensors metadata",
    )
    parser.add_argument("--out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.out and not args.self_check:
        parser.error("either --out or --self-check is required")

    base = TensorFile(args.base)
    template = TensorFile(args.template)
    lora = TensorFile(args.lora) if args.lora else None
    profile = generation_profile(args.generation_profile, lora)
    if profile == "fasth3" and lora is None:
        parser.error("--generation-profile fasth3 requires --lora")
    replacement_keys = (
        [name for name in lora.header if name.endswith(".set_weight")]
        if lora is not None
        else []
    )
    if replacement_keys:
        raise ValueError(
            "adapter carries VSA replacement tensors (for example "
            f"{replacement_keys[0]}); dense conversion cannot drop them"
        )
    if (
        lora is not None
        and lora.metadata.get("format") == "fastvideo-lora-v2"
        and "adaln_t_table" in template.header
    ):
        raise ValueError(
            "FastH3 updates the full timestep/AdaLN network, but this template "
            "contains the legacy compact AdaLN curve; use "
            "convert-fasth3-package.py with the merged Dense FastH3 checkpoint"
        )
    hadamard = hadamard256()

    quantized = {
        name
        for name, t in template.header.items()
        if t["dtype"] == "I8" and name.endswith(".weight")
    }
    exact_merges = {
        name
        for name, t in template.header.items()
        if t["dtype"] in ("BF16", "F32")
        and lora is not None
        and lora_delta(lora, name, args.strength) is not None
    }

    # Output tensor sizes are fully determined by the template header, so the
    # file header can be written before any tensor is computed and the 21 GB
    # payload streamed one tensor at a time.
    names = list(template.header.keys())
    header = {}
    cursor = 0
    for name in names:
        t = template.header[name]
        size = int(np.prod(t["shape"] or [1])) * DTYPE_SIZES[t["dtype"]]
        header[name] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    out = None
    if args.out:
        default_steps, sigma_schedule, conditioning = PROFILE_METADATA[profile]
        payload = json.dumps(
            {
                "__metadata__": {
                    "base_model": "MiniMax-H3-pruned",
                    "conversion": "h3ddle convert-turbo-package.py",
                    # Basename only: the full path names a directory on
                    # whoever ran the conversion, and this metadata ships
                    # inside a published file.
                    "lora": os.path.basename(args.lora) if args.lora else "none",
                    "lora_strength": str(args.strength),
                    "h3.generation_profile": profile,
                    "h3.default_steps": str(default_steps),
                    "h3.sigma_schedule": sigma_schedule,
                    "h3.conditioning": conditioning,
                    "adapter_format": (
                        lora.metadata.get("format", "legacy") if lora else "none"
                    ),
                },
                **header,
            },
            separators=(",", ":"),
        ).encode()
        out = open(args.out, "wb")
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)

    def emit(name, data):
        expected = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
        if len(data) != expected:
            raise ValueError(f"{name}: produced {len(data)} bytes, expected {expected}")
        if out:
            out.write(data)

    stats = {"exact_rows": 0, "rows": 0, "int8_mismatch": 0, "elements": 0}
    derived = {}
    merged_names = set()

    # The template interleaves each weight with its scale in either order, so
    # the pair is derived when its first member appears and the cached half
    # is consumed by the second.
    def derived_pair(weight_name):
        if weight_name in derived:
            return derived[weight_name]
        t = template.header[weight_name]
        weight = base.f32(weight_name)
        if list(weight.shape) != [t["shape"][0], t["shape"][1]]:
            raise ValueError(
                f"{weight_name}: base shape {weight.shape} != {t['shape']}"
            )
        delta = lora_delta(lora, weight_name, args.strength)
        if delta is not None:
            if delta.shape != weight.shape:
                raise ValueError(f"{weight_name}: LoRA delta shape {delta.shape}")
            weight = weight + delta
            merged_names.add(weight_name)
        q, scale = quantize_convrot(weight, hadamard)
        if args.self_check:
            reference = np.frombuffer(template.raw(weight_name), dtype=np.int8)
            stats["int8_mismatch"] += int((q.reshape(-1) != reference).sum())
            stats["elements"] += reference.size
            ref_scale = np.frombuffer(
                template.raw(weight_name + "_scale"), dtype=np.float32
            )
            stats["rows"] += ref_scale.size
            stats["exact_rows"] += int(np.isclose(scale, ref_scale, rtol=1e-6).sum())
        derived[weight_name] = {
            "weight": q.tobytes(),
            "scale": scale.astype(np.float32).tobytes(),
        }
        return derived[weight_name]

    def consume(weight_name, part):
        pair = derived_pair(weight_name)
        data = pair[part]
        pair[part] = None
        if pair["weight"] is None and pair["scale"] is None:
            del derived[weight_name]
        return data

    for index, name in enumerate(names):
        if name in quantized:
            emit(name, consume(name, "weight"))
        elif name.endswith("_scale") and name.removesuffix("_scale") in quantized:
            emit(name, consume(name.removesuffix("_scale"), "scale"))
        elif name in exact_merges:
            weight = base.f32(name)
            delta = lora_delta(lora, name, args.strength)
            if delta.shape != weight.shape:
                raise ValueError(f"{name}: adapter delta shape {delta.shape}")
            emit(name, f32_bytes((weight + delta).reshape(-1), template.header[name]["dtype"]))
            merged_names.add(name)
        else:
            emit(name, bytes(template.raw(name)))
        if index % 100 == 0:
            print(f"  {index}/{len(names)} tensors", file=sys.stderr)

    if derived:
        raise ValueError(f"unconsumed derived pairs: {sorted(derived)[:3]}")
    if lora is not None and args.strength != 0.0:
        payload_keys = {
            name
            for name in lora.header
            if name.endswith(
                (
                    ".lora_A.weight",
                    ".lora_B.weight",
                    ".alpha",
                    ".lora_alpha",
                    ".diff",
                    ".diff_b",
                    ".set_weight",
                )
            )
        }
        unused = sorted(payload_keys - lora.read_names)
        if unused:
            raise ValueError(
                "adapter tensors were not mapped; first unmatched keys: "
                + ", ".join(unused[:3])
            )
    merged = len(merged_names)
    print(f"tensors: {len(names)} · LoRA-merged matrices: {merged}")
    if args.self_check:
        total = max(stats["elements"], 1)
        print(
            f"self-check vs template: int8 mismatches {stats['int8_mismatch']}"
            f" / {stats['elements']} ({100.0 * stats['int8_mismatch'] / total:.5f}%)"
            f" · exact scales {stats['exact_rows']} / {stats['rows']}"
        )
    if out:
        out.close()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
