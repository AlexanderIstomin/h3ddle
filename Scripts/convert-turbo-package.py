#!/usr/bin/env python3
"""Convert MiniMax H3 pruned BF16 weights (optionally merged with a turbo
LoRA) into the INT8 ConvRot single-file layout that h3.c's optimized package
loader reads.

The output copies every tensor from a template file (the official
minimax_h3_fl2va_pruned_int8_convrot.safetensors) verbatim, except:

- the 200 quantized projections (blocks.N.{attn.qkv_proj, attn.out_proj,
  mlp.fc1, mlp.fc2}.weight): re-derived from the BF16 base, with the LoRA
  delta merged at the requested strength, rotated by the grouped Hadamard
  (H256 = H4 kron H4 kron H4 kron H4, /16 — the transform h3_convrot_bf16
  applies to activations), then symmetric per-row absmax INT8 quantization;
- the BF16 token-refiner projections: LoRA delta merged losslessly.

Self-check: run with --strength 0 and the output must reproduce the
template's INT8 payloads and scales (up to rounding ties); --self-check
compares in memory without writing a file.

Usage:
  convert-turbo-package.py --base pruned_bf16.safetensors \
      --template official_int8_convrot.safetensors \
      [--lora turbo_lora.safetensors --strength 1.0] \
      [--out merged_int8_convrot.safetensors | --self-check]
"""

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np

DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I64": 8}
GROUP = 256


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class TensorFile:
    def __init__(self, path):
        self.path = path
        self.header, self.data_offset = read_header(path)
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def raw(self, name):
        t = self.header[name]
        begin, end = t["data_offsets"]
        return self.map[self.data_offset + begin : self.data_offset + end]

    def f32(self, name):
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


def hadamard256():
    h4 = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=np.float32,
    )
    h = h4
    for _ in range(3):
        h = np.kron(h, h4)
    return h / 16.0


def lora_delta(lora, template_name, strength):
    if lora is None or strength == 0.0:
        return None
    stem = "diffusion_model." + template_name.removesuffix(".weight")
    a_name = f"{stem}.lora_A.weight"
    b_name = f"{stem}.lora_B.weight"
    if a_name not in lora.header:
        return None
    a = lora.f32(a_name)
    b = lora.f32(b_name)
    return (strength * (b @ a)).astype(np.float32)


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
    parser.add_argument("--out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.out and not args.self_check:
        parser.error("either --out or --self-check is required")

    base = TensorFile(args.base)
    template = TensorFile(args.template)
    lora = TensorFile(args.lora) if args.lora else None
    hadamard = hadamard256()

    quantized = {
        name
        for name, t in template.header.items()
        if t["dtype"] == "I8" and name.endswith(".weight")
    }
    refiner_merges = {
        name
        for name, t in template.header.items()
        if t["dtype"] == "BF16"
        and name.startswith("token_refiner.")
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
        elif name in refiner_merges:
            weight = base.f32(name) + lora_delta(lora, name, args.strength)
            emit(name, f32_to_bf16_bytes(weight.reshape(-1)))
            merged_names.add(name)
        else:
            emit(name, bytes(template.raw(name)))
        if index % 100 == 0:
            print(f"  {index}/{len(names)} tensors", file=sys.stderr)

    if derived:
        raise ValueError(f"unconsumed derived pairs: {sorted(derived)[:3]}")
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
