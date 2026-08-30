#!/usr/bin/env python3
"""Convert FastVideo's merged Dense or VSA FastH3 checkpoint for native h3.c.

FastH3 fine-tunes MiniMax H3's full 2,688-dimensional timestep/AdaLN path.
The existing 21 GB H3ddle template instead carries an eight-dimensional
pruned curve, so applying the adapter to that template is neither exact nor
shape-compatible. This converter starts from FastVideo's already-merged dense
checkpoint and preserves the trained function at every row the four-call,
T2VA-only serving schedule can request:

* Q/K/V are fused, ConvRot-transformed, INT8-quantized, and stored in the
  template's input-major or output-major layout;
* Diffusers' value-first SwiGLU input halves are swapped into h3.c's
  gate-first Comfy layout before quantization or copying;
* all other shape-compatible transformer, refiner, norm, and boundary weights
  come from the merged checkpoint rather than the old template;
* each full AdaLN matrix is evaluated at the seven unique video/audio
  timesteps used by four calls and serialized as an exact BF16 lookup table.
* VSA packages additionally carry every learned compression-gate projection
  plus an explicit tile-64 / 90%-sparsity contract. They use package format 2;
  dense packages remain format 1 and cannot be mistaken for VSA at runtime.

The output remains one transformer file and reuses H3ddle's optimized Qwen,
VideoVAE, AudioVAE, tokenizer, and preview decoder package files.

Usage:
  convert-fasth3-package.py \
      --source FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/transformer \
      --template minimax_h3_fl2va_pruned_int8_convrot_input_major.safetensors \
      --out model/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
"""

import argparse
import importlib.util
import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_PATH = SCRIPT_DIR / "convert-turbo-package.py"
SPEC = importlib.util.spec_from_file_location("h3ddle_convert_h3", LEGACY_PATH)
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)

FASTH3_DENSE_VERSION = 1
FASTH3_VSA_VERSION = 2
FASTH3_STEPS = 4
FASTH3_DENSE_REVISION = "f624f08c6c279ab43534c003e556fc5b295b6558"
FASTH3_VSA_REVISION = "b65818d41939b5085451074fe8ca8b799f8d4921"
FASTH3_DENSE_REPOSITORY = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
)
FASTH3_VSA_REPOSITORY = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
)
VSA_TILE_SIZE = 64
VSA_SPARSITY = 0.9
HIDDEN = 5376
TIME_INPUT = 256
TIME_HIDDEN = 5376
TIME_DIM = 2688
BLOCK_OUTPUT = 3 * 6 * HIDDEN
FINAL_OUTPUT = 2 * HIDDEN
BLOCKS = 50
INNER = 7168


class ShardedTensorStore:
    """Header-indexed, mmap-backed safetensors directory."""

    def __init__(self, location):
        root = Path(location)
        if (root / "transformer").is_dir():
            root = root / "transformer"
        paths = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
        if not paths:
            raise ValueError(f"no safetensors files in {root}")
        self.files = []
        self.tensors = {}
        for path in paths:
            tensor_file = CONVERT.TensorFile(str(path))
            self.files.append(tensor_file)
            for name, description in tensor_file.header.items():
                if name in self.tensors:
                    raise ValueError(f"duplicate source tensor: {name}")
                self.tensors[name] = (tensor_file, description)

    def has(self, name):
        return name in self.tensors

    def description(self, name):
        try:
            return self.tensors[name][1]
        except KeyError as error:
            raise ValueError(f"merged FastH3 tensor is absent: {name}") from error

    def f32(self, name):
        try:
            tensor_file, _ = self.tensors[name]
        except KeyError as error:
            raise ValueError(f"merged FastH3 tensor is absent: {name}") from error
        return tensor_file.f32(name)

    def f32_rows(self, name, first, stop):
        tensor_file, description = self.tensors[name]
        shape = description["shape"]
        if len(shape) != 2 or first < 0 or stop < first or stop > shape[0]:
            raise ValueError(f"invalid row slice for {name}: {first}:{stop}")
        dtype = description["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"{name}: cannot widen {dtype}")
        element_bytes = CONVERT.DTYPE_SIZES[dtype]
        columns = shape[1]
        begin = description["data_offsets"][0] + first * columns * element_bytes
        end = description["data_offsets"][0] + stop * columns * element_bytes
        raw = tensor_file.map[
            tensor_file.data_offset + begin : tensor_file.data_offset + end
        ]
        values = np.frombuffer(raw, dtype=np.uint8)
        if dtype == "BF16":
            bits = values.view(np.uint16).astype(np.uint32) << 16
            result = bits.view(np.float32)
        else:
            result = values.view(np.float32)
        return result.reshape(stop - first, columns).astype(np.float32)


def source_keys(template_name):
    if not template_name.endswith((".weight", ".bias")):
        return []
    suffix = ".weight" if template_name.endswith(".weight") else ".bias"
    return [stem + suffix for stem in CONVERT.fastvideo_stems(template_name)]


def source_shape(source, template_name):
    keys = source_keys(template_name)
    if not keys or not all(source.has(key) for key in keys):
        return None
    shapes = [source.description(key)["shape"] for key in keys]
    if len(shapes) == 1:
        return shapes[0]
    if any(len(shape) != 2 or shape[1:] != shapes[0][1:] for shape in shapes):
        raise ValueError(f"cannot fuse source tensors for {template_name}: {shapes}")
    return [sum(shape[0] for shape in shapes), *shapes[0][1:]]


def source_f32(source, template_name):
    keys = source_keys(template_name)
    values = [source.f32(key) for key in keys]
    if not values:
        raise ValueError(f"no merged source mapping for {template_name}")
    result = values[0] if len(values) == 1 else np.concatenate(values, axis=0)
    # The Diffusers release stores fc1 as [value, gate], while h3.c's Comfy
    # layout and fused SwiGLU kernels consume [gate, value]. The dimensions are
    # identical, so omitting this surgery passes every shape check and yields
    # structured noise. It applies to both the main DiT and text refiner.
    if template_name.endswith((".mlp.fc1.weight", ".mlp.fc1.bias")):
        if result.shape[0] % 2:
            raise ValueError(f"{template_name}: SwiGLU output is not even")
        half = result.shape[0] // 2
        result = np.concatenate((result[half:], result[:half]), axis=0)
    return result


def vsa_source_name(block):
    return f"transformer_blocks.{block}.attn.to_gate_compress.weight"


def resolve_attention(source, requested):
    gates = [source.has(vsa_source_name(block)) for block in range(BLOCKS)]
    if any(gates) and not all(gates):
        missing = next(index for index, present in enumerate(gates) if not present)
        raise ValueError(f"VSA checkpoint is missing gate projection for block {missing}")
    inferred = "vsa" if all(gates) else "dense"
    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"--attention {requested} does not match the {inferred} source checkpoint"
        )
    return inferred


def vsa_tensor_specifications(template):
    qkv = template.header["blocks.0.attn.qkv_proj.weight"]
    qkv_scale = template.header["blocks.0.attn.qkv_proj.weight_scale"]
    qkv_marker = template.header["blocks.0.attn.qkv_proj.comfy_quant"]
    if qkv["dtype"] != "I8" or len(qkv["shape"]) != 2:
        raise ValueError("VSA conversion requires the optimized INT8 template")
    if qkv["shape"][0] == HIDDEN:
        gate_shape = [HIDDEN, INNER]
    elif qkv["shape"][1] == HIDDEN:
        gate_shape = [INNER, HIDDEN]
    else:
        raise ValueError("template QKV weight has no H3 hidden dimension")
    if qkv_scale["dtype"] != "F32" or not qkv_scale["shape"]:
        raise ValueError("template QKV scale has the wrong schema")
    scale_shape = [INNER, *qkv_scale["shape"][1:]]
    specifications = {}
    for block in range(BLOCKS):
        base = f"blocks.{block}.attn.vsa_gate"
        specifications[base + ".weight"] = {"dtype": "I8", "shape": gate_shape}
        specifications[base + ".weight_scale"] = {
            "dtype": "F32",
            "shape": scale_shape,
        }
        specifications[base + ".comfy_quant"] = {
            "dtype": qkv_marker["dtype"],
            "shape": qkv_marker["shape"],
        }
    return specifications


def precomputed_tensor_specifications(attention, time_rows):
    """Return specs in the exact order their payloads are emitted.

    Safetensors offsets describe a packed byte stream, so this order is part
    of the package format. In particular, VSA's tile contract is emitted
    before the comparatively large AdaLN tables.
    """
    specifications = {
        "h3.fasth3.version": {"dtype": "U32", "shape": [1]},
        "h3.fasth3.steps": {"dtype": "U32", "shape": [1]},
        "h3.fasth3.times": {"dtype": "F32", "shape": [time_rows]},
    }
    if attention == "vsa":
        specifications["h3.fasth3.vsa.tile_size"] = {
            "dtype": "U32",
            "shape": [1],
        }
        specifications["h3.fasth3.vsa.sparsity"] = {
            "dtype": "F32",
            "shape": [1],
        }
    for block in range(BLOCKS):
        specifications[f"h3.fasth3.blocks.{block}.adaln"] = {
            "dtype": "BF16",
            "shape": [time_rows, BLOCK_OUTPUT],
        }
    specifications["h3.fasth3.final.adaln"] = {
        "dtype": "BF16",
        "shape": [time_rows, FINAL_OUTPUT],
    }
    return specifications


def bf16_round(values):
    raw = CONVERT.f32_to_bf16_bytes(np.asarray(values, dtype=np.float32))
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(np.shape(values))


def silu(values):
    values = np.asarray(values, dtype=np.float32)
    exponential = np.exp(-np.abs(values)).astype(np.float32)
    sigmoid = np.where(
        values >= 0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential),
    )
    return (values * sigmoid).astype(np.float32)


def shifted_sigma(base, shift):
    base = np.float32(base)
    shift = np.float32(shift)
    return np.float32(shift * base / np.float32(1.0 + (shift - 1.0) * base))


def fasth3_time_rows():
    """The row order produced by h3_dit_schedule.prepare_rows for 4-step T2VA."""
    rows = []
    for index in range(FASTH3_STEPS):
        base = np.float32(1.0 - np.float32(index) / np.float32(FASTH3_STEPS))
        video = np.float32(1.0 - shifted_sigma(base, 12.0))
        audio = np.float32(1.0 - shifted_sigma(base, 3.0))
        if video == audio:
            rows.append(video)
        elif video < audio:
            rows.extend((video, audio))
        else:
            rows.extend((audio, video))
    return np.asarray(rows, dtype=np.float32)


def timestep_embedding(times):
    half = TIME_INPUT // 2
    frequencies = np.exp(
        -np.log(np.float32(10000.0))
        * np.arange(half, dtype=np.float32)
        / np.float32(half)
    ).astype(np.float32)
    angles = times[:, None].astype(np.float32) * frequencies[None, :]
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(
        np.float32
    )


def linear(inputs, weight, bias):
    return (inputs @ weight.T + bias).astype(np.float32)


def fasth3_time_embedding(source, times):
    features = timestep_embedding(times)
    hidden = linear(
        features,
        source.f32("time_embedder.linear_1.weight"),
        source.f32("time_embedder.linear_1.bias"),
    )
    embedded = linear(
        silu(hidden),
        source.f32("time_embedder.linear_2.weight"),
        source.f32("time_embedder.linear_2.bias"),
    )
    # FastVideo applies SiLU in FP32, then casts to the BF16 AdaLN weight.
    return bf16_round(silu(embedded))


def chunked_projection(source, weight_name, bias_name, inputs, chunk_rows=8192):
    weight_shape = source.description(weight_name)["shape"]
    if weight_shape[1] != inputs.shape[1]:
        raise ValueError(
            f"{weight_name}: input {inputs.shape[1]} != weight {weight_shape[1]}"
        )
    bias = source.f32(bias_name).reshape(-1)
    if bias.size != weight_shape[0]:
        raise ValueError(f"{bias_name}: bias does not match {weight_name}")
    output = np.empty((inputs.shape[0], weight_shape[0]), dtype=np.float32)
    for first in range(0, weight_shape[0], chunk_rows):
        stop = min(weight_shape[0], first + chunk_rows)
        weight = source.f32_rows(weight_name, first, stop)
        output[:, first:stop] = linear(inputs, weight, bias[first:stop])
    return output


def encoded(values, dtype):
    if dtype in ("BF16", "F32"):
        return CONVERT.f32_bytes(values, dtype)
    if dtype == "F16":
        return np.asarray(values, dtype=np.float16).tobytes()
    raise ValueError(f"cannot encode source values as {dtype}")


def require_source_shape(source, name, expected):
    description = source.description(name)
    if description["shape"] != list(expected):
        raise ValueError(
            f"{name}: merged source shape {description['shape']} != {list(expected)}"
        )
    if description["dtype"] not in ("BF16", "F32"):
        raise ValueError(f"{name}: merged source dtype must be BF16 or F32")


def validate_source(source, template, quantized, attention="dense"):
    """Fail before allocating the approximately 21 GB output file."""
    require_source_shape(
        source, "time_embedder.linear_1.weight", [TIME_HIDDEN, TIME_INPUT]
    )
    require_source_shape(source, "time_embedder.linear_1.bias", [TIME_HIDDEN])
    require_source_shape(
        source, "time_embedder.linear_2.weight", [TIME_DIM, TIME_HIDDEN]
    )
    require_source_shape(source, "time_embedder.linear_2.bias", [TIME_DIM])
    for block in range(BLOCKS):
        prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        require_source_shape(source, prefix + ".weight", [BLOCK_OUTPUT, TIME_DIM])
        require_source_shape(source, prefix + ".bias", [BLOCK_OUTPUT])
    require_source_shape(source, "norm_out.linear.weight", [FINAL_OUTPUT, TIME_DIM])
    require_source_shape(source, "norm_out.linear.bias", [FINAL_OUTPUT])
    if attention == "vsa":
        for block in range(BLOCKS):
            require_source_shape(source, vsa_source_name(block), [INNER, HIDDEN])

    for name in sorted(quantized):
        shape = source_shape(source, name)
        if shape is None:
            raise ValueError(f"{name}: no merged FastH3 source mapping")
        target = template.header[name]["shape"]
        if target != shape and target != list(reversed(shape)):
            raise ValueError(
                f"{name}: template shape {target} is incompatible with "
                f"merged source {shape}"
            )
        scale_name = name + "_scale"
        if scale_name not in template.header:
            raise ValueError(f"{name}: template is missing {scale_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="merged Dense FastH3 repository root or transformer shard directory",
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--attention", choices=("auto", "dense", "vsa"), default="auto"
    )
    parser.add_argument("--source-revision")
    args = parser.parse_args()

    output_path = Path(args.out)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial_path.exists():
        parser.error("output and .partial paths must not already exist")

    source = ShardedTensorStore(args.source)
    template = CONVERT.TensorFile(args.template)
    attention = resolve_attention(source, args.attention)
    fasth3_version = (
        FASTH3_VSA_VERSION if attention == "vsa" else FASTH3_DENSE_VERSION
    )
    source_repository = (
        FASTH3_VSA_REPOSITORY if attention == "vsa" else FASTH3_DENSE_REPOSITORY
    )
    source_revision = args.source_revision or (
        FASTH3_VSA_REVISION if attention == "vsa" else FASTH3_DENSE_REVISION
    )
    quantized = {
        name
        for name, description in template.header.items()
        if description["dtype"] == "I8" and name.endswith(".weight")
    }
    validate_source(source, template, quantized, attention)
    hadamard = CONVERT.hadamard256()
    times = fasth3_time_rows()
    time_rows = int(times.size)
    precomputed = precomputed_tensor_specifications(attention, time_rows)
    vsa_tensors = vsa_tensor_specifications(template) if attention == "vsa" else {}

    names = list(template.header) + list(vsa_tensors) + list(precomputed)
    specifications = {**template.header, **vsa_tensors, **precomputed}
    header = {}
    cursor = 0
    for name in names:
        description = specifications[name]
        size = int(np.prod(description["shape"] or [1])) * CONVERT.DTYPE_SIZES[
            description["dtype"]
        ]
        header[name] = {
            "dtype": description["dtype"],
            "shape": description["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    metadata = {
        "base_model": source_repository,
        "source_revision": source_revision,
        "conversion": "h3ddle convert-fasth3-package.py",
        "h3.generation_profile": "fasth3",
        "h3.default_steps": "4",
        "h3.sigma_schedule": "serving",
        "h3.conditioning": "t2va",
        "h3.fasth3.version": str(fasth3_version),
        "h3.fasth3.attention": attention,
    }
    payload = json.dumps(
        {"__metadata__": metadata, **header}, separators=(",", ":")
    ).encode()

    output = None

    def emit(name, data):
        expected = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
        if len(data) != expected:
            raise ValueError(f"{name}: produced {len(data)} bytes, expected {expected}")
        output.write(data)

    derived = {}

    def derived_pair(weight_name):
        if weight_name in derived:
            return derived[weight_name]
        weight = source_f32(source, weight_name)
        quantized_weight, scales = CONVERT.quantize_convrot(weight, hadamard)
        target_shape = template.header[weight_name]["shape"]
        if target_shape == list(quantized_weight.shape):
            payload_weight = quantized_weight.tobytes()
        elif target_shape == list(reversed(quantized_weight.shape)):
            payload_weight = np.ascontiguousarray(quantized_weight.T).tobytes()
        else:
            raise ValueError(
                f"{weight_name}: template shape {target_shape} is incompatible "
                f"with merged source {list(quantized_weight.shape)}"
            )
        derived[weight_name] = {
            "weight": payload_weight,
            "scale": scales.astype(np.float32).tobytes(),
        }
        return derived[weight_name]

    def consume(weight_name, part):
        pair = derived_pair(weight_name)
        data = pair[part]
        pair[part] = None
        if pair["weight"] is None and pair["scale"] is None:
            del derived[weight_name]
        return data

    try:
        output = open(partial_path, "xb")
        output.write(struct.pack("<Q", len(payload)))
        output.write(payload)
        for index, name in enumerate(template.header):
            description = template.header[name]
            if name in quantized:
                emit(name, consume(name, "weight"))
            elif name.endswith("_scale") and name.removesuffix("_scale") in quantized:
                emit(name, consume(name.removesuffix("_scale"), "scale"))
            else:
                shape = source_shape(source, name)
                if (
                    description["dtype"] in ("BF16", "F16", "F32")
                    and shape == description["shape"]
                ):
                    values = source_f32(source, name).reshape(-1)
                    emit(name, encoded(values, description["dtype"]))
                else:
                    emit(name, bytes(template.raw(name)))
            if index % 100 == 0:
                print(f"  {index}/{len(template.header)} template tensors", file=sys.stderr)

        if derived:
            raise ValueError(f"unconsumed quantized pairs: {sorted(derived)[:3]}")
        if attention == "vsa":
            marker = bytes(template.raw("blocks.0.attn.qkv_proj.comfy_quant"))
            for block in range(BLOCKS):
                source_weight = source.f32(vsa_source_name(block))
                quantized_weight, scales = CONVERT.quantize_convrot(
                    source_weight, hadamard
                )
                base = f"blocks.{block}.attn.vsa_gate"
                target_shape = vsa_tensors[base + ".weight"]["shape"]
                if target_shape == list(quantized_weight.shape):
                    weight_bytes = quantized_weight.tobytes()
                elif target_shape == list(reversed(quantized_weight.shape)):
                    weight_bytes = np.ascontiguousarray(quantized_weight.T).tobytes()
                else:
                    raise ValueError(
                        f"{base}.weight: target {target_shape} is incompatible "
                        f"with source {list(quantized_weight.shape)}"
                    )
                emit(base + ".weight", weight_bytes)
                emit(base + ".weight_scale", scales.astype(np.float32).tobytes())
                emit(base + ".comfy_quant", marker)
                print(f"  {block + 1}/{BLOCKS} VSA compression gates", file=sys.stderr)

        emit("h3.fasth3.version", struct.pack("<I", fasth3_version))
        emit("h3.fasth3.steps", struct.pack("<I", FASTH3_STEPS))
        emit("h3.fasth3.times", times.astype(np.float32).tobytes())
        if attention == "vsa":
            emit("h3.fasth3.vsa.tile_size", struct.pack("<I", VSA_TILE_SIZE))
            emit(
                "h3.fasth3.vsa.sparsity",
                struct.pack("<f", VSA_SPARSITY),
            )

        embedded = fasth3_time_embedding(source, times)
        for block in range(BLOCKS):
            prefix = f"transformer_blocks.{block}.adaln_proj.linear"
            values = chunked_projection(
                source, prefix + ".weight", prefix + ".bias", embedded
            )
            emit(
                f"h3.fasth3.blocks.{block}.adaln",
                CONVERT.f32_to_bf16_bytes(values.reshape(-1)),
            )
            print(f"  {block + 1}/{BLOCKS} FastH3 AdaLN tables", file=sys.stderr)
        final = chunked_projection(
            source, "norm_out.linear.weight", "norm_out.linear.bias", embedded
        )
        emit(
            "h3.fasth3.final.adaln",
            CONVERT.f32_to_bf16_bytes(final.reshape(-1)),
        )
        output.close()
        output = None
        os.replace(partial_path, output_path)
    except BaseException:
        if output is not None:
            output.close()
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass
        raise

    print(
        f"wrote {output_path} · {attention} FastH3 format {fasth3_version} · "
        f"{len(template.header)} template tensors · "
        f"{BLOCKS + 1} exact four-step AdaLN tables"
    )


if __name__ == "__main__":
    main()
