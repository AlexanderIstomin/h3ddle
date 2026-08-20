#!/usr/bin/env python3
"""Repack every quantized MiniMax H3 transformer linear input-major.

Comfy's optimized checkpoints store INT8 matrices as [output, input]. H3ddle's
Metal kernel reads [input, output] more efficiently, especially on machines
whose unified memory is under pressure. This tool transposes the four core
projections in all 50 blocks without dequantizing or otherwise changing a
single weight value. Scales, ConvRot metadata, and all non-core tensors are
copied byte for byte.

Without --out, the result is written beside the source as
TRANSFORMER_input_major.safetensors. The source is never modified. Re-running
is safe: an existing output is left alone unless --force is supplied.
"""

import argparse
import json
import mmap
import os
import shutil
import struct
import sys

import numpy as np


LAYERS = 50
HIDDEN = 5376
INNER = 7168
FFN = 14336
FORMAT_VERSION = 1
MARKER_NAME = "h3.transformer_input_major.version"
COPY_CHUNK_BYTES = 64 * 1024 * 1024


def h3_projection_shapes(layers=LAYERS):
    shapes = {}
    for layer in range(layers):
        prefix = f"blocks.{layer}."
        shapes[prefix + "attn.qkv_proj.weight"] = (INNER * 3, HIDDEN)
        shapes[prefix + "attn.out_proj.weight"] = (HIDDEN, INNER)
        shapes[prefix + "mlp.fc1.weight"] = (FFN * 2, HIDDEN)
        shapes[prefix + "mlp.fc2.weight"] = (HIDDEN, FFN)
    return shapes


class TensorFile:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.file = open(self.path, "rb")
        prefix = self.file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        self.header_size = struct.unpack("<Q", prefix)[0]
        if not self.header_size or self.header_size > 256 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header size")
        raw_header = self.file.read(self.header_size)
        if len(raw_header) != self.header_size:
            raise ValueError(f"{path}: truncated safetensors header")
        document = json.loads(raw_header)
        if not isinstance(document, dict):
            raise ValueError(f"{path}: safetensors header is not an object")
        self.metadata = document.pop("__metadata__", None)
        self.tensors = document
        self.data_offset = 8 + self.header_size
        self.file_size = os.fstat(self.file.fileno()).st_size
        self.data_size = self.file_size - self.data_offset
        self.ordered_names = sorted(
            self.tensors,
            key=lambda name: self.tensors[name].get("data_offsets", [-1])[0],
        )
        self._validate_ranges()
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def _validate_ranges(self):
        cursor = 0
        for name in self.ordered_names:
            descriptor = self.tensors[name]
            offsets = descriptor.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] != cursor
                or offsets[1] < offsets[0]
                or offsets[1] > self.data_size
            ):
                raise ValueError(f"{name}: invalid or non-contiguous data range")
            cursor = offsets[1]
        if cursor != self.data_size:
            raise ValueError("safetensors payload has trailing or missing data")

    def close(self):
        self.map.close()
        self.file.close()

    def int8(self, name, shape):
        descriptor = self.tensors.get(name)
        if not descriptor or descriptor.get("dtype") != "I8":
            raise ValueError(f"{name}: required I8 tensor is absent")
        if descriptor.get("shape") != list(shape):
            raise ValueError(
                f"{name}: shape {descriptor.get('shape')} != {list(shape)}"
            )
        begin, end = descriptor["data_offsets"]
        expected = int(np.prod(shape, dtype=np.int64))
        if end - begin != expected:
            raise ValueError(f"{name}: byte count does not match its shape")
        absolute = self.data_offset + begin
        return np.ndarray(shape, dtype=np.int8, buffer=self.map, offset=absolute)

    def raw_bounds(self, name):
        begin, end = self.tensors[name]["data_offsets"]
        return self.data_offset + begin, self.data_offset + end

def default_output(path):
    suffix = ".safetensors"
    if not path.endswith(suffix):
        raise ValueError("the transformer filename must end in .safetensors")
    return path[: -len(suffix)] + "_input_major" + suffix
def padded_header(document):
    payload = json.dumps(document, separators=(",", ":")).encode()
    padding = (-len(payload)) % 8
    return payload + b" " * padding


def validate_projections(source, projection_shapes):
    if MARKER_NAME in source.tensors:
        raise ValueError("checkpoint is already marked input-major")
    for name, shape in projection_shapes.items():
        source.int8(name, shape)
        scale = source.tensors.get(name + "_scale")
        output_rows = shape[0]
        if (
            not scale
            or scale.get("dtype") != "F32"
            or scale.get("shape") not in ([output_rows], [output_rows, 1])
        ):
            raise ValueError(f"{name}_scale: invalid per-output scale tensor")


def output_document(source, projection_shapes):
    document = {}
    if source.metadata is not None:
        if not isinstance(source.metadata, dict):
            raise ValueError("safetensors metadata is not an object")
        metadata = dict(source.metadata)
        metadata["h3ddle_transformer_layout"] = "input-major-v1"
        document["__metadata__"] = metadata
    cursor = 0
    for name in source.ordered_names:
        descriptor = dict(source.tensors[name])
        begin, end = descriptor["data_offsets"]
        byte_count = end - begin
        if name in projection_shapes:
            descriptor["shape"] = list(reversed(projection_shapes[name]))
        descriptor["data_offsets"] = [cursor, cursor + byte_count]
        document[name] = descriptor
        cursor += byte_count
    document[MARKER_NAME] = {
        "dtype": "U32",
        "shape": [1],
        "data_offsets": [cursor, cursor + 4],
    }
    return document, cursor + 4


def copy_range(output, source_map, begin, end):
    while begin < end:
        stop = min(begin + COPY_CHUNK_BYTES, end)
        output.write(source_map[begin:stop])
        begin = stop


def repack(source_path, output_path, projection_shapes, force=False):
    source_path = os.path.abspath(source_path)
    output_path = os.path.abspath(output_path)
    if output_path == source_path:
        raise ValueError("the repacked output must differ from the transformer")
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"repacked checkpoint already exists: {output_path}")

    output_directory = os.path.dirname(output_path) or "."
    os.makedirs(output_directory, exist_ok=True)
    temporary = output_path + ".partial"
    source = TensorFile(source_path)
    try:
        validate_projections(source, projection_shapes)
        document, payload_size = output_document(source, projection_shapes)
        header = padded_header(document)
        required = 8 + len(header) + payload_size
        free = shutil.disk_usage(output_directory).free
        if free < required:
            raise OSError(
                f"repack needs {required / (1024 ** 3):.2f} GiB but only "
                f"{free / (1024 ** 3):.2f} GiB is free"
            )

        completed = 0
        total = len(projection_shapes)
        with open(temporary, "wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for name in source.ordered_names:
                if name in projection_shapes:
                    source_shape = projection_shapes[name]
                    matrix = source.int8(name, source_shape)
                    transposed = np.ascontiguousarray(matrix.T)
                    output.write(memoryview(transposed).cast("B"))
                    completed += 1
                    print(
                        f"  projection {completed}/{total}: {name}",
                        file=sys.stderr,
                    )
                else:
                    begin, end = source.raw_bounds(name)
                    copy_range(output, source.map, begin, end)
            output.write(struct.pack("<I", FORMAT_VERSION))
            output.flush()
            os.fsync(output.fileno())
            if output.tell() != required:
                raise OSError(
                    f"repacked file size {output.tell()} != expected {required}"
                )
        os.replace(temporary, output_path)
        return os.path.getsize(output_path)
    finally:
        source.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transformer")
    parser.add_argument("--out")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        output_path = args.out or default_output(os.path.abspath(args.transformer))
        size = repack(
            args.transformer, output_path, h3_projection_shapes(), args.force
        )
    except (OSError, ValueError) as exception:
        parser.error(str(exception))
    print(f"wrote {size / (1024 ** 3):.3f} GiB: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
