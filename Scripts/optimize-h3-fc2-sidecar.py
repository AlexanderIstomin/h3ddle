#!/usr/bin/env python3
"""Build an input-major FC2 sidecar for an optimized MiniMax H3 DiT.

The optimized checkpoint stores every INT8 linear as [output, input]. H3's
64x40 Metal tile can read that layout, but FC2's deep input dimension makes an
[input, output] copy materially faster. This tool writes only those 50 FC2
matrices to a sibling safetensors file; it never modifies the source model.

Without --out, the result is written beside the transformer as
TRANSFORMER_fc2_input_major.safetensors. Re-running is safe: an existing
sidecar is left alone unless --force is supplied.
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
FFN = 14336
FORMAT_VERSION = 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211


def fnv1a64(data):
    value = FNV_OFFSET
    for octet in data:
        value ^= octet
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


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
        self.header = json.loads(raw_header)
        self.header.pop("__metadata__", None)
        self.data_offset = 8 + self.header_size
        self.header_fingerprint = fnv1a64(prefix + raw_header)
        self.file_size = os.fstat(self.file.fileno()).st_size
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        self.map.close()
        self.file.close()

    def int8(self, name, shape):
        descriptor = self.header.get(name)
        if not descriptor or descriptor.get("dtype") != "I8":
            raise ValueError(f"{name}: required I8 tensor is absent")
        if descriptor.get("shape") != list(shape):
            raise ValueError(
                f"{name}: shape {descriptor.get('shape')} != {list(shape)}"
            )
        begin, end = descriptor["data_offsets"]
        expected = int(np.prod(shape))
        if end - begin != expected:
            raise ValueError(f"{name}: byte count does not match its shape")
        absolute = self.data_offset + begin
        return np.ndarray(shape, dtype=np.int8, buffer=self.map, offset=absolute)


def default_output(path):
    suffix = ".safetensors"
    if not path.endswith(suffix):
        raise ValueError("the transformer filename must end in .safetensors")
    return path[: -len(suffix)] + "_fc2_input_major" + suffix


def padded_header(document):
    payload = json.dumps(document, separators=(",", ":")).encode()
    padding = (8 - ((8 + len(payload)) % 8)) % 8
    return payload + b" " * padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transformer")
    parser.add_argument("--out")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path = os.path.abspath(args.transformer)
    output_path = os.path.abspath(args.out or default_output(source_path))
    if output_path == source_path:
        parser.error("the sidecar output must differ from the transformer")
    if os.path.exists(output_path) and not args.force:
        parser.error(f"sidecar already exists: {output_path}")

    source = TensorFile(source_path)
    temporary = output_path + ".partial"
    try:
        names = [f"blocks.{layer}.mlp.fc2.weight" for layer in range(LAYERS)]
        for name in names:
            source.int8(name, (HIDDEN, FFN))

        marker_data = {
            "h3.fc2_input_major.version": struct.pack("<I", FORMAT_VERSION),
            "h3.fc2_input_major.source_file_size": struct.pack(
                "<Q", source.file_size
            ),
            "h3.fc2_input_major.source_header_fnv1a64": struct.pack(
                "<Q", source.header_fingerprint
            ),
        }
        descriptors = {}
        cursor = 0
        for name, data in marker_data.items():
            descriptors[name] = {
                "dtype": "U32" if len(data) == 4 else "U64",
                "shape": [1],
                "data_offsets": [cursor, cursor + len(data)],
            }
            cursor += len(data)

        matrix_bytes = HIDDEN * FFN
        for name in names:
            descriptors[name] = {
                "dtype": "I8",
                "shape": [FFN, HIDDEN],
                "data_offsets": [cursor, cursor + matrix_bytes],
            }
            cursor += matrix_bytes

        header = padded_header(
            {
                "__metadata__": {
                    "format": "h3ddle-fc2-input-major-v1",
                    "source": os.path.basename(source_path),
                },
                **descriptors,
            }
        )
        required = 8 + len(header) + cursor
        free = shutil.disk_usage(os.path.dirname(output_path) or ".").free
        if free < required:
            raise OSError(
                f"sidecar needs {required / (1024 ** 3):.2f} GiB but only "
                f"{free / (1024 ** 3):.2f} GiB is free"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(temporary, "wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for data in marker_data.values():
                output.write(data)
            for index, name in enumerate(names):
                output.write(source.int8(name, (HIDDEN, FFN)).T.tobytes())
                output.flush()
                print(f"  FC2 {index + 1}/{LAYERS}", file=sys.stderr)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
        print(
            f"wrote {os.path.getsize(output_path) / (1024 ** 3):.3f} GiB: "
            f"{output_path}"
        )
    finally:
        source.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
