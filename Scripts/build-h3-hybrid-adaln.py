#!/usr/bin/env python3
"""Extract the compact Ref2VA AdaLN overlay used by H3ddle's hybrid loader.

The hybrid path keeps the complete FL2VA transformer as its base and replaces
only the compact AdaLN projection weight and bias for blocks 25 through 49.
Those are the layers selected by ComfyUI_MinimaxH3HybridLoader's recommended
"subjective" recipe. The final AdaLN projection, timestep curve, attention,
MLP, patch, and output weights deliberately remain FL2VA.

The source is never changed. Without --out, the overlay is written next to it
as TRANSFORMER_hybrid_adaln_25_49.safetensors. Re-running is safe: an existing
output is left alone unless --force is supplied.
"""

import argparse
import importlib.util
import os
import struct
import sys


FIRST_BLOCK = 25
BLOCK_COUNT = 25
FORMAT_VERSION = 1
OUTPUT_DIM = 96_768
COMPACT_TIME_DIM = 8
VERSION_MARKER = "h3.hybrid_adaln.version"
FIRST_BLOCK_MARKER = "h3.hybrid_adaln.first_block"
BLOCK_COUNT_MARKER = "h3.hybrid_adaln.block_count"


def _load_safetensors_helpers():
    path = os.path.join(os.path.dirname(__file__), "repack-h3-input-major.py")
    spec = importlib.util.spec_from_file_location("h3_input_major_repack", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load safetensors helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_safetensors_helpers()


def overlay_names(first_block=FIRST_BLOCK, block_count=BLOCK_COUNT):
    names = []
    for block in range(first_block, first_block + block_count):
        prefix = f"blocks.{block}.adaln_proj.linear."
        names.extend((prefix + "weight", prefix + "bias"))
    return names


def default_output(path):
    suffix = ".safetensors"
    if not path.endswith(suffix):
        raise ValueError("the Ref2VA transformer filename must end in .safetensors")
    return path[: -len(suffix)] + "_hybrid_adaln_25_49" + suffix


def validate_source(source, names):
    for name in names:
        descriptor = source.tensors.get(name)
        expected_shape = [OUTPUT_DIM, COMPACT_TIME_DIM] if name.endswith("weight") else [OUTPUT_DIM]
        if not descriptor or descriptor.get("dtype") != "F16":
            raise ValueError(f"{name}: required compact F16 tensor is absent")
        if descriptor.get("shape") != expected_shape:
            raise ValueError(
                f"{name}: shape {descriptor.get('shape')} != {expected_shape}"
            )


def output_document(source, names):
    document = {
        "__metadata__": {
            "h3ddle_hybrid_layout": "fl2va-base-ref2va-adaln-25-49-v1",
            "h3ddle_hybrid_source": "Ref2VA",
        }
    }
    cursor = 0
    for name in names:
        descriptor = dict(source.tensors[name])
        begin, end = descriptor["data_offsets"]
        byte_count = end - begin
        descriptor["data_offsets"] = [cursor, cursor + byte_count]
        document[name] = descriptor
        cursor += byte_count
    for marker in (VERSION_MARKER, FIRST_BLOCK_MARKER, BLOCK_COUNT_MARKER):
        document[marker] = {
            "dtype": "U32",
            "shape": [1],
            "data_offsets": [cursor, cursor + 4],
        }
        cursor += 4
    return document, cursor


def build_overlay(source_path, output_path, force=False):
    source_path = os.path.abspath(source_path)
    output_path = os.path.abspath(output_path)
    if source_path == output_path:
        raise ValueError("the overlay output must differ from the transformer")
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"hybrid overlay already exists: {output_path}")

    temporary = output_path + ".partial"
    source = HELPERS.TensorFile(source_path)
    names = overlay_names()
    try:
        validate_source(source, names)
        document, payload_size = output_document(source, names)
        header = HELPERS.padded_header(document)
        required = 8 + len(header) + payload_size
        with open(temporary, "wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for index, name in enumerate(names, start=1):
                begin, end = source.raw_bounds(name)
                HELPERS.copy_range(output, source.map, begin, end)
                print(f"  tensor {index}/{len(names)}: {name}", file=sys.stderr)
            output.write(struct.pack("<I", FORMAT_VERSION))
            output.write(struct.pack("<I", FIRST_BLOCK))
            output.write(struct.pack("<I", BLOCK_COUNT))
            output.flush()
            os.fsync(output.fileno())
            if output.tell() != required:
                raise OSError(
                    f"overlay file size {output.tell()} != expected {required}"
                )
        os.replace(temporary, output_path)
        return os.path.getsize(output_path)
    finally:
        source.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref2va_transformer")
    parser.add_argument("--out")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_path = args.out or default_output(os.path.abspath(args.ref2va_transformer))
    try:
        size = build_overlay(args.ref2va_transformer, output_path, args.force)
    except (OSError, ValueError) as exception:
        parser.error(str(exception))
    print(f"wrote {size / (1024 ** 2):.2f} MiB: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
