#!/usr/bin/env python3
"""Repack every quantized LTX-2.5 transformer projection input-major.

The released Comfy INT8 ConvRot checkpoint stores matrices as [output, input].
H3ddle's Metal tile reads [input, output] more efficiently. This tool transposes
all 1,344 quantized projections in the 48-block dual-stream transformer without
dequantizing or changing any value. Scales, ConvRot metadata, dense tensors, and
all other payloads are copied byte for byte.

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

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


BLOCKS = 48
VIDEO_DIM = 4096
AUDIO_DIM = 2048
VIDEO_FF = VIDEO_DIM * 4
AUDIO_FF = AUDIO_DIM * 4
FORMAT_VERSION = 1
MARKER_NAME = "ltx.transformer_input_major.version"
METADATA_NAME = "h3ddle_ltx_transformer_layout"
COPY_CHUNK_BYTES = 64 * 1024 * 1024
# Keep the tool importable and its exact fixture tests runnable on a clean
# Python installation. Full checkpoints still use NumPy: a Python-level
# transpose of multi-megabyte projection matrices would take unreasonably long.
STDLIB_TRANSPOSE_LIMIT = 1024 * 1024


ATTENTIONS = (
    ("attn1", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM),
    ("attn2", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM),
    ("audio_attn1", AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM),
    ("audio_attn2", AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM),
    ("audio_to_video_attn", VIDEO_DIM, AUDIO_DIM, AUDIO_DIM, VIDEO_DIM),
    ("video_to_audio_attn", AUDIO_DIM, VIDEO_DIM, AUDIO_DIM, AUDIO_DIM),
)


def ltx_projection_shapes(blocks=BLOCKS):
    shapes = {}
    for block in range(blocks):
        prefix = f"model.diffusion_model.transformer_blocks.{block}."
        for name, query_in, kv_in, inner, out_dim in ATTENTIONS:
            attention = prefix + name + "."
            shapes[attention + "to_q.weight"] = (inner, query_in)
            shapes[attention + "to_k.weight"] = (inner, kv_in)
            shapes[attention + "to_v.weight"] = (inner, kv_in)
            shapes[attention + "to_out.0.weight"] = (out_dim, inner)
        shapes[prefix + "ff.net.0.proj.weight"] = (VIDEO_FF, VIDEO_DIM)
        shapes[prefix + "ff.net.2.weight"] = (VIDEO_DIM, VIDEO_FF)
        shapes[prefix + "audio_ff.net.0.proj.weight"] = (AUDIO_FF, AUDIO_DIM)
        shapes[prefix + "audio_ff.net.2.weight"] = (AUDIO_DIM, AUDIO_FF)
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

    def int8_bounds(self, name, shape):
        descriptor = self.tensors.get(name)
        if not descriptor or descriptor.get("dtype") != "I8":
            raise ValueError(f"{name}: required I8 tensor is absent")
        if descriptor.get("shape") != list(shape):
            raise ValueError(
                f"{name}: shape {descriptor.get('shape')} != {list(shape)}"
            )
        begin, end = descriptor["data_offsets"]
        expected = 1
        for dimension in shape:
            expected *= dimension
        if end - begin != expected:
            raise ValueError(f"{name}: byte count does not match its shape")
        absolute = self.data_offset + begin
        return absolute, expected

    def int8(self, name, shape):
        absolute, _ = self.int8_bounds(name, shape)
        if np is None:
            raise RuntimeError(
                "NumPy is required for full LTX checkpoint repacking; "
                "install it with `python3 -m pip install numpy`"
            )
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
        raise ValueError("checkpoint is already marked LTX input-major")
    for name, shape in projection_shapes.items():
        source.int8_bounds(name, shape)
        scale = source.tensors.get(name + "_scale")
        output_rows = shape[0]
        if (
            not scale
            or scale.get("dtype") != "F32"
            or scale.get("shape") not in ([output_rows], [output_rows, 1])
        ):
            raise ValueError(f"{name}_scale: invalid per-output scale tensor")


def require_fast_transpose(projection_shapes):
    if np is not None:
        return
    largest = max((rows * columns for rows, columns in projection_shapes.values()),
                  default=0)
    if largest > STDLIB_TRANSPOSE_LIMIT:
        raise RuntimeError(
            "NumPy is required for full LTX checkpoint repacking; "
            "install it with `python3 -m pip install numpy`"
        )


def write_transpose(output, source, name, shape):
    if np is not None:
        matrix = source.int8(name, shape)
        transposed = np.ascontiguousarray(matrix.T)
        output.write(memoryview(transposed).cast("B"))
        return

    absolute, byte_count = source.int8_bounds(name, shape)
    columns = shape[1]
    view = memoryview(source.map)[absolute:absolute + byte_count]
    try:
        for column in range(columns):
            output.write(view[column:byte_count:columns].tobytes())
    finally:
        view.release()


def transpose_matches(source, output, name, source_shape):
    if np is not None:
        source_matrix = source.int8(name, source_shape)
        output_shape = tuple(reversed(source_shape))
        output_matrix = output.int8(name, output_shape)
        # Bound the temporary transpose to a few MiB even for the
        # 4096x16384 feed-forward matrices.
        rows_per_chunk = max(1, COPY_CHUNK_BYTES // output_shape[1])
        for begin in range(0, output_shape[0], rows_per_chunk):
            end = min(begin + rows_per_chunk, output_shape[0])
            expected = np.ascontiguousarray(source_matrix[:, begin:end].T)
            if not np.array_equal(output_matrix[begin:end], expected):
                return False
        return True

    source_begin, byte_count = source.int8_bounds(name, source_shape)
    output_begin, output_count = output.int8_bounds(
        name, tuple(reversed(source_shape))
    )
    if output_count != byte_count:
        return False
    columns = source_shape[1]
    source_view = memoryview(source.map)[source_begin:source_begin + byte_count]
    try:
        cursor = output_begin
        for column in range(columns):
            expected = source_view[column:byte_count:columns].tobytes()
            if output.map[cursor:cursor + len(expected)] != expected:
                return False
            cursor += len(expected)
        return cursor == output_begin + output_count
    finally:
        source_view.release()


def output_document(source, projection_shapes):
    if source.metadata is not None and not isinstance(source.metadata, dict):
        raise ValueError("safetensors metadata is not an object")
    metadata = dict(source.metadata or {})
    metadata[METADATA_NAME] = "input-major-v1"
    document = {"__metadata__": metadata}
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


def ranges_equal(first_map, first_begin, first_end,
                 second_map, second_begin, second_end):
    if first_end - first_begin != second_end - second_begin:
        return False
    while first_begin < first_end:
        count = min(COPY_CHUNK_BYTES, first_end - first_begin)
        if (first_map[first_begin:first_begin + count] !=
                second_map[second_begin:second_begin + count]):
            return False
        first_begin += count
        second_begin += count
    return True


def verify_repack(source_path, output_path, projection_shapes):
    source = TensorFile(source_path)
    output = TensorFile(output_path)
    try:
        validate_projections(source, projection_shapes)
        require_fast_transpose(projection_shapes)
        expected_names = set(source.tensors) | {MARKER_NAME}
        if set(output.tensors) != expected_names:
            missing = sorted(expected_names - set(output.tensors))
            extra = sorted(set(output.tensors) - expected_names)
            raise ValueError(
                f"repacked tensor set differs; missing={missing[:3]} "
                f"extra={extra[:3]}"
            )
        expected_metadata = dict(source.metadata or {})
        expected_metadata[METADATA_NAME] = "input-major-v1"
        if output.metadata != expected_metadata:
            raise ValueError("repacked metadata does not preserve the source")

        marker_begin, marker_end = output.raw_bounds(MARKER_NAME)
        if output.map[marker_begin:marker_end] != struct.pack("<I", FORMAT_VERSION):
            raise ValueError("repacked layout marker has the wrong value")

        untouched = 0
        converted = 0
        for name in source.ordered_names:
            original = source.tensors[name]
            repacked = output.tensors.get(name)
            if not repacked:
                raise ValueError(f"repacked checkpoint lost {name}")
            original_schema = {
                key: value for key, value in original.items()
                if key != "data_offsets"
            }
            repacked_schema = {
                key: value for key, value in repacked.items()
                if key != "data_offsets"
            }
            if name in projection_shapes:
                expected_schema = dict(original_schema)
                expected_schema["shape"] = list(reversed(projection_shapes[name]))
                if repacked_schema != expected_schema:
                    raise ValueError(f"repacked projection schema differs: {name}")
                if not transpose_matches(
                    source, output, name, projection_shapes[name]
                ):
                    raise ValueError(f"repacked projection differs: {name}")
                converted += 1
            else:
                if repacked_schema != original_schema:
                    raise ValueError(f"untouched tensor schema differs: {name}")
                source_begin, source_end = source.raw_bounds(name)
                output_begin, output_end = output.raw_bounds(name)
                if not ranges_equal(
                    source.map, source_begin, source_end,
                    output.map, output_begin, output_end,
                ):
                    raise ValueError(f"untouched tensor payload differs: {name}")
                untouched += 1
        return converted, untouched
    finally:
        output.close()
        source.close()


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
        require_fast_transpose(projection_shapes)
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
                    write_transpose(output, source, name, projection_shapes[name])
                    completed += 1
                    if completed % 28 == 0 or completed == total:
                        print(
                            f"  projections {completed}/{total}",
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all source projection and scale schemas without writing",
    )
    parser.add_argument(
        "--verify",
        metavar="REPACKED",
        help="compare a completed repack exactly against the source",
    )
    args = parser.parse_args()

    try:
        projection_shapes = ltx_projection_shapes()
        if args.verify:
            converted, untouched = verify_repack(
                args.transformer, args.verify, projection_shapes
            )
            print(
                f"verified {converted} exact transposes and {untouched} "
                f"byte-identical tensors: {os.path.abspath(args.verify)}"
            )
            return
        if args.validate_only:
            source = TensorFile(args.transformer)
            try:
                validate_projections(source, projection_shapes)
            finally:
                source.close()
            print(
                f"validated {len(projection_shapes)} output-major projections: "
                f"{os.path.abspath(args.transformer)}"
            )
            return
        output_path = args.out or default_output(os.path.abspath(args.transformer))
        size = repack(args.transformer, output_path, projection_shapes, args.force)
    except (OSError, RuntimeError, ValueError) as exception:
        parser.error(str(exception))
    print(f"wrote {size / (1024 ** 3):.3f} GiB: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
