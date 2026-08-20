#!/usr/bin/env python3
"""Focused regression test for repack-h3-input-major.py."""

import importlib.util
import json
import os
import struct
import tempfile
import unittest


SCRIPT = os.path.join(os.path.dirname(__file__), "repack-h3-input-major.py")
SPEC = importlib.util.spec_from_file_location("h3_input_major_repack", SCRIPT)
REPACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPACK)


def write_fixture(path):
    weight = bytes([1, 2, 3, 4, 5, 6])
    scales = struct.pack("<ff", 0.25, 0.5)
    auxiliary = b"same"
    document = {
        "__metadata__": {"fixture": "preserved"},
        "blocks.0.test.weight": {
            "dtype": "I8",
            "shape": [2, 3],
            "data_offsets": [0, len(weight)],
        },
        "blocks.0.test.weight_scale": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [len(weight), len(weight) + len(scales)],
        },
        "auxiliary": {
            "dtype": "U8",
            "shape": [len(auxiliary)],
            "data_offsets": [
                len(weight) + len(scales),
                len(weight) + len(scales) + len(auxiliary),
            ],
        },
    }
    header = REPACK.padded_header(document)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        output.write(weight)
        output.write(scales)
        output.write(auxiliary)


class InputMajorRepackTests(unittest.TestCase):
    def test_repack_transposes_only_selected_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.safetensors")
            output_path = os.path.join(directory, "output.safetensors")
            write_fixture(source_path)
            with open(source_path, "rb") as source_file:
                source_before = source_file.read()

            size = REPACK.repack(
                source_path,
                output_path,
                {"blocks.0.test.weight": (2, 3)},
            )

            output = REPACK.TensorFile(output_path)
            try:
                self.assertEqual(size, os.path.getsize(output_path))
                self.assertEqual(
                    output.metadata,
                    {
                        "fixture": "preserved",
                        "h3ddle_transformer_layout": "input-major-v1",
                    },
                )
                self.assertEqual(
                    output.int8("blocks.0.test.weight", (3, 2)).tobytes(),
                    bytes([1, 4, 2, 5, 3, 6]),
                )
                scale_begin, scale_end = output.raw_bounds(
                    "blocks.0.test.weight_scale"
                )
                self.assertEqual(
                    output.map[scale_begin:scale_end], struct.pack("<ff", 0.25, 0.5)
                )
                aux_begin, aux_end = output.raw_bounds("auxiliary")
                self.assertEqual(output.map[aux_begin:aux_end], b"same")
                marker_begin, marker_end = output.raw_bounds(REPACK.MARKER_NAME)
                self.assertEqual(
                    output.map[marker_begin:marker_end],
                    struct.pack("<I", REPACK.FORMAT_VERSION),
                )
            finally:
                output.close()

            with open(source_path, "rb") as source_file:
                self.assertEqual(source_file.read(), source_before)

if __name__ == "__main__":
    unittest.main()
