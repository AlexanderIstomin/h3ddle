#!/usr/bin/env python3
"""Focused regression tests for build-h3-hybrid-adaln.py."""

import importlib.util
import json
import os
import struct
import tempfile
import unittest


SCRIPT = os.path.join(os.path.dirname(__file__), "build-h3-hybrid-adaln.py")
SPEC = importlib.util.spec_from_file_location("h3_hybrid_adaln", SCRIPT)
HYBRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HYBRID)


def write_fixture(path):
    payload = bytearray()
    document = {"__metadata__": {"fixture": "source"}}
    for name in HYBRID.overlay_names():
        shape = (
            [HYBRID.OUTPUT_DIM, HYBRID.COMPACT_TIME_DIM]
            if name.endswith("weight")
            else [HYBRID.OUTPUT_DIM]
        )
        byte_count = 2
        for dimension in shape:
            byte_count *= dimension
        begin = len(payload)
        payload.extend(bytes([len(document) % 251]) * byte_count)
        document[name] = {
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }
    payload.extend(b"excluded")
    document["blocks.24.adaln_proj.linear.bias"] = {
        "dtype": "U8",
        "shape": [8],
        "data_offsets": [len(payload) - 8, len(payload)],
    }
    header = HYBRID.HELPERS.padded_header(document)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        output.write(payload)


class HybridAdaLNTests(unittest.TestCase):
    def test_overlay_copies_only_late_adaln_and_writes_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "ref.safetensors")
            output_path = os.path.join(directory, "overlay.safetensors")
            write_fixture(source_path)
            HYBRID.build_overlay(source_path, output_path)

            overlay = HYBRID.HELPERS.TensorFile(output_path)
            source = HYBRID.HELPERS.TensorFile(source_path)
            try:
                self.assertEqual(
                    overlay.metadata,
                    {
                        "h3ddle_hybrid_layout": "fl2va-base-ref2va-adaln-25-49-v1",
                        "h3ddle_hybrid_source": "Ref2VA",
                    },
                )
                self.assertNotIn("blocks.24.adaln_proj.linear.bias", overlay.tensors)
                for name in HYBRID.overlay_names():
                    source_begin, source_end = source.raw_bounds(name)
                    overlay_begin, overlay_end = overlay.raw_bounds(name)
                    self.assertEqual(
                        overlay.map[overlay_begin:overlay_end],
                        source.map[source_begin:source_end],
                    )
                markers = {
                    HYBRID.VERSION_MARKER: HYBRID.FORMAT_VERSION,
                    HYBRID.FIRST_BLOCK_MARKER: HYBRID.FIRST_BLOCK,
                    HYBRID.BLOCK_COUNT_MARKER: HYBRID.BLOCK_COUNT,
                }
                for name, expected in markers.items():
                    begin, end = overlay.raw_bounds(name)
                    self.assertEqual(overlay.map[begin:end], struct.pack("<I", expected))
            finally:
                source.close()
                overlay.close()


if __name__ == "__main__":
    unittest.main()
