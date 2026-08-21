#!/usr/bin/env python3
"""Focused regression tests for repack-ltx-input-major.py."""

import importlib.util
import os
import struct
import tempfile
import unittest


SCRIPT = os.path.join(os.path.dirname(__file__), "repack-ltx-input-major.py")
SPEC = importlib.util.spec_from_file_location("ltx_input_major_repack", SCRIPT)
REPACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPACK)


def write_fixture(path, include_scale=True):
    weight = bytes([1, 2, 3, 4, 5, 6])
    scales = struct.pack("<ff", 0.25, 0.5)
    auxiliary = b"same"
    scale_size = len(scales) if include_scale else 0
    document = {
        "__metadata__": {"fixture": "preserved"},
        "model.diffusion_model.transformer_blocks.0.test.weight": {
            "dtype": "I8",
            "shape": [2, 3],
            "data_offsets": [0, len(weight)],
        },
        "auxiliary": {
            "dtype": "U8",
            "shape": [len(auxiliary)],
            "data_offsets": [
                len(weight) + scale_size,
                len(weight) + scale_size + len(auxiliary),
            ],
        },
    }
    if include_scale:
        document["model.diffusion_model.transformer_blocks.0.test.weight_scale"] = {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [len(weight), len(weight) + len(scales)],
        }
    # Safetensors data ranges, unlike header keys, must be contiguous in offset
    # order. TensorFile sorts them before validating.
    header = REPACK.padded_header(document)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        output.write(weight)
        if include_scale:
            output.write(scales)
        output.write(auxiliary)


class InputMajorRepackTests(unittest.TestCase):
    def test_ltx_projection_map_covers_every_block_projection(self):
        shapes = REPACK.ltx_projection_shapes()
        self.assertEqual(len(shapes), REPACK.BLOCKS * 28)
        prefix = "model.diffusion_model.transformer_blocks.47."
        self.assertEqual(
            shapes[prefix + "audio_to_video_attn.to_q.weight"],
            (REPACK.AUDIO_DIM, REPACK.VIDEO_DIM),
        )
        self.assertEqual(
            shapes[prefix + "video_to_audio_attn.to_k.weight"],
            (REPACK.AUDIO_DIM, REPACK.VIDEO_DIM),
        )
        self.assertEqual(
            shapes[prefix + "ff.net.2.weight"],
            (REPACK.VIDEO_DIM, REPACK.VIDEO_FF),
        )

    def test_repack_transposes_only_selected_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.safetensors")
            output_path = os.path.join(directory, "output.safetensors")
            write_fixture(source_path)
            with open(source_path, "rb") as source_file:
                source_before = source_file.read()

            name = "model.diffusion_model.transformer_blocks.0.test.weight"
            numpy = REPACK.np
            REPACK.np = None
            try:
                size = REPACK.repack(source_path, output_path, {name: (2, 3)})
                converted, untouched = REPACK.verify_repack(
                    source_path, output_path, {name: (2, 3)}
                )
            finally:
                REPACK.np = numpy
            self.assertEqual(converted, 1)
            self.assertEqual(untouched, 2)

            output = REPACK.TensorFile(output_path)
            try:
                self.assertEqual(size, os.path.getsize(output_path))
                self.assertEqual(
                    output.metadata,
                    {
                        "fixture": "preserved",
                        REPACK.METADATA_NAME: "input-major-v1",
                    },
                )
                weight_begin, weight_end = output.raw_bounds(name)
                self.assertEqual(
                    output.map[weight_begin:weight_end], bytes([1, 4, 2, 5, 3, 6])
                )
                scale_begin, scale_end = output.raw_bounds(name + "_scale")
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

    def test_missing_scale_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.safetensors")
            output_path = os.path.join(directory, "output.safetensors")
            write_fixture(source_path, include_scale=False)
            name = "model.diffusion_model.transformer_blocks.0.test.weight"
            with self.assertRaisesRegex(ValueError, "invalid per-output scale"):
                REPACK.repack(source_path, output_path, {name: (2, 3)})
            self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
