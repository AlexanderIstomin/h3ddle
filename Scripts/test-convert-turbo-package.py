#!/usr/bin/env python3
"""Focused tests for adapter merging in convert-turbo-package.py."""

import importlib.util
import os
import unittest

import numpy as np


SCRIPT = os.path.join(os.path.dirname(__file__), "convert-turbo-package.py")
SPEC = importlib.util.spec_from_file_location("convert_turbo_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeLora:
    def __init__(self, tensors, metadata=None):
        self.tensors = tensors
        self.header = {name: {} for name in tensors}
        self.metadata = metadata or {}

    def f32(self, name):
        return np.asarray(self.tensors[name], dtype=np.float32)


class LoraDeltaTests(unittest.TestCase):
    stem = "diffusion_model.blocks.0.attn.out_proj"
    template = "blocks.0.attn.out_proj.weight"

    def adapter(self, alpha=None):
        tensors = {
            f"{self.stem}.lora_A.weight": [[1, 2], [3, 4]],
            f"{self.stem}.lora_B.weight": [[5, 6], [7, 8]],
        }
        if alpha is not None:
            tensors[f"{self.stem}.alpha"] = alpha
        return FakeLora(tensors)

    def test_legacy_adapter_has_unit_pair_scale(self):
        adapter = self.adapter()
        expected = adapter.f32(f"{self.stem}.lora_B.weight") @ adapter.f32(
            f"{self.stem}.lora_A.weight"
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(adapter, self.template, 1.0), expected
        )

    def test_comfy_alpha_is_divided_by_pair_rank(self):
        adapter = self.adapter(alpha=1.0)
        expected = 0.5 * (
            adapter.f32(f"{self.stem}.lora_B.weight")
            @ adapter.f32(f"{self.stem}.lora_A.weight")
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(adapter, self.template, 1.0), expected
        )

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected one finite value"):
            MODULE.lora_delta(self.adapter(alpha=[1.0, 2.0]), self.template, 1.0)

    def test_fastvideo_separate_qkv_pairs_are_fused_in_qkv_order(self):
        tensors = {}
        for index, projection in enumerate(("to_q", "to_k", "to_v"), start=1):
            stem = f"transformer_blocks.0.attn.{projection}"
            tensors[f"{stem}.lora_A.weight"] = [[1, 0]]
            tensors[f"{stem}.lora_B.weight"] = [[index], [index + 10]]
        adapter = FakeLora(tensors, metadata={"format": "fastvideo-lora-v2"})
        expected = np.asarray(
            [[1, 0], [11, 0], [2, 0], [12, 0], [3, 0], [13, 0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(
                adapter, "blocks.0.attn.qkv_proj.weight", 1.0
            ),
            expected,
        )

    def test_fastvideo_exact_weight_and_bias_deltas_use_compact_names(self):
        adapter = FakeLora(
            {
                "proj_in.diff": [[1, 2], [3, 4]],
                "proj_in.diff_b": [5, 6],
            }
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(adapter, "video_patch_proj.weight", 0.5),
            [[0.5, 1.0], [1.5, 2.0]],
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(adapter, "video_patch_proj.bias", 0.5),
            [2.5, 3.0],
        )

    def test_fastvideo_refiner_and_block_names_are_mapped(self):
        adapter = FakeLora(
            {
                "token_refiner.refiner_blocks.1.ff.net.2.lora_A.weight": [[1, 2]],
                "token_refiner.refiner_blocks.1.ff.net.2.lora_B.weight": [[3], [4]],
                "transformer_blocks.7.norm2.diff": [0.25, -0.25],
            }
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(
                adapter, "token_refiner.blocks.1.mlp.fc2.weight", 1.0
            ),
            [[3, 6], [4, 8]],
        )
        np.testing.assert_array_equal(
            MODULE.lora_delta(adapter, "blocks.7.norm2.weight", 2.0),
            [0.5, -0.5],
        )

    def test_fastvideo_profile_uses_four_step_serving_defaults(self):
        adapter = FakeLora({}, metadata={"format": "fastvideo-lora-v2"})
        self.assertEqual(MODULE.generation_profile("auto", adapter), "fasth3")
        self.assertEqual(
            MODULE.PROFILE_METADATA["fasth3"], (4, "serving", "t2va")
        )


if __name__ == "__main__":
    unittest.main()
