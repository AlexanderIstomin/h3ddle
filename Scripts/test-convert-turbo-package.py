#!/usr/bin/env python3
"""Focused tests for LoRA scaling in convert-turbo-package.py."""

import importlib.util
import os
import unittest

import numpy as np


SCRIPT = os.path.join(os.path.dirname(__file__), "convert-turbo-package.py")
SPEC = importlib.util.spec_from_file_location("convert_turbo_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeLora:
    def __init__(self, tensors):
        self.tensors = tensors
        self.header = {name: {} for name in tensors}

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


if __name__ == "__main__":
    unittest.main()
