#!/usr/bin/env python3
"""Focused tests for the native FastH3 conversion primitives."""

import importlib.util
import os
import unittest

import numpy as np


SCRIPT = os.path.join(os.path.dirname(__file__), "convert-fasth3-package.py")
SPEC = importlib.util.spec_from_file_location("convert_fasth3_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeStore:
    def __init__(self, weight, bias):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    def description(self, name):
        values = self.weight if name == "w" else self.bias
        return {"shape": list(values.shape)}

    def f32(self, name):
        return self.weight if name == "w" else self.bias

    def f32_rows(self, name, first, stop):
        self.assert_weight_name(name)
        return self.weight[first:stop]

    @staticmethod
    def assert_weight_name(name):
        if name != "w":
            raise AssertionError(name)


class FastH3ConversionTests(unittest.TestCase):
    def test_attention_profile_is_inferred_from_all_fifty_gates(self):
        class Store:
            def __init__(self, missing=None):
                self.missing = missing

            def has(self, name):
                return name != MODULE.vsa_source_name(self.missing)

        self.assertEqual(MODULE.resolve_attention(Store(missing=-1), "auto"), "vsa")

        class DenseStore:
            @staticmethod
            def has(_name):
                return False

        self.assertEqual(MODULE.resolve_attention(DenseStore(), "auto"), "dense")
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.resolve_attention(DenseStore(), "vsa")
        with self.assertRaisesRegex(ValueError, "missing gate projection"):
            MODULE.resolve_attention(Store(missing=17), "auto")

    def test_vsa_adds_input_major_gate_weights_scales_and_markers(self):
        class Template:
            header = {
                "blocks.0.attn.qkv_proj.weight": {
                    "dtype": "I8",
                    "shape": [5376, 21504],
                },
                "blocks.0.attn.qkv_proj.weight_scale": {
                    "dtype": "F32",
                    "shape": [21504, 1],
                },
                "blocks.0.attn.qkv_proj.comfy_quant": {
                    "dtype": "U8",
                    "shape": [72],
                },
            }

        specifications = MODULE.vsa_tensor_specifications(Template())
        self.assertEqual(len(specifications), 150)
        self.assertEqual(
            specifications["blocks.49.attn.vsa_gate.weight"],
            {"dtype": "I8", "shape": [5376, 7168]},
        )
        self.assertEqual(
            specifications["blocks.49.attn.vsa_gate.weight_scale"],
            {"dtype": "F32", "shape": [7168, 1]},
        )
        self.assertEqual(
            specifications["blocks.49.attn.vsa_gate.comfy_quant"],
            {"dtype": "U8", "shape": [72]},
        )

    def test_vsa_contract_offsets_follow_payload_emission_order(self):
        specifications = MODULE.precomputed_tensor_specifications("vsa", 7)
        names = list(specifications)
        self.assertEqual(
            names[:6],
            [
                "h3.fasth3.version",
                "h3.fasth3.steps",
                "h3.fasth3.times",
                "h3.fasth3.vsa.tile_size",
                "h3.fasth3.vsa.sparsity",
                "h3.fasth3.blocks.0.adaln",
            ],
        )
        self.assertEqual(
            specifications["h3.fasth3.vsa.tile_size"],
            {"dtype": "U32", "shape": [1]},
        )
        self.assertEqual(
            specifications["h3.fasth3.vsa.sparsity"],
            {"dtype": "F32", "shape": [1]},
        )

    def test_four_step_schedule_has_the_native_seven_row_order(self):
        rows = MODULE.fasth3_time_rows()
        expected = np.asarray(
            [0.0, 1 / 37, 0.1, 1 / 13, 0.25, 0.2, 0.5], dtype=np.float32
        )
        np.testing.assert_allclose(rows, expected, rtol=0, atol=2e-7)

    def test_fastvideo_checkpoint_names_map_to_fused_h3_names(self):
        self.assertEqual(
            MODULE.source_keys("blocks.2.attn.qkv_proj.weight"),
            [
                "transformer_blocks.2.attn.to_q.weight",
                "transformer_blocks.2.attn.to_k.weight",
                "transformer_blocks.2.attn.to_v.weight",
            ],
        )
        self.assertEqual(
            MODULE.source_keys("token_refiner.blocks.1.attn.q_norm.weight"),
            ["token_refiner.refiner_blocks.1.attn.norm_q.weight"],
        )

    def test_diffusers_swiglu_halves_are_swapped_for_h3(self):
        class Store:
            def f32(self, name):
                self.name = name
                return np.arange(8, dtype=np.float32).reshape(4, 2)

        store = Store()
        actual = MODULE.source_f32(store, "blocks.2.mlp.fc1.weight")
        self.assertEqual(store.name, "transformer_blocks.2.ff.net.0.proj.weight")
        np.testing.assert_array_equal(
            actual,
            np.asarray([[4, 5], [6, 7], [0, 1], [2, 3]], dtype=np.float32),
        )

        refiner = MODULE.source_f32(
            store, "token_refiner.blocks.1.mlp.fc1.weight"
        )
        self.assertEqual(
            store.name, "token_refiner.refiner_blocks.1.ff.net.0.proj.weight"
        )
        np.testing.assert_array_equal(refiner, actual)

    def test_chunked_projection_matches_one_matrix_product(self):
        weight = np.arange(30, dtype=np.float32).reshape(6, 5) / 10
        bias = np.arange(6, dtype=np.float32) / 7
        inputs = np.arange(15, dtype=np.float32).reshape(3, 5) / 4
        store = FakeStore(weight, bias)
        expected = inputs @ weight.T + bias
        actual = MODULE.chunked_projection(
            store, "w", "b", inputs, chunk_rows=2
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_bf16_rounding_is_stable(self):
        values = np.asarray([0.0, 0.1, -3.25, 1000.125], dtype=np.float32)
        once = MODULE.bf16_round(values)
        twice = MODULE.bf16_round(once)
        np.testing.assert_array_equal(once, twice)

    def test_source_validation_uses_the_released_time_mlp_shapes(self):
        class ShapeOnlyStore:
            def description(self, name):
                shapes = {
                    "time_embedder.linear_1.weight": [5376, 256],
                    "time_embedder.linear_1.bias": [5376],
                    "time_embedder.linear_2.weight": [2688, 5376],
                    "time_embedder.linear_2.bias": [2688],
                    "norm_out.linear.weight": [10752, 2688],
                    "norm_out.linear.bias": [10752],
                }
                if ".adaln_proj.linear.weight" in name:
                    shape = [96768, 2688]
                elif ".adaln_proj.linear.bias" in name:
                    shape = [96768]
                else:
                    shape = shapes[name]
                return {"shape": shape, "dtype": "BF16"}

        class EmptyTemplate:
            header = {}

        MODULE.validate_source(ShapeOnlyStore(), EmptyTemplate(), set())


if __name__ == "__main__":
    unittest.main()
