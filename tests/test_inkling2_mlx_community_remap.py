# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the pure naming and quantization rules of
scripts/inkling2_mlx_community_remap.py (issue #3378). The tensor work needs
mlx and the real export; the name table and the mode inference do not.

Run with:
    python3 -m unittest tests/test_inkling2_mlx_community_remap.py
"""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inkling2_mlx_community_remap.py"

_spec = importlib.util.spec_from_file_location("inkling2_mlx_community_remap", SCRIPT)
remap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remap)

P = "language_model.model."


class FakeArray:
    def __init__(self, *shape):
        self.shape = shape


class MapKeyTests(unittest.TestCase):
    def test_plain_renames(self):
        cases = {
            "lm_head.weight": "language_model.lm_head.weight",
            "model.language_model.embed_tokens.scales": P + "embed_tokens.scales",
            "model.language_model.message_input.query": P + "message_queries",
            "model.language_model.layers.3.mlp.gate.weight": P + "layers.3.mlp.gate_weight",
            "model.language_model.layers.3.post_attention_convolution.conv1d.weight": P
            + "layers.3.attn_sconv.conv.weight",
            "model.language_model.layers.3.self_attn.q_l2_clip.scale": P
            + "layers.3.self_attn.q_scale",
            "model.language_model.layers.3.self_attn.q_proj.biases": P
            + "layers.3.self_attn.q_proj.biases",
            "model.vision_tower.merger.linear_1.weight": "vision_tower.proj_in.weight",
            "model.vision_tower.embeddings.position_embedding.weight": "vision_tower.embeddings.position_embedding",
        }
        for key, want in cases.items():
            self.assertEqual(remap.map_key(key), [(want, "copy")], key)

    def test_message_gates_become_controllers(self):
        got = remap.map_key("model.language_model.layers.0.q_message_gate.message_gate_proj.weight")
        self.assertEqual(got, [(P + "layers.0.controllers.attn.gain_weight", "copy")])
        got = remap.map_key("model.language_model.layers.7.mlp_message_gate.forget_gate_proj.weight")
        self.assertEqual(got, [(P + "layers.7.controllers.mlp.forget_weight", "copy")])

    def test_fused_experts_split_and_scale(self):
        base = P + "layers.2.mlp.switch_mlp."
        got = remap.map_key("model.language_model.layers.2.mlp.experts.gate_up_proj.weight")
        self.assertEqual(got, [(base + "gate_proj.weight", "split"), (base + "up_proj.weight", "split")])
        got = remap.map_key("model.language_model.layers.2.mlp.experts.gate_up_proj.global_scale")
        self.assertEqual(
            got, [(base + "gate_proj.weight_scale_2", "scale2"), (base + "up_proj.weight_scale_2", "scale2")]
        )
        got = remap.map_key("model.language_model.layers.2.mlp.experts.down_proj.scales")
        self.assertEqual(got, [(base + "down_proj.scales", "copy")])

    def test_unknown_tensor_is_refused(self):
        with self.assertRaises(ValueError):
            remap.map_key("model.other.weight")
        with self.assertRaises(ValueError):
            remap.map_key("model.language_model.layers.0.q_message_gate.foo_proj.weight")


class QuantizationTests(unittest.TestCase):
    def test_modes_from_plane_widths(self):
        self.assertEqual(remap.expert_quantization(288, 144)["mode"], "nvfp4")
        self.assertEqual(remap.expert_quantization(144, 72)["group_size"], 16)
        mxfp8 = remap.expert_quantization(576, 72)
        self.assertEqual((mxfp8["mode"], mxfp8["bits"], mxfp8["group_size"]), ("mxfp8", 8, 32))
        with self.assertRaises(ValueError):
            remap.expert_quantization(100, 30)

    def test_config_declares_each_expert_projection(self):
        weights = {}
        for layer, (w, s) in {0: (288, 144), 20: (576, 72)}.items():
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"{P}layers.{layer}.mlp.switch_mlp.{proj}"
                weights[name + ".weight"] = FakeArray(128, 2304, w)
                weights[name + ".scales"] = FakeArray(128, 2304, s)
                weights[name + ".weight_scale_2"] = FakeArray(128)
        config = remap.remap_config({"model_type": "x", "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}, weights)
        quant = config["quantization"]
        self.assertEqual(quant["mode"], "affine")
        self.assertEqual(quant[P + "layers.0.mlp.switch_mlp.up_proj"]["mode"], "nvfp4")
        self.assertEqual(quant[P + "layers.20.mlp.switch_mlp.down_proj"]["mode"], "mxfp8")
        self.assertEqual(len(quant), 3 + 6)
        self.assertEqual(config["quantization_config"], quant)


if __name__ == "__main__":
    unittest.main()
