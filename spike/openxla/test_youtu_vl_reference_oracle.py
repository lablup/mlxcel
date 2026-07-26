#!/usr/bin/env python3
"""Model-free adversarial tests for the Youtu-VL oracle contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import youtu_vl_reference_oracle as oracle


class YoutuVlOracleContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.actual = self.root / "actual"
        self.reference.mkdir()
        self.actual.mkdir()
        self.reference_manifest = self.make_manifest("hf-transformers-eager")
        self.actual_manifest = self.make_manifest("mlxcel-xla-diagnostics")
        self.write_capture(self.reference, self.reference_manifest)
        self.write_capture(self.actual, self.actual_manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def make_manifest(producer: str) -> dict[str, Any]:
        sequence = oracle.MERGED_TOKENS + 4
        arrays = {}
        for stage, shape in oracle.expected_shapes(sequence).items():
            arrays[stage] = {
                "file": f"image_text.{stage}.bin",
                "dtype": (
                    "int32" if stage in oracle.INTEGER_STAGES else "float32"
                ),
                "shape": shape,
                "sha256": "",
            }
        return {
            "schema": oracle.SCHEMA_VERSION,
            "contract": oracle.CONTRACT,
            "producer": producer,
            "checkpoint": {
                "repo": oracle.CHECKPOINT_REPO,
                "revision": oracle.CHECKPOINT_REVISION,
                "artifact_manifest": {
                    "canonical_sha256": (
                        oracle.CHECKPOINT_ARTIFACT_MANIFEST_SHA256
                    ),
                    "files": oracle.CHECKPOINT_ARTIFACTS,
                },
            },
            "fixture": {
                "path": oracle.FIXTURE_PATH,
                "sha256": oracle.FIXTURE_SHA256,
            },
            "architecture": {
                "vision_depth": oracle.VISION_DEPTH,
                "vision_hidden": oracle.VISION_HIDDEN,
                "selected_vision_layers": [
                    {"index": index, "attention": kind}
                    for index, kind in oracle.SELECTED_VISION_LAYERS
                ],
                "text_layers": oracle.TEXT_LAYERS,
                "text_hidden": oracle.TEXT_HIDDEN,
                "vocab": oracle.VOCAB,
                "kv_lora_rank": oracle.KV_LORA_RANK,
                "qk_rope_head_dim": oracle.ROPE_WIDTH,
            },
            "generation": {
                "mode": "greedy",
                "max_new_tokens": oracle.MAX_NEW_TOKENS,
            },
            "lifecycle": copy.deepcopy(oracle.LIFECYCLE),
            "case": {
                "name": "image_text",
                "prompt": oracle.PROMPT,
                "unexpanded_input_ids": [oracle.IMAGE_TOKEN_ID, 42, 43],
                "arrays": arrays,
            },
        }

    @staticmethod
    def array(stage: str, shape: list[int], dtype: str) -> np.ndarray:
        value = np.zeros(shape, dtype=dtype)
        if stage == "expanded_input_ids":
            value[:] = np.arange(value.size, dtype=np.int32) + 1
            value[: oracle.MERGED_TOKENS] = oracle.IMAGE_TOKEN_ID
        elif stage == "placeholder_positions":
            value[:] = np.arange(oracle.MERGED_TOKENS, dtype=np.int32)
        elif stage == "spatial_shapes":
            value[:] = [[16, 16]]
        elif stage in {"window_group_index", "reverse_group_index"}:
            value[:] = np.arange(oracle.MERGED_TOKENS, dtype=np.int32)
        elif stage in {"window_cu_seqlens", "full_cu_seqlens"}:
            value[:] = [0, oracle.PATCHES]
        return value

    def write_capture(self, root: Path, manifest: dict[str, Any]) -> None:
        for stage, spec in manifest["case"]["arrays"].items():
            value = self.array(stage, spec["shape"], spec["dtype"])
            path = root / spec["file"]
            value.tofile(path)
            spec["sha256"] = oracle.sha256(path)
        oracle.write_manifest(root, manifest)

    def rewrite(self, root: Path, manifest: dict[str, Any]) -> None:
        oracle.write_manifest(root, manifest)

    def assert_contract_error(self, contains: str) -> None:
        report = oracle.compare_capture_roots(self.reference, self.actual)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["first_divergence"]["stage"], "contract")
        self.assertIn(contains, report["error"].lower())

    def test_valid_synthetic_capture_passes(self) -> None:
        report = oracle.compare_capture_roots(self.reference, self.actual)
        self.assertTrue(report["passed"], report)
        self.assertIsNone(report["first_divergence"])

    def test_checkpoint_identity_and_canonical_hash_are_pinned(self) -> None:
        self.actual_manifest["checkpoint"]["revision"] = "main"
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("checkpoint identity")
        self.actual_manifest = self.make_manifest("mlxcel-xla-diagnostics")
        self.write_capture(self.actual, self.actual_manifest)
        self.actual_manifest["checkpoint"]["artifact_manifest"][
            "canonical_sha256"
        ] = "0" * 64
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("checkpoint identity")

    def test_every_array_is_content_addressed(self) -> None:
        path = self.actual / "image_text.prefill_logits.bin"
        with path.open("r+b") as stream:
            stream.write(np.asarray([1.0], dtype=np.float32).tobytes())
        self.assert_contract_error("artifact hash")

    def test_manifest_hash_is_required(self) -> None:
        (self.actual / "manifest.sha256").write_text("0" * 64 + "\n")
        self.assert_contract_error("manifest.sha256")

    def test_unlisted_capture_artifact_is_rejected(self) -> None:
        (self.actual / "untracked.csv").write_text("not,an,oracle\n")
        self.assert_contract_error("file set differs")

    def test_missing_selected_layer_is_rejected(self) -> None:
        del self.actual_manifest["case"]["arrays"]["layer.23.full"]
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("stage set differs")

    def test_actual_27_layer_schedule_identity_is_required(self) -> None:
        self.actual_manifest["architecture"]["vision_depth"] = 2
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("architecture contract")

    def test_lifecycle_events_are_closed_and_ordered(self) -> None:
        self.actual_manifest["lifecycle"]["events"][3] = "reuse_without_reset"
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("lifecycle contract")

    def test_reuse_must_equal_fresh_capture(self) -> None:
        stage = "reuse_greedy_tokens"
        spec = self.actual_manifest["case"]["arrays"][stage]
        path = self.actual / spec["file"]
        value = np.fromfile(path, dtype=np.int32)
        value[0] = 99
        value.tofile(path)
        spec["sha256"] = oracle.sha256(path)
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("lifecycle reuse differs")

    def test_window_permutation_and_inverse_are_validated(self) -> None:
        stage = "window_group_index"
        spec = self.actual_manifest["case"]["arrays"][stage]
        path = self.actual / spec["file"]
        value = np.fromfile(path, dtype=np.int32)
        value[-1] = value[-2]
        value.tofile(path)
        spec["sha256"] = oracle.sha256(path)
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("not a permutation")

    def test_placeholder_selection_is_validated(self) -> None:
        stage = "placeholder_positions"
        spec = self.actual_manifest["case"]["arrays"][stage]
        path = self.actual / spec["file"]
        value = np.fromfile(path, dtype=np.int32)
        value[-1] += 1
        value.tofile(path)
        spec["sha256"] = oracle.sha256(path)
        self.rewrite(self.actual, self.actual_manifest)
        self.assert_contract_error("placeholder selection")

    def test_capture_output_is_immutable(self) -> None:
        with self.assertRaisesRegex(oracle.ContractError, "already exists"):
            oracle.ensure_new_output(self.actual)

    def test_generator_has_no_mlxcel_dependency(self) -> None:
        source = Path(oracle.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import mlxcel", source)
        self.assertNotIn("from mlxcel", source)


if __name__ == "__main__":
    unittest.main()
