#!/usr/bin/env python3
"""Adversarial, model-free tests for the LLaVA capture comparator."""

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
import llava_reference_oracle as oracle


class ComparatorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.actual = self.root / "actual"
        self.reference.mkdir()
        self.actual.mkdir()
        self.reference_manifest = self.manifest("reference")
        self.actual_manifest = self.manifest("actual")
        self.write_arrays(self.reference, self.reference_manifest)
        self.write_arrays(self.actual, self.actual_manifest)
        self.write_manifests()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def manifest(role: str) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema": 1,
            "producer": role,
            "image_fixture": {
                "path": oracle.FIXTURE_PATH,
                "sha256": oracle.FIXTURE_SHA256,
                "two_image_transform": "horizontal_mirror",
            },
            "converted_checkpoint": {
                "repo": oracle.CONVERTED_REPO,
                "revision": oracle.CONVERTED_REVISION,
                "artifact_manifest": {
                    "canonical_sha256": oracle.CONVERTED_ARTIFACT_MANIFEST_SHA256,
                    "files": oracle.CONVERTED_ARTIFACTS,
                },
            },
            "cases": [],
        }
        if role == "reference":
            manifest["source"] = {
                "repo": oracle.SOURCE_REPO,
                "revision": oracle.SOURCE_REVISION,
                "artifact_manifest": {
                    "canonical_sha256": oracle.SOURCE_ARTIFACT_MANIFEST_SHA256,
                    "files": oracle.SOURCE_ARTIFACTS,
                },
            }
        else:
            manifest["negative_cases"] = copy.deepcopy(
                oracle.EXPECTED_NEGATIVE_CASES
            )
        for case_name, stages in oracle.CASE_REQUIRED_STAGES.items():
            arrays = {}
            for stage in stages:
                dtype = "int32" if stage in oracle.INTEGER_STAGES else "float32"
                arrays[stage] = {
                    "file": f"{case_name}.{stage}.bin",
                    "dtype": dtype,
                    "shape": [1],
                }
            manifest["cases"].append(
                {
                    "name": case_name,
                    "image_count": len(oracle.CASE_IMAGE_TRANSFORMS[case_name]),
                    "image_transforms": list(
                        oracle.CASE_IMAGE_TRANSFORMS[case_name]
                    ),
                    "arrays": arrays,
                }
            )
        return manifest

    @staticmethod
    def write_arrays(root: Path, manifest: dict[str, Any]) -> None:
        for case in manifest["cases"]:
            for spec in case["arrays"].values():
                np.zeros(spec["shape"], dtype=spec["dtype"]).tofile(
                    root / spec["file"]
                )

    def write_manifests(self) -> None:
        (self.reference / "manifest.json").write_text(
            json.dumps(self.reference_manifest), encoding="utf-8"
        )
        (self.actual / "manifest.json").write_text(
            json.dumps(self.actual_manifest), encoding="utf-8"
        )

    def assert_rejected(self, contains: str) -> None:
        report = oracle.compare_capture_roots(self.reference, self.actual)
        self.assertFalse(report["passed"], report)
        self.assertIn(contains, report.get("error", "").lower())

    def test_valid_capture_passes(self) -> None:
        self.assertTrue(
            oracle.compare_capture_roots(self.reference, self.actual)["passed"]
        )

    def test_missing_required_stage_from_both_is_rejected(self) -> None:
        for manifest in (self.reference_manifest, self.actual_manifest):
            del manifest["cases"][0]["arrays"]["selected_kv"]
        self.write_manifests()
        self.assert_rejected("stage set differs")

    def test_extra_and_duplicate_cases_are_rejected(self) -> None:
        baseline = copy.deepcopy(self.actual_manifest)
        with self.subTest("extra"):
            self.actual_manifest["cases"].append(
                copy.deepcopy(self.actual_manifest["cases"][0])
                | {"name": "unexpected"}
            )
            self.write_manifests()
            self.assert_rejected("case set differs")
        self.actual_manifest = copy.deepcopy(baseline)
        with self.subTest("duplicate"):
            self.actual_manifest["cases"].append(
                copy.deepcopy(self.actual_manifest["cases"][0])
            )
            self.write_manifests()
            self.assert_rejected("duplicate case")

    def test_extra_and_duplicate_stages_are_rejected(self) -> None:
        baseline = copy.deepcopy(self.actual_manifest)
        with self.subTest("extra"):
            self.actual_manifest["cases"][0]["arrays"]["unexpected"] = {
                "file": "image_text.unexpected.bin",
                "dtype": "float32",
                "shape": [1],
            }
            np.zeros([1], dtype=np.float32).tofile(
                self.actual / "image_text.unexpected.bin"
            )
            self.write_manifests()
            self.assert_rejected("stage set differs")
        self.actual_manifest = copy.deepcopy(baseline)
        self.write_manifests()
        with self.subTest("duplicate-json-key"):
            path = self.actual / "manifest.json"
            text = path.read_text(encoding="utf-8")
            injection = (
                '"arrays": {"selected_kv": '
                '{"file": "image_text.selected_kv.bin", '
                '"dtype": "float32", "shape": [1]},'
            )
            path.write_text(text.replace('"arrays": {', injection, 1), encoding="utf-8")
            self.assert_rejected("duplicate json key")

    def test_non_finite_arrays_are_rejected(self) -> None:
        path = self.actual / "image_text.first_prefill_logits.bin"
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                np.asarray([value], dtype=np.float32).tofile(path)
                report = oracle.compare_capture_roots(self.reference, self.actual)
                self.assertFalse(report["passed"], report)
                stage = next(
                    stage
                    for case in report["cases"]
                    if case["name"] == "image_text"
                    for stage in case["stages"]
                    if stage["stage"] == "first_prefill_logits"
                )
                self.assertEqual(stage.get("error"), "non-finite array value")

    def test_dtype_mismatch_is_rejected(self) -> None:
        self.actual_manifest["cases"][0]["arrays"]["first_prefill_logits"][
            "dtype"
        ] = "int32"
        self.write_manifests()
        self.assert_rejected("dtype must be float32")

    def test_invalid_shape_and_path_are_rejected(self) -> None:
        baseline = copy.deepcopy(self.actual_manifest)
        with self.subTest("shape"):
            self.actual_manifest["cases"][0]["arrays"]["selected_kv"]["shape"] = [0]
            self.write_manifests()
            self.assert_rejected("invalid shape")
        self.actual_manifest = copy.deepcopy(baseline)
        with self.subTest("path"):
            self.actual_manifest["cases"][0]["arrays"]["selected_kv"][
                "file"
            ] = "../selected_kv.bin"
            self.write_manifests()
            self.assert_rejected("invalid array path")

    def test_truncated_and_oversized_binaries_are_rejected(self) -> None:
        path = self.actual / "image_text.selected_kv.bin"
        with self.subTest("truncated"):
            path.write_bytes(b"\x00\x00")
            self.assert_rejected("binary size differs")
        np.zeros([1], dtype=np.float32).tofile(path)
        with self.subTest("oversized"):
            path.write_bytes(path.read_bytes() + b"\x00")
            self.assert_rejected("binary size differs")

    def test_negative_case_names_and_outcomes_are_exact(self) -> None:
        mutations = (
            lambda value: value.pop("context_overflow"),
            lambda value: value["malformed_placeholder"].update(passed=False),
            lambda value: value.update(
                unexpected={
                    "passed": True,
                    "outcome": "rejected",
                    "category": "unexpected",
                }
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.actual_manifest["negative_cases"] = copy.deepcopy(
                    oracle.EXPECTED_NEGATIVE_CASES
                )
                mutate(self.actual_manifest["negative_cases"])
                self.write_manifests()
                self.assert_rejected("negative_cases")


if __name__ == "__main__":
    unittest.main()
