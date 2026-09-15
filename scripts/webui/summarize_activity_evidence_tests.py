#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import importlib.util, json, stat, sys, tempfile, unittest
from pathlib import Path
MODULE = Path(__file__).with_name("summarize_activity_evidence.py")
spec = importlib.util.spec_from_file_location("summarize_activity_evidence", MODULE)
assert spec and spec.loader
summarize = importlib.util.module_from_spec(spec)
sys.modules["summarize_activity_evidence"] = summarize
spec.loader.exec_module(summarize)

def full_evidence() -> dict[str, object]:
    return {
        "status": "passed",
        "source_commit": "a" * 40,
        "server_bin_sha256": "b" * 64,
        "features": ["cuda", "webui"],
        "work_dir": "/secret/runner/path",
        "display": {"value": ":123", "virtual": True, "minimum_geometry": "1600x1100x24"},
        "checkpoint": {
            "path": "/secret/model/path",
            "display_name": "qwen3-0.6b-4bit",
            "revision": "ci-fixtures/qwen3-0.6b-4bit-v1",
            "file_count": 4,
            "total_bytes": 123,
            "safetensors_count": 1,
            "safetensors_total_bytes": 100,
            "metadata_sha256": {"config.json": "c" * 64},
            "safetensors_sha256": [{"relative_path": "weights.safetensors", "size": 100, "sha256": "d" * 64}],
        },
        "activity_performance": [{
            "output": "/secret/activity.json",
            "log": "/secret/activity.log",
            "output_sha256": "e" * 64,
            "summary": {
                "status": "within-target",
                "hidden_native": "measured",
                "summaries": [
                    {"mode": mode, "status": "within-target", "paired_runs": 5, "median_decode_degradation_percent": 1.0, "baseline_cv_percent": 1.0, "paired_range_percent": [0.0, 1.0]}
                    for mode in ("one-visible", "two-visible", "hidden")
                ],
            },
        }],
        "server_shutdown": {"exit_code": 0, "forced": False, "process_group_empty": True, "log": "/secret/server.log"},
    }

class SummarizeActivityEvidenceTests(unittest.TestCase):
    def test_summary_allowlist_excludes_paths_and_keeps_native_hidden_status(self) -> None:
        result = summarize.build_summary(full_evidence())
        encoded = json.dumps(result, sort_keys=True)
        self.assertEqual(result["activity_performance"]["hidden_native"], "measured")
        self.assertEqual(result["activity_performance"]["status"], "within-target")
        self.assertEqual(result["activity_performance"]["output_sha256"], "e" * 64)
        self.assertNotIn("/secret", encoded)
        self.assertNotIn('"work_dir"', encoded)
        self.assertNotIn('"path"', encoded)
        self.assertNotIn('"log"', encoded)

    def test_summary_rejects_nonpassing_or_non_native_hidden_input(self) -> None:
        data = full_evidence()
        data["status"] = "failed"
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)
        data = full_evidence()
        data["activity_performance"][0]["summary"]["hidden_native"] = "not-run"  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)

    def test_summary_rejects_malicious_nested_checkpoint_paths_and_bad_hashes(self) -> None:
        data = full_evidence()
        data["checkpoint"]["metadata_sha256"] = {"/abs/config.json": "c" * 64}  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)
        data = full_evidence()
        data["checkpoint"]["safetensors_sha256"][0]["relative_path"] = "../weights.safetensors"  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)
        data = full_evidence()
        data["checkpoint"]["safetensors_sha256"][0]["sha256"] = "/not/a/hash"  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)

    def test_summary_rejects_missing_modes_nan_and_bad_shutdown(self) -> None:
        data = full_evidence()
        data["activity_performance"][0]["summary"]["summaries"] = data["activity_performance"][0]["summary"]["summaries"][:2]  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)
        data = full_evidence()
        data["activity_performance"][0]["summary"]["summaries"][0]["baseline_cv_percent"] = float("nan")  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)
        data = full_evidence()
        data["server_shutdown"]["forced"] = True  # type: ignore[index]
        with self.assertRaises(AssertionError):
            summarize.build_summary(data)

    def test_output_is_exclusive_0600_and_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.json"
            summarize.write_private_json(out, {"status": "passed"})
            self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                summarize.write_private_json(out, {"status": "again"})
            link = Path(tmp) / "link.json"
            link.symlink_to(out)
            with self.assertRaises(AssertionError):
                summarize.write_private_json(link, {"status": "bad"})

if __name__ == "__main__":
    unittest.main()
