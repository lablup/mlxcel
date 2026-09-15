#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("verify_activity_performance.py")
spec = importlib.util.spec_from_file_location("verify_activity_performance", MODULE_PATH)
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
sys.modules["verify_activity_performance"] = verify
spec.loader.exec_module(verify)


def valid_activity_output() -> dict[str, object]:
    samples = [{"mode": "warmup", "tokens_per_second": 1.0, "predicted_tokens": 1, "predicted_ms": 1.0}]
    for mode, count in {"off": 15, "one-visible": 5, "two-visible": 5, "hidden": 5}.items():
        samples.extend({"mode": mode, "tokens_per_second": 1.0, "predicted_tokens": 1, "predicted_ms": 1.0} for _ in range(count))
    geom = {"innerWidth": 700, "innerHeight": 900, "visualWidth": 700, "visualHeight": 900}
    preflight = []
    for mode, count in {"one-visible": 6, "two-visible": 12, "hidden": 6}.items():
        preflight.extend({"mode": mode, "geometry": geom} for _ in range(count))
    return {
        "status": "within-target",
        "hidden_native": "measured",
        "preflight": preflight,
        "samples": samples,
        "summaries": [
            {"mode": mode, "paired_runs": 5, "median_decode_degradation_percent": 1.0, "baseline_cv_percent": 1.0, "paired_range_percent": [0.0, 1.0], "status": "within-target"}
            for mode in ("one-visible", "two-visible", "hidden")
        ],
    }


class ActivityPerformanceHelperTests(unittest.TestCase):
    def test_validate_activity_output_requires_full_native_hidden_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps(valid_activity_output()))
            self.assertEqual(verify.validate_activity_output(path)["hidden_native"], "measured")

    def test_validate_activity_output_rejects_incomplete_or_investigate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps({"status": "incomplete", "hidden_native": "not-run", "preflight": [], "summaries": []}))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            path.write_text(json.dumps({"status": "within-target", "hidden_native": "measured", "preflight": [{}], "summaries": [{"mode": "hidden", "status": "investigate"}]}))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)

    def test_write_private_uses_0600(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            verify.write_private(path, "secret\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_text(), "secret\n")
            log = Path(tmp) / "log"
            with verify.open_private_append(log) as fp:
                fp.write(b"line\n")
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)

    def test_unique_work_dir_creates_private_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            first = verify.unique_work_dir(parent)
            second = verify.unique_work_dir(parent)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, parent)
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o700)

    def test_redact_hides_bearer_and_explicit_secret(self) -> None:
        value = {"log": "Authorization: Bearer secret-token", "path": "contains explicit"}
        redacted = verify.redact(value, ["explicit"])
        self.assertNotIn("secret-token", json.dumps(redacted))
        self.assertNotIn("explicit", json.dumps(redacted))

    def test_parse_args_requires_build_sha_and_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "server"
            server.write_text("#!/bin/sh\n")
            server.chmod(0o700)
            model = root / "model"
            model.mkdir()
            args = verify.parse_args([
                "--server-bin", str(server),
                "--model", str(model),
                "--checkpoint-revision", "rev",
                "--source-sha", "a" * 40,
                "--features", "cuda,webui",
                "--evidence", str(root / "evidence.json"),
                "--port", "53001",
            ])
            self.assertEqual(args.features, ["cuda", "webui"])
            self.assertEqual(args.port, 53001)
            nonexec = root / "nonexec"
            nonexec.write_text("#!/bin/sh\n")
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    verify.parse_args([
                        "--server-bin", str(nonexec),
                        "--model", str(model),
                        "--checkpoint-revision", "rev",
                        "--source-sha", "a" * 40,
                        "--features", "cuda",
                        "--evidence", str(root / "evidence.json"),
                        "--port", "53001",
                    ])
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    verify.parse_args([
                        "--server-bin", str(server),
                        "--model", str(model),
                        "--checkpoint-revision", "rev",
                        "--source-sha", "bad",
                        "--features", "cuda",
                        "--evidence", str(root / "evidence.json"),
                        "--port", "53001",
                    ])

    def test_catalog_pages_quotes_cursor_and_stops_on_null(self) -> None:
        calls = []
        def fake_json(url, *, key=None):  # type: ignore[no-untyped-def]
            calls.append(url)
            if len(calls) == 1:
                return {"items": [], "pagination": {"next_cursor": "cursor with space/slash"}}
            return {"items": [], "pagination": {"next_cursor": None}}
        with mock.patch.object(verify, "json_request", side_effect=fake_json):
            pages = verify.catalog_pages("http://127.0.0.1/lab", "key")
        self.assertEqual(len(pages), 2)
        self.assertIn("cursor=cursor%20with%20space/slash", calls[1])

    def test_check_require_models_fails_closed(self) -> None:
        old = os.environ.get("MLXCEL_REQUIRE_MODELS")
        try:
            os.environ.pop("MLXCEL_REQUIRE_MODELS", None)
            with self.assertRaises(SystemExit):
                verify.check_require_models()
            os.environ["MLXCEL_REQUIRE_MODELS"] = "1"
            verify.check_require_models()
        finally:
            if old is None:
                os.environ.pop("MLXCEL_REQUIRE_MODELS", None)
            else:
                os.environ["MLXCEL_REQUIRE_MODELS"] = old

    def test_clean_env_preserves_explicit_playwright_cache_only(self) -> None:
        old = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        try:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/pw-cache"
            env = verify.clean_env(Path("/home"), Path("/store"), ":99")
            self.assertEqual(env["PLAYWRIGHT_BROWSERS_PATH"], "/pw-cache")
            self.assertEqual(env["HOME"], "/home")
            self.assertEqual(env["DISPLAY"], ":99")
        finally:
            if old is None:
                os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            else:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old

    def test_summarize_checkpoint_records_metadata_and_weight_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "weights.safetensors").write_bytes(b"not-real")
            summary = verify.summarize_checkpoint(model, "rev123")
            self.assertEqual(summary["revision"], "rev123")
            self.assertEqual(summary["safetensors_count"], 1)
            self.assertEqual(len(summary["safetensors_sha256"]), 1)
            self.assertIn("config.json", summary["metadata_sha256"])
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(AssertionError):
                verify.summarize_checkpoint(empty, "rev123")

    def test_start_virtual_display_requires_tools_without_autoinstall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(verify.shutil, "which", return_value=None):
                with self.assertRaises(AssertionError):
                    verify.start_virtual_display(Path(tmp))

    def test_materialize_model_view_uses_private_reflink_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source-model"
            source.mkdir()
            work = root / "work"
            work.mkdir()
            calls = []
            def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
                calls.append(command)
                Path(command[-1]).mkdir(parents=True)
                return Namespace(returncode=0, stderr="")
            with mock.patch.object(verify.subprocess, "run", side_effect=fake_run):
                dest, copy_kind = verify.materialize_model_view(source, work)
            self.assertEqual(dest, work / "models" / "source-model")
            self.assertFalse(dest.is_symlink())
            self.assertTrue(copy_kind.endswith("-deref"))
            self.assertTrue(any("RL" in item for item in calls[0]))
            if verify.sys.platform != "darwin":
                self.assertIn("--reflink=auto", calls[0])

    def test_select_entry_requires_unique_model_or_explicit_id(self) -> None:
        entry = {"identity": {"id": "mdl_" + "A" * 43, "display_name": "m", "inference_id": "m", "revision": 1}}
        self.assertEqual(verify.select_entry([entry], Path("/tmp/m"), None), entry)
        with self.assertRaises(AssertionError):
            verify.select_entry([entry, entry], Path("/tmp/m"), None)
        with self.assertRaises(AssertionError):
            verify.select_entry([entry], Path("/tmp/m"), "bad")

    def test_stop_server_strict_rejects_sigint_exit_code(self) -> None:
        class Proc:
            returncode = -verify.signal.SIGINT
            def poll(self):
                return self.returncode
        owned = verify.OwnedProcess("server", Proc(), Path("server.log"))
        with self.assertRaises(RuntimeError):
            verify.stop_server_strict(owned)

    def test_validate_activity_output_requires_all_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps({"status": "within-target", "hidden_native": "measured", "preflight": [{}], "summaries": [{"mode": "hidden", "status": "within-target"}]}))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)

    def test_validate_activity_output_rejects_nonfinite_or_wrong_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            data = valid_activity_output()
            data["summaries"][0]["baseline_cv_percent"] = float("inf")  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            data = valid_activity_output()
            data["samples"] = data["samples"][:-1]  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            data = valid_activity_output()
            data["summaries"][0]["baseline_cv_percent"] = -0.1  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            data = valid_activity_output()
            data["summaries"][0]["paired_range_percent"] = [1.0, 0.0]  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            data = valid_activity_output()
            data["summaries"][1]["mode"] = data["summaries"][0]["mode"]  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)


if __name__ == "__main__":
    unittest.main()
