#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Pure-Python unit tests for verify_installed_artifact.py."""
from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("verify_installed_artifact.py")
spec = importlib.util.spec_from_file_location("verify_installed_artifact", MODULE_PATH)
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
sys.modules["verify_installed_artifact"] = verify
spec.loader.exec_module(verify)


class InstalledArtifactHelperTests(unittest.TestCase):
    def test_redact_hides_explicit_secret_and_bearer_values(self) -> None:
        value = {"log": "Authorization: Bearer abc.def-ghi", "nested": ["token secret-value"]}
        self.assertEqual(
            verify.redact(value, ["secret-value"]),
            {"log": "Authorization: Bearer <redacted>", "nested": ["token <redacted>"]},
        )

    def test_clean_env_strips_proxy_and_secret_like_inputs(self) -> None:
        old = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update({"PATH": "/bin", "HTTPS_PROXY": "http://proxy.invalid", "http_proxy": "http://proxy.invalid", "LLAMA_API_KEY": "leak", "DYLD_LIBRARY_PATH": "/mlx"})
            with tempfile.TemporaryDirectory() as tmp:
                env = verify.clean_env(Path(tmp) / "home", Path(tmp) / "store")
            self.assertEqual(env["PATH"], "/bin")
            self.assertEqual(env["DYLD_LIBRARY_PATH"], "/mlx")
            self.assertNotIn("HTTPS_PROXY", env)
            self.assertNotIn("http_proxy", env)
            self.assertNotIn("LLAMA_API_KEY", env)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["NO_PROXY"], "*")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_write_private_uses_0600_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            verify.write_private(path, "secret\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_text(), "secret\n")

    def test_harness_flush_redacts_durable_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_path = Path(tmp) / "evidence.json"
            h = verify.Harness(Namespace(evidence=str(evidence_path)), Path(tmp), {"result": "fail"}, ["secret-token"])
            h.evidence["error"] = {"message": "secret-token Authorization: Bearer abc123"}
            h.flush()
            text = evidence_path.read_text()
            self.assertIn("<redacted>", text)
            self.assertNotIn("secret-token", text)
            self.assertNotIn("abc123", text)

    def test_network_denial_does_not_treat_reachable_network_as_pass(self) -> None:
        class Reachable:
            def open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                class Response:
                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):  # type: ignore[no-untyped-def]
                        return False

                    def read(self, size: int) -> bytes:
                        return b"x"

                return Response()

        old_active = os.environ.get("MLXCEL_WEBUI_NETNS_ACTIVE")
        old_opener = verify.NO_PROXY_OPENER
        try:
            os.environ["MLXCEL_WEBUI_NETNS_ACTIVE"] = "1"
            verify.NO_PROXY_OPENER = Reachable()
            with tempfile.TemporaryDirectory() as tmp:
                h = verify.Harness(Namespace(evidence=str(Path(tmp) / "evidence.json")), Path(tmp), {"result": "fail"}, [])
                with self.assertRaises(AssertionError):
                    verify.network_denial(h)
        finally:
            verify.NO_PROXY_OPENER = old_opener
            if old_active is None:
                os.environ.pop("MLXCEL_WEBUI_NETNS_ACTIVE", None)
            else:
                os.environ["MLXCEL_WEBUI_NETNS_ACTIVE"] = old_active


if __name__ == "__main__":
    unittest.main()
