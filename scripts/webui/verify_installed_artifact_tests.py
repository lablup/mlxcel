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
from unittest import mock
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

    def test_network_denial_rejects_external_interface_or_route(self) -> None:
        with mock.patch.object(verify, "network_interfaces", return_value=[{"name": "lo", "operstate": "up"}, {"name": "eth0", "operstate": "up"}]), mock.patch.object(verify, "default_routes", return_value=[]):
            with self.assertRaises(AssertionError):
                verify.assert_network_namespace_isolated()
        with mock.patch.object(verify, "network_interfaces", return_value=[{"name": "lo", "operstate": "up"}]), mock.patch.object(verify, "default_routes", return_value=["eth0 default"]):
            with self.assertRaises(AssertionError):
                verify.assert_network_namespace_isolated()

    def test_network_interfaces_read_the_namespace_scoped_proc_table(self) -> None:
        # Regression for the GB10 run where /sys/class/net reported host interfaces for a
        # process whose own namespace held only loopback, because sysfs stays bound to the
        # namespace that mounted it while /proc/net follows the reading task.
        table = (
            "Inter-|   Receive                                                |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop\n"
            "    lo:  123456     789    0    0    0     0          0         0   123456     789    0    0\n"
            "  eth0:       0       0    0    0    0     0          0         0        0       0    0    0\n"
        )
        self.assertEqual(verify.proc_net_dev_names(table), ["eth0", "lo"])
        self.assertEqual(verify.proc_net_dev_names(""), [])
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "dev"
            dev.write_text(table)
            with mock.patch.object(verify, "interface_flags", return_value=verify.IFF_UP):
                interfaces = verify.network_interfaces(dev)
            self.assertEqual([iface["name"] for iface in interfaces], ["eth0", "lo"])
            self.assertEqual({iface["source"] for iface in interfaces}, {"proc-net-namespace"})
            self.assertEqual({iface["operstate"] for iface in interfaces}, {"up"})
            with mock.patch.object(verify, "interface_flags", return_value=None):
                unknown = verify.network_interfaces(dev)
            self.assertEqual({iface["operstate"] for iface in unknown}, {"unknown"})
            with mock.patch.object(verify, "sysfs_interface_names", return_value=["lo"]):
                fallback = verify.network_interfaces(Path(tmp) / "absent")
            self.assertEqual(fallback, [{"name": "lo", "operstate": "unknown", "flags": "unknown", "source": "sysfs-mount-namespace"}])

    def test_network_denial_rejects_reachable_tcp_negative_control(self) -> None:
        class ReachableSocket:
            def __enter__(self):
                return self

            def __exit__(self, *exc):  # type: ignore[no-untyped-def]
                return False

            def settimeout(self, value: int) -> None:
                pass

            def connect(self, address):  # type: ignore[no-untyped-def]
                return None

        with mock.patch.object(verify, "network_interfaces", return_value=[{"name": "lo", "operstate": "up"}]), mock.patch.object(verify, "default_routes", return_value=[]), mock.patch.object(verify.socket, "socket", return_value=ReachableSocket()):
            with self.assertRaises(AssertionError):
                verify.assert_network_namespace_isolated()

    def test_network_denial_records_socket_oserror_as_denied(self) -> None:
        class DeniedSocket:
            def __enter__(self):
                return self

            def __exit__(self, *exc):  # type: ignore[no-untyped-def]
                return False

            def settimeout(self, value: int) -> None:
                pass

            def connect(self, address):  # type: ignore[no-untyped-def]
                raise OSError("network unreachable")

        with mock.patch.object(verify, "network_interfaces", return_value=[{"name": "lo", "operstate": "up"}]), mock.patch.object(verify, "default_routes", return_value=[]), mock.patch.object(verify.socket, "socket", return_value=DeniedSocket()):
            result = verify.assert_network_namespace_isolated()
        self.assertEqual(result["status"], "enforced")
        self.assertIn("external TCP connect denied", result["negative_control"])

    def test_shutdown_failure_still_deletes_private_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"
            verify.write_private(key, "secret\n")
            def failing_stopper(proc, log_file, log_path):  # type: ignore[no-untyped-def]
                raise RuntimeError("injected shutdown failure")
            with self.assertRaises(RuntimeError):
                verify.stop_proc_with_key_cleanup(object(), object(), Path(tmp) / "log", key, failing_stopper)
            self.assertFalse(key.exists())

    def test_feature_off_probe_failure_still_deletes_private_key(self) -> None:
        class Result:
            returncode = 0
            stdout = b"unexpected success"
        def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
            return Result()
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"
            verify.write_private(key, "secret\n")
            with mock.patch.object(verify, "free_port", return_value=31337):
                with self.assertRaises(AssertionError):
                    verify.run_feature_off_probe_with_key_cleanup(["fake"], Path(tmp), key, {}, runner)
            self.assertFalse(key.exists())

    def test_tls_cleanup_deletes_key_and_cert_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cert = Path(tmp) / "cert.pem"
            key = Path(tmp) / "key.pem"
            cert.write_text("cert")
            verify.write_private(key, "key")
            verify.cleanup_tls_material(["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)])
            self.assertFalse(cert.exists())
            self.assertFalse(key.exists())

    def test_tls_matrix_runs_both_installed_commands_before_cleanup(self) -> None:
        seen: list[str] = []
        artifacts = [verify.Artifact("server", Path("s"), Path("s"), ["s"], "sha"), verify.Artifact("cli", Path("c"), Path("c"), ["c", "serve"], "sha")]
        with tempfile.TemporaryDirectory() as tmp:
            h = verify.Harness(Namespace(evidence=str(Path(tmp) / "evidence.json")), Path(tmp), {"result": "fail"})
            with mock.patch.object(verify, "generate_tls", return_value=["--ssl-cert-file", "cert", "--ssl-key-file", "key"]), mock.patch.object(verify, "run_on_off", side_effect=lambda _h, artifact, **kwargs: seen.append(artifact.name)), mock.patch.object(verify, "cleanup_tls_material") as cleanup:
                verify.run_tls_matrix(h, artifacts)
        self.assertEqual(seen, ["server", "cli"])
        cleanup.assert_called_once()

    def test_sigint_server_shutdown_is_failure_but_key_cleanup_still_runs(self) -> None:
        class Proc:
            returncode = -verify.signal.SIGINT
            def poll(self):
                return self.returncode
        class Log:
            closed = False
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"
            verify.write_private(key, "secret\n")
            log = Log()
            with self.assertRaises(RuntimeError):
                verify.stop_proc_with_key_cleanup(Proc(), log, Path(tmp) / "server.log", key)
            self.assertFalse(key.exists())
            self.assertTrue(log.closed)

    def test_generated_key_signal_wait_status_is_failure(self) -> None:
        with self.assertRaises(AssertionError):
            verify.assert_wait_status_zero(verify.signal.SIGINT, "generated-key server")

    def test_generated_key_zero_wait_status_is_success(self) -> None:
        verify.assert_wait_status_zero(0, "generated-key server")

    def test_headless_explanation_is_not_credential_marker(self) -> None:
        explanation = b"--webui without --api-key is only allowed on an interactive loopback terminal so the generated session key can be shown once"
        self.assertFalse(verify.generated_key_marker_present(explanation))
        emitted = b"mlxcel WebUI session key (shown once): secret-seed"
        self.assertTrue(verify.generated_key_marker_present(emitted))

    def test_generated_key_format_drift_error_does_not_leak_unknown_secret(self) -> None:
        transcript = b"format drift generated session key: secret-seed\n"
        error = verify.generated_key_error("missing-recognized-key", None, transcript)
        text = str(error)
        self.assertNotIn("secret-seed", text)
        self.assertNotIn("format drift", text)
        self.assertIn("bytes=48", text)

    def test_generated_key_marker_error_does_not_include_recognized_secret(self) -> None:
        transcript = b"mlxcel WebUI session key (shown once): secret-seed\n"
        error = verify.generated_key_error("exited-before-key", 1, transcript)
        text = str(error)
        self.assertNotIn("secret-seed", text)
        self.assertIn("credential_marker_present=True", text)

    def test_generated_key_helper_failure_does_not_dump_output(self) -> None:
        error = verify.generated_key_helper_failure(1, b"secret stdout", b"secret stderr")
        text = str(error)
        self.assertIn("stdout_bytes=13", text)
        self.assertIn("stderr_bytes=13", text)
        self.assertNotIn("secret", text)

    def test_compat_surface_accepts_feature_disabled_envelope(self) -> None:
        body = b'{"error":{"message":"this feature is disabled","type":"feature_disabled"}}'
        result = verify.assert_compat_surface("tools", 403, body)
        self.assertEqual(result["mode"], "disabled_feature_stub")
        self.assertEqual(result["error_type"], "feature_disabled")

    def test_compat_surface_accepts_router_missing_model_envelope(self) -> None:
        body = b'{"error":{"message":"model name is missing from the request","type":"invalid_request_error","code":"invalid_request_error"}}'
        result = verify.assert_compat_surface("tools", 400, body)
        self.assertEqual(result["mode"], "router_dispatch_missing_model")
        self.assertEqual(result["status"], 400)
        self.assertEqual(result["error_code"], "invalid_request_error")

    def test_compat_surface_accepts_probe_model_not_found_without_security_claim(self) -> None:
        body = b"{\"error\":{\"message\":\"model 'mlxcel-installed-empty-router-probe' not found\",\"type\":\"invalid_request_error\"}}"
        result = verify.assert_compat_surface("tools_probe", 400, body)
        self.assertEqual(result["mode"], "router_dispatch_scope_not_forwarded")

    def test_compat_surface_rejects_unrelated_400_with_sanitized_summary(self) -> None:
        body = b'{"error":{"message":"different failure","type":"invalid_request_error","code":"secret-code"}}'
        with self.assertRaises(AssertionError) as raised:
            verify.assert_compat_surface("tools", 400, body)
        text = str(raised.exception)
        self.assertIn("status=400", text)
        self.assertIn("different failure", text)
        self.assertNotIn(body.decode(), text)


if __name__ == "__main__":
    unittest.main()
