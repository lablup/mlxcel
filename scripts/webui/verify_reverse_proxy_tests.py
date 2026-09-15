#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import importlib.util, signal, subprocess, sys, tempfile, unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
SCRIPT = Path(__file__).with_name("verify_reverse_proxy.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("verify_reverse_proxy", SCRIPT)
assert spec and spec.loader
proxy = importlib.util.module_from_spec(spec); sys.modules["verify_reverse_proxy"] = proxy; spec.loader.exec_module(proxy)

class ReverseProxyHelperTests(unittest.TestCase):
    def test_public_navigation_allows_missing_origin_and_fetch_metadata(self) -> None:
        headers = {"Host": "public.local:443"}
        self.assertIsNone(proxy.edge_rejection("GET", "/lab/webui/", headers, "public.local:443", "https://public.local:443"))
        self.assertIsNone(proxy.edge_rejection("GET", "/lab/webui/app.js", {**headers, "Sec-Fetch-Site": "none"}, "public.local:443", "https://public.local:443"))
        self.assertEqual(proxy.edge_rejection("GET", "/lab/webui/", {**headers, "Sec-Fetch-Site": "cross-site"}, "public.local:443", "https://public.local:443"), "invalid_fetch_metadata")

    def test_private_get_allows_missing_origin_but_rejects_bad_origin(self) -> None:
        headers = {"Host": "public.local:443", "Sec-Fetch-Site": "same-origin"}
        self.assertIsNone(proxy.edge_rejection("GET", "/lab/ui-api/v1/bootstrap", headers, "public.local:443", "https://public.local:443"))
        self.assertEqual(proxy.edge_rejection("GET", "/lab/ui-api/v1/bootstrap", {**headers, "Origin": "https://evil.local"}, "public.local:443", "https://public.local:443"), "invalid_origin")

    def test_mutations_require_exact_origin_and_fetch_metadata(self) -> None:
        headers = {"Host": "public.local:443", "Origin": "https://public.local:443", "Sec-Fetch-Site": "same-origin"}
        self.assertIsNone(proxy.edge_rejection("POST", "/lab/ui-api/v1/catalog/refresh", headers, "public.local:443", "https://public.local:443"))
        self.assertEqual(proxy.edge_rejection("POST", "/lab/ui-api/v1/catalog/refresh", {**headers, "Host": "evil.local"}, "public.local:443", "https://public.local:443"), "invalid_host")
        self.assertEqual(proxy.edge_rejection("POST", "/lab/ui-api/v1/catalog/refresh", {"Host": "public.local:443", "Sec-Fetch-Site": "same-origin"}, "public.local:443", "https://public.local:443"), "invalid_origin")
        self.assertEqual(proxy.edge_rejection("POST", "/lab/ui-api/v1/catalog/refresh", {**headers, "Sec-Fetch-Site": "cross-site"}, "public.local:443", "https://public.local:443"), "invalid_fetch_metadata")

    def test_edge_rejects_url_credentials_before_forwarding(self) -> None:
        headers = {"Host": "public.local:443", "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(proxy.edge_rejection("GET", "/lab/ui-api/v1/bootstrap?api_key=secret", headers, "public.local:443", "https://public.local:443"), "query_credentials")

    def test_upstream_headers_rewrite_trusted_authority_and_preserve_security_headers(self) -> None:
        headers = {"Host": "public.local:443", "Origin": "https://public.local:443", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", "Authorization": "Bearer secret", "Connection": "close", "Content-Length": "4"}
        out = proxy.make_upstream_headers(headers, "127.0.0.1:18080", "http://127.0.0.1:18080")
        self.assertEqual(out["Host"], "127.0.0.1:18080")
        self.assertEqual(out["Origin"], "http://127.0.0.1:18080")
        self.assertEqual(out["Authorization"], "Bearer secret")
        self.assertEqual(out["Sec-Fetch-Site"], "same-origin")
        self.assertNotIn("Connection", out)
        self.assertNotIn("Content-Length", out)

    def test_upstream_headers_do_not_invent_origin_when_browser_omits_it(self) -> None:
        out = proxy.make_upstream_headers({"Host": "public.local:443", "Authorization": "Bearer secret"}, "127.0.0.1:18080", "http://127.0.0.1:18080")
        self.assertEqual(out["Host"], "127.0.0.1:18080")
        self.assertNotIn("Origin", out)

    def test_forward_record_never_stores_bearer_value_or_url_key(self) -> None:
        record = proxy.forward_record("GET", "http://127.0.0.1:18080/lab/ui-api/v1/bootstrap", {"Authorization": "Bearer secret", "Host": "127.0.0.1:18080", "Origin": "http://127.0.0.1:18080", "Sec-Fetch-Site": "same-origin"}, 200)
        self.assertTrue(record["authorization_header"])
        self.assertFalse(record["url_has_credential_query"])
        self.assertNotIn("secret", repr(record))

    def test_response_headers_drop_hop_by_hop_and_recompute_length(self) -> None:
        result = dict(proxy.response_headers({"Content-Type": "text/plain", "Connection": "close", "Content-Length": "999"}, 3))
        self.assertEqual(result["Content-Type"], "text/plain")
        self.assertEqual(result["Content-Length"], "3")
        self.assertNotIn("Connection", result)

    def test_harness_flush_redacts_structured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence.json"
            h = proxy.Harness(Namespace(evidence=str(evidence)), Path(tmp), {"result": "fail"}, ["secret"])
            h.add("reverse_proxy", {"message": "Bearer secret"})
            h.add("reverse_proxy", {"message": "second update"})
            text = evidence.read_text()
            self.assertIn("<redacted>", text)
            self.assertIn("second update", text)
            self.assertNotIn("secret", text)

    def test_harness_refuses_preexisting_evidence_path_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "evidence.json"
            existing.write_text("do not truncate")
            h = proxy.Harness(Namespace(evidence=str(existing)), Path(tmp), {"result": "fail"})
            with self.assertRaises(FileExistsError):
                h.flush()
            self.assertEqual(existing.read_text(), "do not truncate")
            symlink = Path(tmp) / "evidence-link.json"
            symlink.symlink_to(existing)
            h = proxy.Harness(Namespace(evidence=str(symlink)), Path(tmp), {"result": "fail"})
            with self.assertRaises(AssertionError):
                h.flush()

    def test_backend_cleanup_uses_process_group_and_reports_timeout(self) -> None:
        class Proc:
            pid = 123
            returncode = None
            def poll(self):
                return None
            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired(["server"], timeout or 0)
        class Log:
            closed = False
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"; key.write_text("secret")
            log = Log(); calls: list[tuple[int, int]] = []
            with mock.patch.object(proxy.os, "killpg", side_effect=lambda pid, sig: calls.append((pid, sig))):
                with self.assertRaises(subprocess.TimeoutExpired):
                    proxy.stop_backend_group(Proc(), log, Path(tmp) / "log", key)
            self.assertEqual(calls, [(123, signal.SIGINT), (123, signal.SIGKILL)])
            self.assertTrue(log.closed)
            self.assertFalse(key.exists())

if __name__ == "__main__":
    unittest.main()
