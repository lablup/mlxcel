#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""CPU-only verifier helper tests: all process and HTTP work is mocked."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "verify_single_model", Path(__file__).with_name("verify_single_model.py")
)
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verify
spec.loader.exec_module(verify)


def options(root: Path) -> Namespace:
    return Namespace(
        server_bin=root / "server",
        cli_bin=root / "cli",
        model=root / "model",
        build_sha="a" * 40,
        features="metal,accelerate,webui",
        api_prefix="/lab/single",
        source_root=root,
        output_dir=root / "output",
        startup_resource_report=None,
        startup_timeout=10,
        request_timeout=5,
        shutdown_timeout=5,
    )


def files(root: Path) -> Namespace:
    args = options(root)
    args.model.mkdir()
    (args.model / "config.json").write_text("{}")
    (args.model / "weights.safetensors").write_bytes(b"CPU test data, not a model")
    for binary in (args.server_bin, args.cli_bin):
        binary.write_bytes(b"CPU test data, never executed")
        binary.chmod(0o700)
    return args


class SingleModelHelperTests(unittest.TestCase):
    def test_commands_cover_both_entrypoints_and_explicit_ui_flags(self):
        for cli in (True, False):
            for ui in (True, False):
                cmd = verify.command(
                    Path("/binary"),
                    cli,
                    ui,
                    Path("/model"),
                    Path("/store"),
                    Path("/key"),
                    1234,
                    "/lab/single",
                )
                self.assertEqual(cmd[1] == "serve", cli)
                self.assertEqual(cmd[-1], "--webui" if ui else "--no-webui")
                self.assertEqual(cmd[cmd.index("--api-prefix") + 1], "/lab/single")
                self.assertIn("--model-store-root", cmd)
                self.assertNotIn("Bearer", " ".join(cmd))

    def test_prefix_rejects_root_external_encoded_and_ambiguous_paths(self):
        for prefix in (
            "",
            "/",
            "//host",
            "/x/",
            "/x//y",
            "/..",
            "/x%2fy",
            "/x?y",
            "https://evil",
        ):
            with self.subTest(prefix=prefix), self.assertRaises(verify.GateError):
                verify.validate_prefix(prefix)
        self.assertEqual(verify.validate_prefix("/one-two/three_4"), "/one-two/three_4")

    def test_checkpoint_hashes_and_revision_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            before = {
                p: (p.read_bytes(), p.stat().st_mtime_ns) for p in args.model.iterdir()
            }
            provenance = verify.checkpoint_provenance(args.model)
            self.assertEqual(
                provenance["config_sha256"], verify.sha256(args.model / "config.json")
            )
            self.assertIsNone(provenance["cached_config_revision"])
            self.assertEqual(provenance["revision_source"], "unknown")
            self.assertEqual(
                before,
                {
                    p: (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in args.model.iterdir()
                },
            )
            metadata = args.model / ".cache/huggingface/download/config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("b" * 40 + "\netag\n123\n")
            self.assertEqual(
                verify.checkpoint_provenance(args.model)["cached_config_revision"],
                "b" * 40,
            )
            metadata.write_text("not-a-revision\n")
            self.assertIsNone(
                verify.checkpoint_provenance(args.model)["cached_config_revision"]
            )

    def test_checkpoint_missing_or_empty_weights_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text("{}")
            with self.assertRaises(verify.GateError):
                verify.checkpoint_provenance(path)
            (path / "empty.safetensors").touch()
            with self.assertRaises(verify.GateError):
                verify.checkpoint_provenance(path)

    def test_environment_excludes_parent_credentials_config_and_proxy(self):
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "HOME": "/real-home",
                "LLAMA_ARG_MODEL": "/other",
                "LLAMA_API_KEY": "secret",
                "HTTPS_PROXY": "proxy",
                "HF_TOKEN": "secret",
            },
            clear=True,
        ):
            env = verify.clean_env(Path("/private/home"), Path("/private/store"))
        self.assertEqual(env["HOME"], "/private/home")
        self.assertEqual(env["MLXCEL_MODELS_DIR"], "/private/store")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertFalse(
            {"LLAMA_API_KEY", "LLAMA_ARG_MODEL", "HF_TOKEN", "HTTPS_PROXY"} & env.keys()
        )

    def test_private_evidence_never_overwrites_and_redacts_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log"
            verify.private_write(
                path, b"token secret-value Authorization: Bearer other-token\n"
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                verify.private_write(path, b"overwrite")
            self.assertTrue(verify.sanitize_log(path, "secret-value"))
            self.assertNotIn(b"secret-value", path.read_bytes())
            self.assertNotIn(b"other-token", path.read_bytes())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with mock.patch.object(verify, "MAX_LOG", 1):
                self.assertTrue(verify.sanitize_log(path, "secret-value"))
            self.assertIn(b"omitted", path.read_bytes())

    def test_completion_requires_actual_nonempty_finished_text(self):
        def response(content, finish="stop"):
            return {
                "choices": [{"message": {"content": content}, "finish_reason": finish}]
            }

        self.assertEqual(verify.completion_text(response("Hello.")), "Hello.")
        for payload in (
            None,
            {},
            {"choices": []},
            response(""),
            response("  "),
            response(None),
            response("Hello", None),
            response("x" * 65537),
        ):
            with (
                self.subTest(payload=str(payload)[:80]),
                self.assertRaises(verify.GateError),
            ):
                verify.completion_text(payload)

    def test_shutdown_requires_alive_and_exact_zero_not_negative_sigint(self):
        for code, expected in ((0, True), (-signal.SIGINT, False), (1, False)):
            process = mock.Mock(returncode=code)
            process.poll.return_value = None
            self.assertEqual(verify.shutdown(process, 1)["passed"], expected)
            process.send_signal.assert_called_once_with(signal.SIGINT)
            process.wait.assert_called_once_with(timeout=1)
        process.poll.return_value = 0
        self.assertFalse(verify.shutdown(process, 1)["passed"])

    def test_shutdown_timeout_kills_group_and_never_passes(self):
        process = mock.Mock(returncode=-signal.SIGKILL, pid=2345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("test", 1), 0]
        with mock.patch.object(verify.os, "killpg") as kill:
            result = verify.shutdown(process, 1)
        kill.assert_called_once_with(2345, signal.SIGKILL)
        self.assertTrue(result["forced"])
        self.assertFalse(result["passed"])

    def test_request_blocks_redirects_proxies_oversize_and_reflected_key(self):
        response = mock.MagicMock(
            status=200, headers={"Content-Type": "application/json"}
        )
        response.__enter__.return_value = response
        opener = mock.Mock()
        opener.open.return_value = response
        with (
            mock.patch.object(
                verify.urllib.request, "build_opener", return_value=opener
            ) as build,
            mock.patch.object(
                verify, "deadline", return_value=contextlib.nullcontext()
            ),
        ):
            response.read.return_value = b"{}"
            self.assertEqual(
                verify.request(
                    "http://127.0.0.1:1234", "/lab/v1/models", "secret-key", 1
                )[0],
                200,
            )
            req = opener.open.call_args.args[0]
            self.assertEqual(req.full_url, "http://127.0.0.1:1234/lab/v1/models")
            self.assertEqual(req.get_header("Authorization"), "Bearer secret-key")
            self.assertEqual(build.call_args.args[0].proxies, {})
            self.assertIsInstance(build.call_args.args[1], verify.NoRedirect)
            response.read.return_value = b"secret-key"
            with self.assertRaisesRegex(verify.GateError, "credential_in_response"):
                verify.request(
                    "http://127.0.0.1:1234", "/lab/v1/models", "secret-key", 1
                )
            response.read.return_value = b"x" * (verify.MAX_BODY + 1)
            with self.assertRaisesRegex(verify.GateError, "response_body_limit"):
                verify.request(
                    "http://127.0.0.1:1234", "/lab/v1/models", "secret-key", 1
                )
        self.assertIsNone(
            verify.NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://evil"
            )
        )

    def test_deadline_restores_previous_handler_and_timer(self):
        old_handler = object()
        with (
            mock.patch.object(
                verify.signal, "signal", return_value=old_handler
            ) as handler,
            mock.patch.object(
                verify.signal, "setitimer", return_value=(10, 0)
            ) as timer,
            mock.patch.object(verify.time, "monotonic", side_effect=[100, 102]),
            verify.deadline(3),
            self.assertRaises(TimeoutError),
        ):
            handler.call_args.args[1](signal.SIGALRM, None)
        self.assertEqual(handler.call_args, mock.call(signal.SIGALRM, old_handler))
        self.assertEqual(timer.call_args, mock.call(signal.ITIMER_REAL, 8, 0))

    def test_arm_failure_retains_safe_evidence_and_deletes_key_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = options(root)
            with (
                mock.patch.object(
                    verify.subprocess,
                    "Popen",
                    side_effect=RuntimeError("raw-secret-must-not-persist"),
                ) as launch,
                mock.patch.object(verify, "request") as request,
            ):
                result = verify.run_arm(
                    Path("/binary"),
                    False,
                    True,
                    Path("/model"),
                    args,
                    root,
                    mock.Mock(),
                )
            launch.assert_called_once()
            request.assert_not_called()
            self.assertFalse((root / "server-ui-on/private-key").exists())
            durable = (root / "server-ui-on/result.json").read_text()
            self.assertNotIn("raw-secret", durable)
            self.assertEqual(result["error"], "RuntimeError")
            self.assertEqual(result["status"], "FAIL")

    def test_arm_log_setup_failure_also_deletes_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write = verify.private_write

            def fail_log(path, body):
                if path.name == "server.log":
                    raise OSError("private detail")
                write(path, body)

            with (
                mock.patch.object(verify, "private_write", side_effect=fail_log),
                mock.patch.object(verify.subprocess, "Popen") as launch,
            ):
                result = verify.run_arm(
                    Path("/binary"),
                    False,
                    True,
                    Path("/model"),
                    options(root),
                    root,
                    mock.Mock(),
                )
            launch.assert_not_called()
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse((root / "server-ui-on/private-key").exists())

    def exercise_transport(self, ui, *, mutation_status=422, missing_completion=False):
        calls = []
        identity = {
            "id": "opaque-id",
            "revision": "rev-1",
            "source": "single_model",
            "inference_id": "inference-id",
        }
        entry = {"identity": identity, "lifecycle": {"state": "ready"}}
        bootstrap = {
            "server": {
                "mode": "single_model",
                "api_base": "/lab/single",
                "server_instance_id": "instance",
            },
            "actions": {
                action: {"state": "read_only", "reason": "Single model"}
                for action in ("load", "unload", "download", "cache_delete")
            },
        }

        def transport(origin, path, key, timeout, body=None):
            calls.append((path, key, body))
            headers = {}
            if not path.startswith("/lab/single/"):
                status, value = 404, {}
            elif path == "/lab/single/webui/":
                status, value = (200 if ui else 404), {}
                headers = {
                    "content-type": "text/html",
                    "content-security-policy": "default-src 'self'; script-src 'self'",
                }
            elif "/ui-api/" in path and not ui:
                status, value = 404, {}
            elif not key:
                status, value = 401, {"error": {"code": "unauthorized"}}
            elif path == "/lab/single/v1/models":
                status, value = 200, {"data": [{"id": "inference-id"}]}
            elif body is not None and "/ui-api/" in path:
                status, value = mutation_status, {"error": {"code": "unsupported"}}
            elif path.endswith("/bootstrap"):
                status, value = 200, bootstrap
            elif path.endswith("/catalog"):
                status, value = (
                    200,
                    {"items": [entry], "pagination": {"next_cursor": None}},
                )
            elif "/runtime?" in path:
                self.assertIn("model_id=opaque-id&autoload=false", path)
                status, value = (
                    200,
                    {"model_id": "opaque-id", "server_instance_id": "instance"},
                )
            elif path == "/lab/single/v1/chat/completions?autoload=false":
                self.assertEqual(body["model"], "inference-id")
                self.assertEqual(body["max_tokens"], 32)
                status, value = (
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "" if missing_completion else "Hello."
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
            else:
                self.fail("Unexpected route in CPU mock: " + path)
            return status, headers, json.dumps(value).encode()

        return transport, calls

    def test_exercise_checks_prefix_auth_readonly_mutations_and_single_completion(self):
        for ui in (True, False):
            with self.subTest(ui=ui):
                transport, calls = self.exercise_transport(ui)
                contract = mock.Mock()
                contract.check.side_effect = lambda schema, value: value
                with mock.patch.object(verify, "request", side_effect=transport):
                    result = verify.exercise(
                        "http://127.0.0.1:1234",
                        "private-key",
                        ui,
                        options(Path("/unused")),
                        contract,
                    )
                self.assertIn(("/webui/", "private-key", None), calls)
                self.assertEqual(result["actual_reply"], "Hello.")
                self.assertIn("REQUIRED", result["semantic_review"])
                completions = [
                    c for c in calls if "/chat/completions?" in c[0] and c[1]
                ]
                self.assertEqual(len(completions), 1)
                if ui:
                    self.assertEqual(len(result["readonly_refusals"]), 4)
                    validated = [c.args[0] for c in contract.check.call_args_list]
                    for schema in (
                        "BootstrapResponse",
                        "CatalogListResponse",
                        "RuntimeSnapshot",
                        "ModelActionRequest",
                        "RemovalRequest",
                        "DownloadRequest",
                        "ErrorEnvelope",
                    ):
                        self.assertIn(schema, validated)
                else:
                    contract.check.assert_not_called()
                    self.assertTrue(result["ui_routes_absent"])

    def test_exercise_rejects_unprefixed_webui_in_both_modes(self):
        for ui in (True, False):
            with self.subTest(ui=ui):
                transport, calls = self.exercise_transport(ui)
                contract = mock.Mock()
                contract.check.side_effect = lambda schema, value: value

                def exposed_shell(
                    origin, path, key, timeout, body=None, transport=transport
                ):
                    if path == "/webui/":
                        self.assertEqual(key, "private-key")
                        return 200, {"content-type": "text/html"}, b"<html></html>"
                    return transport(origin, path, key, timeout, body)

                with (
                    mock.patch.object(verify, "request", side_effect=exposed_shell),
                    self.assertRaisesRegex(
                        verify.GateError, "unprefixed_route_exposed"
                    ),
                ):
                    verify.exercise(
                        "http://127.0.0.1:1234",
                        "private-key",
                        ui,
                        options(Path("/unused")),
                        contract,
                    )
                self.assertFalse(any("/chat/completions?" in c[0] for c in calls))

    def test_exercise_fails_if_readonly_mutation_succeeds_or_completion_empty(self):
        for kwargs, code in (
            ({"mutation_status": 200}, "readonly_mutation_status"),
            ({"missing_completion": True}, "nonempty_bounded_completion_required"),
        ):
            transport, calls = self.exercise_transport(True, **kwargs)
            contract = mock.Mock()
            contract.check.side_effect = lambda schema, value: value
            with (
                mock.patch.object(verify, "request", side_effect=transport),
                self.assertRaisesRegex(verify.GateError, code),
            ):
                verify.exercise(
                    "http://127.0.0.1:1234",
                    "private-key",
                    True,
                    options(Path("/unused")),
                    contract,
                )
            self.assertLessEqual(
                len([c for c in calls if "/chat/completions?" in c[0] and c[1]]), 1
            )

    def test_main_requires_explicit_optin_before_hash_or_process(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                verify, "parse_args", return_value=options(Path("/not-real"))
            ),
            mock.patch.object(verify, "checkpoint_provenance") as checkpoint,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(verify.main(), 1)
        checkpoint.assert_not_called()
        self.assertEqual(
            json.loads(output.getvalue())["error"], "MLXCEL_REQUIRE_MODELS_1_required"
        )

    def test_main_runs_four_serial_arms_and_requires_semantic_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            with (
                mock.patch.dict(os.environ, {"MLXCEL_REQUIRE_MODELS": "1"}),
                mock.patch.object(verify, "parse_args", return_value=args),
                mock.patch.object(verify, "Contract"),
                mock.patch.object(
                    verify,
                    "run_arm",
                    return_value={"status": "CAPTURED_REQUIRES_SEMANTIC_REVIEW"},
                ) as arm,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(verify.main(), 0)
            self.assertEqual(
                [(c.args[1], c.args[2]) for c in arm.call_args_list],
                [(False, True), (False, False), (True, True), (True, False)],
            )
            report = json.loads((args.output_dir / "result.json").read_text())
            self.assertEqual(report["status"], "CAPTURED_REQUIRES_SEMANTIC_REVIEW")
            self.assertTrue(report["checkpoint_unchanged"])
            self.assertEqual(
                report["binary_sha256"]["cli"], verify.sha256(args.cli_bin)
            )
            self.assertEqual(stat.S_IMODE(args.output_dir.stat().st_mode), 0o700)

    def test_main_stops_after_first_failed_arm_no_model_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            with (
                mock.patch.dict(os.environ, {"MLXCEL_REQUIRE_MODELS": "1"}),
                mock.patch.object(verify, "parse_args", return_value=args),
                mock.patch.object(verify, "Contract"),
                mock.patch.object(
                    verify, "run_arm", return_value={"status": "FAIL"}
                ) as arm,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(verify.main(), 1)
            arm.assert_called_once()

    def test_main_does_not_write_into_preexisting_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            args.output_dir.mkdir()
            with (
                mock.patch.dict(os.environ, {"MLXCEL_REQUIRE_MODELS": "1"}),
                mock.patch.object(verify, "parse_args", return_value=args),
                mock.patch.object(verify, "run_arm") as arm,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(verify.main(), 1)
            arm.assert_not_called()
            self.assertEqual(list(args.output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
