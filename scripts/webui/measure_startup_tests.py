#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Pure-Python unit tests for measure_startup.py."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("measure_startup.py")
spec = importlib.util.spec_from_file_location("measure_startup", MODULE_PATH)
assert spec and spec.loader
measure = importlib.util.module_from_spec(spec)
sys.modules["measure_startup"] = measure
spec.loader.exec_module(measure)


class FakeProc:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None
        self.signals: list[int] = []
        self.killed = False

    def poll(self):  # type: ignore[no-untyped-def]
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = 0

    def wait(self, timeout=None):  # type: ignore[no-untyped-def]
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class MeasureStartupHelperTests(unittest.TestCase):
    def test_repeats_must_be_at_least_three(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                measure.parse_args(["--server-bin", __file__, "--cli-bin", __file__, "--repeats", "2"])

    def test_private_key_file_uses_0600_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            measure.write_private(path, "secret\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_text(), "secret\n")

    def test_build_command_toggles_webui_and_public_command_hides_key_path(self) -> None:
        artifact = measure.Artifact("mlxcel-server", Path("/bin/mlxcel-server"), ["/bin/mlxcel-server"], "abc", "head", "webui,metal")
        command = measure.build_command(artifact, 18080, Path("models"), Path("store"), Path("private-key"), True)
        self.assertIn("--webui", command)
        self.assertNotIn("--no-webui", command)
        public = measure.public_command(command, Path("private-key"))
        self.assertEqual(public[0], "mlxcel-server")
        self.assertIn("<private-key-file>", public)
        self.assertNotIn("private-key", public)
        off = measure.build_command(artifact, 18081, Path("models"), Path("store"), Path("private-key"), False)
        self.assertIn("--no-webui", off)

    def test_sample_rss_parses_ps_output(self) -> None:
        class Result:
            returncode = 0
            stdout = "  12345\n"
            stderr = ""
        self.assertEqual(measure.sample_rss_kib(111, runner=lambda *args, **kwargs: Result()), 12345)

    def test_run_one_records_observation_redacts_secret_and_deletes_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = measure.Artifact("mlxcel-server", Path("/bin/mlxcel-server"), ["/bin/mlxcel-server"], "abc", "head", "webui")
            captured: dict[str, object] = {}
            fake_proc = FakeProc()

            def launcher(command, work, env, log_file):  # type: ignore[no-untyped-def]
                captured["command"] = command
                captured["work"] = work
                captured["env"] = env
                log_file.write(b"server started without secret\n")
                return fake_proc

            result = measure.run_one(
                artifact,
                webui=True,
                repeat_index=1,
                root=root,
                timeout_secs=1,
                shutdown_timeout_secs=1,
                secrets_to_hide=[],
                launcher=launcher,
                waiter=lambda base, proc, timeout: 12.345,
                rss_sampler=lambda pid: 67890,
                port_picker=lambda: 19000,
            )

            self.assertEqual(result["startup_ms_to_root_health_200"], 12.345)
            self.assertEqual(result["rss_kib_after_health_200"], 67890)
            self.assertEqual(result["shutdown"]["exit_code"], 0)
            self.assertFalse((root / "mlxcel-server/webui-on/repeat-01/api-key.txt").exists())
            rendered = json.dumps(result)
            self.assertNotIn("secret", rendered.lower())
            self.assertIn("<private-key-file>", rendered)
            self.assertIn("--webui", captured["command"])

    def test_clean_env_strips_proxy_and_secret_like_inputs(self) -> None:
        old = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update({"PATH": "/bin", "HTTPS_PROXY": "http://proxy.invalid", "LLAMA_API_KEY": "leak", "DYLD_LIBRARY_PATH": "/mlx"})
            env = measure.clean_env(Path("/tmp/home"), Path("/tmp/store"))
            self.assertEqual(env["PATH"], "/bin")
            self.assertEqual(env["DYLD_LIBRARY_PATH"], "/mlx")
            self.assertNotIn("HTTPS_PROXY", env)
            self.assertNotIn("LLAMA_API_KEY", env)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["NO_PROXY"], "*")
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
