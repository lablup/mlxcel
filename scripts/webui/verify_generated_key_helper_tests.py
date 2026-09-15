#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import importlib.util, signal, subprocess, sys, unittest
from pathlib import Path
from unittest import mock
SCRIPT = Path(__file__).with_name("verify_generated_key_helper.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("verify_generated_key_helper", SCRIPT)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec); sys.modules["verify_generated_key_helper"] = helper; spec.loader.exec_module(helper)

class GeneratedKeyHelperTests(unittest.TestCase):
    def test_already_exited_process_is_not_fabricated_unknown_success(self) -> None:
        class Proc:
            pid = 123
            returncode = 7
            def poll(self) -> int:
                return self.returncode
            def wait(self, timeout: float | None = None) -> int:
                raise AssertionError("wait should not run for already-exited child")
        self.assertEqual(helper.terminate_process(Proc())["returncode"], 7)

    def test_timed_out_child_cleanup_uses_process_group_and_reports_failure(self) -> None:
        class Proc:
            pid = 123
            returncode = None
            def poll(self):
                return None
            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired(["server"], timeout or 0)
        calls: list[tuple[int, int]] = []
        with mock.patch.object(helper.os, "killpg", side_effect=lambda pid, sig: calls.append((pid, sig))):
            result = helper.terminate_process(Proc(), sigint_timeout=0.01, sigkill_timeout=0.01)
        self.assertEqual(calls, [(123, signal.SIGINT), (123, signal.SIGKILL)])
        self.assertIn("SIGKILL deadline", result["cleanup_error"])

    def test_close_fd_suppresses_oserror(self) -> None:
        with mock.patch.object(helper.os, "close", side_effect=OSError("already closed")):
            helper.close_fd(7)

if __name__ == "__main__":
    unittest.main()
