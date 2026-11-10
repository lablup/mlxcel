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


verdict = sys.modules["activity_performance_verdict"]
POLICY = verdict.load_policy()
FIXTURE = json.loads(verdict.FIXTURE_PATH.read_text(encoding="utf-8"))
GEOMETRY = {"innerWidth": 700, "innerHeight": 900, "visualWidth": 700, "visualHeight": 900}


def fixture_pairs(prefix: str) -> list[dict[str, float]]:
    case = next(item for item in FIXTURE["cases"] if item["name"].startswith(prefix))
    (pairs,) = verdict.pairs_from_samples(case["samples"]).values()
    return pairs


NULL_PAIRS = fixture_pairs("null: every recorded one-visible")


def activity_output(modes: tuple[str, ...], pairs_by_mode: dict[str, list[dict[str, float]]] | None = None) -> dict[str, object]:
    """Harness output shaped exactly as activity-performance.mjs writes it, for recorded pairs."""
    samples: list[dict[str, object]] = [{"mode": "warmup", "pair": -1, "tokens_per_second": 150.0, "predicted_tokens": 256, "predicted_ms": 1700}]
    preflight = [{"mode": mode, "client": 0, "geometry": GEOMETRY} for mode in modes for _ in range(verdict.CLIENTS_PER_MODE[mode])]
    summaries = []
    for mode in modes:
        pairs = (pairs_by_mode or {}).get(mode, NULL_PAIRS)
        for index, pair in enumerate(pairs):
            off = {"mode": "off", "pair": index, "tokens_per_second": pair["off"], "predicted_tokens": 256, "predicted_ms": 1700}
            on = {"mode": mode, "pair": index, "tokens_per_second": pair["on"], "predicted_tokens": 256, "predicted_ms": 1700}
            samples.extend([off, on] if index % 2 == 0 else [on, off])
            preflight.extend({"mode": mode, "client": client, "geometry": GEOMETRY} for client in range(verdict.CLIENTS_PER_MODE[mode]))
        summaries.append(verdict.mode_summary(mode, pairs, POLICY))
    hidden_ran = "hidden" in modes
    failing = {item["status"] for item in summaries} - {"within-target"}
    status = "incomplete" if not hidden_ran else "investigate" if "investigate" in failing else "inconclusive" if failing else "within-target"
    return {"status": status, "hidden_native": "measured" if hidden_ran else "not-run", "preflight": preflight, "verdict_policy": POLICY, "samples": samples, "summaries": summaries}


def valid_activity_output() -> dict[str, object]:
    return activity_output(("one-visible", "two-visible", "hidden"))


def visible_only_activity_output() -> dict[str, object]:
    return activity_output(("one-visible", "two-visible"))

# Tail of the activity-performance.log from CI run 42753658537 (disk filled mid-run): the last, truncated
# sample line followed by Node's unhandled ENOSPC stream error. The runner checkout path is neutralized.
ENOSPC_LOG_TAIL = r'''{"mode":"off","pair":3,"tokens_per_second":165.58441558441558,"predicted_tokens":256,"predicted_ms":1540,"diagnostics":{"before":{"observed_at":"2026-11-10T04:43:02.288Z","host":{"status":"captured","load_one":2.44,"load_five":3.28,"load_fifteen":3.4,"running":6,"total":1466},"gpu":{"status":"captured","index":"0","pstate":"P0","gpu_utilization_percent":"85","memory_clock_mhz":"[N/A]"}},"after":{"observed_at":"2026-11-10T04:43:03.913Z","host":{"status":"captured","load_one":2.57,"load_five":3.29,"load_fifteen":3.4node:events:487
      throw er; // Unhandled 'error' event
      ^

Error: ENOSPC: no space left on device, write
    at writeSync (node:fs:1128:3)
    at SyncWriteStream._write (node:internal/fs/sync_write_stream:27:5)
    at writeOrBuffer (node:internal/streams/writable:572:12)
    at _write (node:internal/streams/writable:501:10)
    at Writable.write (node:internal/streams/writable:510:10)
    at request (file:///runner/_work/mlxcel/mlxcel/webui/scripts/activity-performance.mjs:121:40)
    at process.processTicksAndRejections (node:internal/process/task_queues:104:5)
    at async file:///runner/_work/mlxcel/mlxcel/webui/scripts/activity-performance.mjs:137:126
Emitted 'error' event on SyncWriteStream instance at:
    at emitErrorNT (node:internal/streams/destroy:170:8)
    at emitErrorCloseNT (node:internal/streams/destroy:129:3)
    at process.processTicksAndRejections (node:internal/process/task_queues:90:21) {
  errno: -28,
  syscall: 'write',
  code: 'ENOSPC'
}

Node.js v26.5.1
'''
BUDGET_LOG_TAIL = '{"mode":"off","pair":3,"tokens_per_second":165.5}\nError: budget exceeded: hidden slowdown 6.1% over the 2% target\n    at verdict (file:///runner/_work/mlxcel/mlxcel/webui/scripts/activity-performance.mjs:300:11)\n'


class ActivityPerformanceHelperTests(unittest.TestCase):
    def test_visible_only_mode_accepts_its_own_shape_and_no_other(self) -> None:
        # The deferral is declared by the caller, so neither shape may pass under the other mode.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps(visible_only_activity_output()))
            self.assertEqual(verify.validate_activity_output(path, "visible-only-headed")["hidden_native"], "not-run")
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path)
            path.write_text(json.dumps(valid_activity_output()))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path, "visible-only-headed")

    def test_visible_only_mode_still_enforces_budgets_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            over = visible_only_activity_output()
            over["summaries"][0]["median_decode_degradation_percent"] = 3.0  # type: ignore[index]
            path.write_text(json.dumps(over))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path, "visible-only-headed")
            short = visible_only_activity_output()
            short["samples"] = short["samples"][:-1]  # type: ignore[index]
            path.write_text(json.dumps(short))
            with self.assertRaises(AssertionError):
                verify.validate_activity_output(path, "visible-only-headed")

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
            nvidia_smi = root / "nvidia-smi"
            nvidia_smi.write_text("#!/bin/sh\n")
            nvidia_smi.chmod(0o700)
            args = verify.parse_args([
                "--server-bin", str(server),
                "--model", str(model),
                "--checkpoint-revision", "rev",
                "--source-sha", "a" * 40,
                "--features", "cuda,webui",
                "--evidence", str(root / "evidence.json"),
                "--nvidia-smi", str(nvidia_smi),
                "--port", "53001",
            ])
            self.assertEqual(args.features, ["cuda", "webui"])
            self.assertEqual(args.port, 53001)
            self.assertEqual(args.nvidia_smi, nvidia_smi)
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

    def test_terminate_owned_survives_a_process_that_outlives_sigkill(self) -> None:
        # A process stuck in the kernel can outlive SIGKILL's wait; the cleanup in `finally`
        # must report it rather than raise and abandon the processes after it (#1949).
        class Stuck:
            pid = 999_999_999
            returncode = None
            def poll(self):
                return None
            def wait(self, timeout=None):
                raise verify.subprocess.TimeoutExpired("stuck", timeout)
        later = mock.Mock()
        later.poll.return_value = 0
        later.pid = 999_999_998
        signals = []
        def fake_killpg(pgid, sig):
            signals.append((pgid, sig))
            if pgid == later.pid:
                raise ProcessLookupError
        owned = [verify.OwnedProcess("later", later), verify.OwnedProcess("stuck", Stuck())]
        with mock.patch.object(verify.os, "killpg", side_effect=fake_killpg), mock.patch.object(verify, "KILL_WAIT", 0.2, create=True), mock.patch.object(verify, "TERM_WAIT", 0.2, create=True):
            results = verify.terminate_owned(owned)
        self.assertEqual([r["label"] for r in results], ["stuck", "later"])
        self.assertTrue(results[0]["forced"])
        self.assertFalse(results[0]["process_group_empty"])
        self.assertIn((Stuck.pid, verify.signal.SIGKILL), signals)
        self.assertTrue(results[1]["process_group_empty"])

    def test_terminate_owned_empties_the_group_a_quick_exit_left_behind(self) -> None:
        # The leader exits inside the SIGTERM wait and leaves a member that ignores SIGTERM;
        # recording process_group_empty is not enough, the member has to be stopped (#1949).
        proc = verify.subprocess.Popen(["sh", "-c", "trap '' TERM; sleep 60 & exit 0"], start_new_session=True)
        try:
            proc.wait(timeout=10)
            self.assertFalse(verify.process_group_empty(proc.pid, 0.3), "the fixture must leave a live group member")
            results = verify.terminate_owned([verify.OwnedProcess("quick-exit", proc)])
            self.assertTrue(results[0]["process_group_empty"])
            self.assertTrue(results[0]["forced"])
            self.assertTrue(verify.process_group_empty(proc.pid, 0.5))
        finally:
            try:
                verify.os.killpg(proc.pid, verify.signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_record_owned_pid_appends_reapable_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.pid"
            verify.record_owned_pid(None, mock.Mock())
            self.assertFalse(path.exists())
            proc = verify.subprocess.Popen(["sleep", "30"], start_new_session=True)
            try:
                verify.record_owned_pid(path, verify.OwnedProcess("xvfb", proc))
                verify.record_owned_pid(path, verify.OwnedProcess("mlxcel-server", proc))
                records = json.loads(path.read_text())["records"]
                self.assertEqual([r["label"] for r in records], ["xvfb", "mlxcel-server"])
                self.assertEqual(records[0]["pid"], proc.pid)
                self.assertEqual(records[0]["pgid"], proc.pid)
                if sys.platform.startswith("linux"):
                    self.assertIsInstance(records[0]["start_ticks"], int)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual([p.name for p in Path(tmp).iterdir()], ["gate.pid"])
            finally:
                proc.kill()
                proc.wait()

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

    def test_validate_activity_output_reports_slowdown_and_noise_distinctly(self) -> None:
        # The job log has to tell a slowdown from measurement noise without the artifact (#1925).
        slow = fixture_pairs("injected 6% slowdown on the recorded two-visible")
        noisy = fixture_pairs("noise too wide to decide")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps(activity_output(("one-visible", "two-visible"), {"two-visible": slow})))
            with self.assertRaisesRegex(AssertionError, r"two-visible: investigate \(slowdown\): median 6\.12%, 98% interval \[2\.15%, 8\.57%\] over 14 pairs; slowdown: the interval lies wholly above the 2% budget"):
                verify.validate_activity_output(path, "visible-only-headed")
            path.write_text(json.dumps(activity_output(("one-visible", "two-visible"), {"one-visible": noisy})))
            with self.assertRaisesRegex(AssertionError, r"one-visible: inconclusive \(measurement noise\): .* over 40 pairs; measurement noise, not a slowdown"):
                verify.validate_activity_output(path, "visible-only-headed")

    def test_baseline_cv_alone_no_longer_fails(self) -> None:
        drift = fixture_pairs("baseline drift with CV above 5%")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            data = activity_output(("one-visible", "two-visible"), {"two-visible": drift})
            self.assertGreater(data["summaries"][1]["baseline_cv_percent"], 5)  # type: ignore[index]
            path.write_text(json.dumps(data))
            self.assertEqual(verify.validate_activity_output(path, "visible-only-headed")["summaries"][1]["status"], "within-target")

    def test_validate_activity_output_rederives_the_verdict_from_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            data = visible_only_activity_output()
            data["verdict_policy"] = {**POLICY, "precision_ceiling_percent": 10}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "verdict policy"):
                verify.validate_activity_output(path, "visible-only-headed")
            data = activity_output(("one-visible", "two-visible"), {"two-visible": fixture_pairs("injected 6% slowdown on the recorded two-visible")})
            data["summaries"][1]["status"] = "within-target"  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "status is 'within-target', but its samples give 'investigate'"):
                verify.validate_activity_output(path, "visible-only-headed")
            data = visible_only_activity_output()
            data["preflight"] = data["preflight"][:-1]  # type: ignore[index]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "preflight counts"):
                verify.validate_activity_output(path, "visible-only-headed")
            data = visible_only_activity_output()
            data["samples"].append({"mode": "warmup", "pair": -1, "tokens_per_second": 1.0, "predicted_tokens": 1, "predicted_ms": 1.0})  # type: ignore[union-attr]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "sample counts"):
                verify.validate_activity_output(path, "visible-only-headed")

class ActivityEnvironmentFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def log(self, text: str) -> Path:
        path = self.root / "activity-performance.log"
        path.write_text(text, encoding="utf-8")
        return path

    def harness(self, node_log_text: str) -> object:
        """A harness whose node_bin is a shell script that prints the log text and exits 1."""
        (self.root / "payload.txt").write_text(node_log_text, encoding="utf-8")
        node = self.root / "fake-node"
        node.write_text(f'#!/bin/sh\ncat "{self.root}/payload.txt"\nexit 1\n')
        node.chmod(0o700)
        args = Namespace(node_bin=str(node), repo_root=self.root, activity_timeout=30, perf_mode="hidden", prompt_file=None, nvidia_smi=None, pid_file=None)
        return verify.Harness(args=args, work=self.root, model_for_server=self.root, key_file=self.root / "key", base_root="http://x", base_prefixed="http://x/p", env={}, secrets=[])

    def test_classifies_the_real_enospc_tail_and_other_host_errors(self) -> None:
        self.assertEqual(verify.classify_environment_failure(self.log(ENOSPC_LOG_TAIL)), "ENOSPC (no space left on device)")
        self.assertEqual(verify.classify_environment_failure(self.log("x\nError: EDQUOT: quota\n")), "EDQUOT (disk quota exceeded)")
        self.assertEqual(verify.classify_environment_failure(self.log("x\nError: EIO: i/o error, write\n")), "EIO (input/output error)")
        self.assertEqual(verify.classify_environment_failure(self.log("Emitted 'error' event on Socket instance at:\n")), "unhandled stream error event")

    def test_does_not_classify_a_performance_failure_or_a_missing_log(self) -> None:
        self.assertIsNone(verify.classify_environment_failure(self.log(BUDGET_LOG_TAIL)))
        self.assertIsNone(verify.classify_environment_failure(self.log("MEIOSIS ENOSPCX preENOSPC\n")))
        self.assertIsNone(verify.classify_environment_failure(self.root / "missing.log"))

    def test_only_the_bounded_tail_is_scanned(self) -> None:
        old = "Error: ENOSPC: no space left on device\n" + "x" * (verify.ENVIRONMENT_LOG_TAIL_BYTES + 10)
        self.assertIsNone(verify.classify_environment_failure(self.log(old)))

    def test_run_activity_script_raises_infrastructure_error_for_a_full_disk(self) -> None:
        with self.assertRaises(verify.ActivityInfrastructureError) as ctx:
            verify.run_activity_script(self.harness(ENOSPC_LOG_TAIL), "m", "i")
        message = str(ctx.exception)
        self.assertIn("ENOSPC (no space left on device)", message)
        self.assertIn("exited 1", message)
        self.assertIn("activity-performance.log", message)

    def test_run_activity_script_keeps_the_plain_error_for_a_budget_failure(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            verify.run_activity_script(self.harness(BUDGET_LOG_TAIL), "m", "i")
        self.assertNotIsInstance(ctx.exception, verify.ActivityInfrastructureError)
        self.assertTrue(str(ctx.exception).startswith("activity-performance.mjs exited 1; log="))

    def test_main_labels_infrastructure_failures_and_keeps_the_old_message_otherwise(self) -> None:
        argv = ["unused"]
        with mock.patch.object(verify, "parse_args", return_value=Namespace(evidence=self.root / "e.json")):
            for error, label in (
                (verify.ActivityInfrastructureError("ENOSPC (no space left on device); activity-performance.mjs exited 1; log=/l"), "activity gate infrastructure failure: ENOSPC"),
                (AssertionError("activity-performance.mjs exited 1; log=/l"), "activity performance verification failed: activity-performance.mjs exited 1"),
            ):
                with mock.patch.object(verify, "run", side_effect=error), redirect_stderr(StringIO()) as err:
                    self.assertEqual(verify.main(argv), 1)
                self.assertIn(label, err.getvalue())
                if label.startswith("activity gate"):
                    self.assertNotIn("verification failed", err.getvalue())


if __name__ == "__main__":
    unittest.main()
