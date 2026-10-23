"""Mock-only containment and identity regression tests; never launch a model."""
from __future__ import annotations

import io
import json
from pathlib import Path
import signal
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import capture_identity as identity
import run_rust_capture as driver


def memory(available: int = identity.RESERVE + 1, swap: int = 1024**3) -> dict[str, int]:
    return {"MemAvailable": available, "SwapFree": swap, "MemTotal": 16 * 1024**3}


class CaptureIdentityTests(unittest.TestCase):
    def test_actual_gpu_line_and_exact_success_are_required(self) -> None:
        log = (f"running 1 test\ntest {identity.TEST} ... Runtime device: GPU\nok\n"
               "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 1.23s\n")
        identity.validate_log(log)
        for changed in [log.replace("GPU", "CPU"), log + "Runtime device: GPU\n",
                        log.replace("running 1 test", "running 0 tests"),
                        log.replace("1 passed; 0 failed", "0 passed; 1 failed"),
                        log.replace("0 ignored", "1 ignored"), log.replace(identity.TEST, "other")]:
            with self.subTest(log=changed), self.assertRaises(ValueError):
                identity.validate_log(changed)

    def test_duplicate_json_and_nonfinite_values_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    identity.read_json(path)

    def test_source_archive_is_rehashed_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "source"
            root.mkdir()
            manifest = {}
            for name in identity.SOURCE_REQUIRED:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"unit-test source placeholder: {name}\n")
                manifest[name] = identity.sha(path)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            archive = base / "source.tar"
            revision = "a" * 40

            def write_archive(extra: str | None = None) -> None:
                with tarfile.open(archive, "w") as tar:
                    for name in manifest:
                        tar.add(root / name, arcname=name, recursive=False)
                    text = (revision + "\n").encode()
                    member = tarfile.TarInfo("SOURCE_REVISION")
                    member.size = len(text)
                    tar.addfile(member, io.BytesIO(text))
                    if extra:
                        tar.addfile(tarfile.TarInfo(extra), io.BytesIO(b""))

            write_archive()
            source = {"revision": revision, "root": str(root),
                      "archive": {"path": str(archive), "sha256": identity.sha(archive)},
                      "manifest": {"path": str(manifest_path), "sha256": identity.sha(manifest_path)}}
            self.assertEqual(identity._source({"source": source}), manifest)
            write_archive("unmanifested.rs")
            source["archive"]["sha256"] = identity.sha(archive)
            with self.assertRaisesRegex(ValueError, "Unmanifested"):
                identity._source({"source": source})
            write_archive("../escape")
            source["archive"]["sha256"] = identity.sha(archive)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                identity._source({"source": source})
            write_archive()
            source["archive"]["sha256"] = identity.sha(archive)
            (root / "src/lib.rs").write_text("changed after source freeze")
            with self.assertRaisesRegex(ValueError, "Source differs"):
                identity._source({"source": source})

    def test_source_manifest_cannot_omit_current_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extra = root / "src/new.rs"
            extra.parent.mkdir()
            extra.write_text("new module")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({name: "a" * 64 for name in identity.SOURCE_REQUIRED}))
            source = {"revision": "b" * 40, "root": str(root), "archive": {},
                      "manifest": {"path": str(manifest), "sha256": identity.sha(manifest)}}
            with self.assertRaisesRegex(ValueError, "omits current"):
                identity._source({"source": source})

    def test_canonical_paths_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_text("content")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(ValueError, "symlink"):
                identity._path(str(root / "link"))

    def test_instances_sidecar_requires_exact_empty_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / identity.INSTANCE_SIDECAR
            path.write_bytes(b"[]")
            self.assertEqual(identity.empty_instances_sha(path), identity.sha(path))
            for payload in (b"[{}]", b"{", b"[]\n"):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "canonical empty"):
                    identity.empty_instances_sha(path)


class CaptureRunnerTests(unittest.TestCase):
    def test_unapproved_plan_and_existing_run_are_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            plan = root / "plan.json"
            plan.write_text(json.dumps({"run_dir": str(run), "helpers": {
                "run_rust_capture.py": {"sha256": driver.digest(Path(driver.__file__))}}}))
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                driver._bootstrap(plan, "0" * 64)
            self.assertFalse(run.exists())
            run.mkdir()
            marker = run / "keep"
            marker.write_text("existing capture")
            with self.assertRaises(FileExistsError):
                driver._bootstrap(plan, driver.digest(plan))
            self.assertEqual(marker.read_text(), "existing capture")

    def test_global_swap_loss_alone_does_not_stop_with_host_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            child = Mock(pid=1234)
            child.poll.side_effect = [None, 0]
            child.wait.return_value = 0
            with patch.object(driver, "memory", return_value=memory(8 * 2**30, swap=1024**3 - 256 * 2**20 - 1)), \
                 patch.object(driver, "child_memory", return_value={}), \
                 patch.object(driver.time, "sleep"), patch.object(driver.os, "killpg") as kill:
                self.assertEqual(driver.monitor(child, Path(directory), memory(8 * 2**30), identity.POLICY, 0), (0, None))
            kill.assert_not_called()

    def test_monitor_stops_on_two_low_samples_or_swap_limits(self) -> None:
        cases = [(memory(identity.RESERVE - 1), {}, 2, "two consecutive"),
                 (memory(swap=1024**3 - 256 * 2**20 - 1), {}, 1, "SwapFree"),
                 (memory(), {"VmSwap": 64 * 2**20 + 1}, 1, "VmSwap")]
        for host, child_status, samples, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                child = Mock(pid=1234)
                child.poll.return_value = None
                child.wait.return_value = -signal.SIGTERM
                with patch.object(driver, "memory", return_value=host), \
                     patch.object(driver, "child_memory", return_value=child_status), \
                     patch.object(driver.time, "sleep"), \
                     patch.object(driver.os, "killpg") as kill:
                    code, message = driver.monitor(child, Path(directory), memory(), identity.POLICY, 0)
                self.assertEqual(code, -signal.SIGTERM)
                self.assertIn(reason, message)
                self.assertEqual(len((Path(directory) / "memory.jsonl").read_text().splitlines()), samples)
                kill.assert_called_once_with(1234, signal.SIGTERM)
                child.wait.assert_called_once_with(timeout=5)

    def test_low_memory_counter_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            child = Mock(pid=1234)
            child.poll.side_effect = [None, None, None, 0]
            child.wait.return_value = 0
            with patch.object(driver, "memory", side_effect=[memory(1), memory(), memory(1)]), \
                 patch.object(driver, "child_memory", return_value={}), \
                 patch.object(driver.time, "sleep"), patch.object(driver.os, "killpg") as kill:
                self.assertEqual(driver.monitor(child, Path(directory), memory(), identity.POLICY, 0), (0, None))
            kill.assert_not_called()

    def test_term_escalates_and_reaps_actual_child_exit(self) -> None:
        child = Mock(pid=1234)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("mock", 5), -signal.SIGKILL]
        with patch.object(driver.os, "killpg") as kill:
            self.assertEqual(driver.terminate(child), -signal.SIGKILL)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(1234, signal.SIGTERM), (1234, signal.SIGKILL)])

    def test_monitor_exception_terminates_child_and_records_actual_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            frozen = run / "frozen"
            frozen.mkdir()
            for name in identity.HELPERS:
                (frozen / name).write_text("mock helper; no execution")
            plan = {"run_dir": str(run), "source": {"root": str(run)},
                    "executable": {"path": "/never-executed"}, "checkpoint_dir": "/unused",
                    "input_dir": "/unused", "case": "default"}
            validator = SimpleNamespace(authenticate_plan=Mock(return_value=(plan, {}, {})),
                                        command_environment=identity.command_environment,
                                        HELPERS=identity.HELPERS, POLICY=identity.POLICY, RESERVE=identity.RESERVE)
            child = Mock(pid=1234)
            child.poll.return_value = None
            child.wait.return_value = -signal.SIGTERM
            with patch.object(driver, "_bootstrap", return_value=(plan, run)), \
                 patch.object(driver, "_freeze", return_value=validator), \
                 patch.object(driver, "memory", return_value=memory()), \
                 patch.object(driver.subprocess, "Popen", return_value=child), \
                 patch.object(driver, "monitor", side_effect=OSError("mock telemetry failure")), \
                 patch.object(driver.os, "killpg") as kill, \
                 patch.object(driver.sys, "stderr", io.StringIO()):
                self.assertEqual(driver.run_capture(run / "absent-plan", "a" * 64), 1)
            kill.assert_called_once_with(1234, signal.SIGTERM)
            self.assertEqual((run / "child.exit").read_text(), f"{-signal.SIGTERM}\n")
            self.assertEqual((run / "wrapper.exit").read_text(), "1\n")
            self.assertFalse((run / "capture-receipt.json").exists())
            self.assertFalse((run / "output").exists())
            failure = identity.read_json(run / "failure.json")
            self.assertTrue(failure["child_started"])
            self.assertEqual(failure["child_exit"], -signal.SIGTERM)

    def test_launch_failure_preserves_evidence_without_receipt_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            frozen = run / "frozen"
            frozen.mkdir()
            for name in identity.HELPERS:
                (frozen / name).write_text("mock helper; no execution")
            plan = {"run_dir": str(run), "source": {"root": str(run)},
                    "executable": {"path": "/never-executed"}, "checkpoint_dir": "/unused",
                    "input_dir": "/unused", "case": "default"}
            validator = SimpleNamespace(authenticate_plan=Mock(return_value=(plan, {}, {})),
                                        command_environment=identity.command_environment,
                                        HELPERS=identity.HELPERS, POLICY=identity.POLICY, RESERVE=identity.RESERVE)
            with patch.object(driver, "_bootstrap", return_value=(plan, run)), \
                 patch.object(driver, "_freeze", return_value=validator), \
                 patch.object(driver, "memory", return_value=memory()), \
                 patch.object(driver.subprocess, "Popen", side_effect=OSError("mock launch failure")) as launch, \
                 patch.object(driver.sys, "stderr", io.StringIO()):
                self.assertEqual(driver.run_capture(run / "absent-plan", "a" * 64), 1)
            self.assertEqual((run / "wrapper.exit").read_text(), "1\n")
            self.assertFalse((run / "capture-receipt.json").exists())
            self.assertFalse((run / "output").exists())
            failure = identity.read_json(run / "failure.json")
            self.assertFalse(failure["child_started"])
            self.assertIsNone(failure["child_exit"])
            args, kwargs = launch.call_args
            self.assertEqual(args[0], identity.command_environment(plan)[0])
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["MLX_ENABLE_TF32"], "0")


if __name__ == "__main__":
    unittest.main()
