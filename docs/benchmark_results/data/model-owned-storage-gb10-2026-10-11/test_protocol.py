#!/usr/bin/env python3
"""Lightweight protocol fixtures for the issue #2230 storage sweep."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module("storage_analyze", HERE / "analyze.py")
gate_driver = load_module("storage_gate", HERE / "gate.py")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProtocolFixture:
    """A complete synthetic 60-arm protocol with no model or GPU work."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.worktree = Path(
            subprocess.run(
                ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("gate.py", "analyze.py", "run.sh"):
            (self.root / name).write_text(f"fixture {name}\n", encoding="utf-8")
        hostgate = self.root / "hostgate.py"
        hostgate.write_text("fixture hostgate\n", encoding="utf-8")
        binary = self.root / "mlxcel"
        binary.write_bytes(b"fixture release binary")
        binary_epoch = int(dt.datetime(2026, 10, 10, 23, tzinfo=dt.UTC).timestamp())
        os.utime(binary, (binary_epoch, binary_epoch))

        environment = {
            "started_utc": "2026-10-11T00:00:00Z",
            "completed_utc": "2026-10-11T23:00:00Z",
            "worktree": str(self.worktree),
            "worktree_clean": "true",
            "binary": str(binary),
            "binary_sha256": file_sha256(binary),
            "binary_size_bytes": str(binary.stat().st_size),
            "binary_mtime_utc": "2026-10-10T23:00:00Z",
            "commit": self.commit,
            "mlx_cmake_git_tag": "1" * 40,
            "mlx_checkout": str(self.root / "mlx-src"),
            "mlx_checkout_commit": "1" * 40,
            "hostgate_source": str(hostgate),
            "hostgate_sha256": file_sha256(hostgate),
            "gate_sha256": file_sha256(self.root / "gate.py"),
            "analyzer_sha256": file_sha256(self.root / "analyze.py"),
            "run_sha256": file_sha256(self.root / "run.sh"),
            "host": "fixture-host",
            "uname": "Linux fixture",
            "gpu": "Fixture GPU, 1.0, 0000:00:00.0",
        }
        (self.root / "environment.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in environment.items()),
            encoding="utf-8",
        )
        (self.root / "complete").touch()

        schedule_lines = ["\t".join(analyze.SCHEDULE_FIELDS)]
        evidence = [
            {
                "label": "run-preflight",
                "started_utc": "2026-10-11T00:00:01Z",
                "completed_utc": "2026-10-11T00:00:02Z",
                "port": 18080,
                "port_free_before_wait": True,
                "port_free_after_wait": True,
            }
        ]
        for index, spec in enumerate(analyze.expected_arms(), start=1):
            scheduled = dt.datetime(2026, 10, 11, tzinfo=dt.UTC) + dt.timedelta(
                minutes=10 * index
            )
            gate_started = scheduled - dt.timedelta(minutes=2)
            gate_completed = scheduled - dt.timedelta(minutes=1)
            scheduled_utc = scheduled.strftime("%Y-%m-%dT%H:%M:%SZ")
            gate_started_utc = gate_started.strftime("%Y-%m-%dT%H:%M:%SZ")
            gate_completed_utc = gate_completed.strftime("%Y-%m-%dT%H:%M:%SZ")
            schedule_lines.append(
                "\t".join(
                    (
                        spec.model,
                        str(spec.prompt_tokens),
                        str(spec.round),
                        str(spec.position),
                        spec.arm,
                        spec.backend,
                        scheduled_utc,
                    )
                )
            )
            evidence.append(
                {
                    "label": spec.label,
                    "started_utc": gate_started_utc,
                    "completed_utc": gate_completed_utc,
                    "port": 18080,
                    "port_free_before_wait": True,
                    "port_free_after_wait": True,
                    "accepted_quiet": True,
                    "ci_job_running": False,
                    "busy_procs": [],
                    "foreign_model_procs": [],
                    "gate_wait_s": 0,
                    "driver_wait_s": 0,
                    "mem_wait_s": 0,
                    "load1": 0.5,
                    "mem_available_gib": 64.0,
                    "mem_floor_gib": 32.0,
                    "nvrm_window_5m": 0,
                    "nvrm_window_max": 50,
                    "nvrm_total_since_boot": 0,
                    "nvrm_total_max": 400,
                    "quiet_rechecks": index % 2,
                }
            )
            (self.root / f"server-{spec.label}.log").write_text(
                f"requested_decode_storage_backend={spec.backend}\nfixture server output\n",
                encoding="utf-8",
            )
            (self.root / spec.client_name).write_text(
                "".join(
                    f"{concurrency} 1 0 1.0 2.0 3.0 {100.0 + index + concurrency:.1f}\n"
                    for concurrency in analyze.CONCURRENCIES
                ),
                encoding="utf-8",
            )
        (self.root / "schedule.tsv").write_text(
            "\n".join(schedule_lines) + "\n", encoding="utf-8"
        )
        (self.root / "hostgate.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in evidence),
            encoding="utf-8",
        )


class ProtocolValidationTests(unittest.TestCase):
    def test_complete_quiet_protocol_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            protocol = analyze.validate_protocol(fixture.root)
            self.assertEqual(len(protocol.schedule_times), 60)
            self.assertEqual(len(protocol.runs), 180)

    def test_missing_gate_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            path = fixture.root / "hostgate.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 61 records"):
                analyze.validate_protocol(fixture.root)

    def test_dirty_gate_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            path = fixture.root / "hostgate.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[1]["busy_procs"] = ["rustc"]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "busy_procs must be empty"):
                analyze.validate_protocol(fixture.root)

    def test_reordered_gate_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            path = fixture.root / "hostgate.jsonl"
            records = path.read_text(encoding="utf-8").splitlines()
            records[1], records[2] = records[2], records[1]
            path.write_text("\n".join(records) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "out of order"):
                analyze.validate_protocol(fixture.root)

    def test_nonmonotonic_schedule_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            path = fixture.root / "schedule.tsv"
            rows = path.read_text(encoding="utf-8").splitlines()
            fields = rows[2].split("\t")
            fields[-1] = "2026-10-11T00:05:00Z"
            rows[2] = "\t".join(fields)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not nondecreasing"):
                analyze.validate_protocol(fixture.root)

    def test_missing_server_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            first = analyze.expected_arms()[0]
            (fixture.root / f"server-{first.label}.log").unlink()
            with self.assertRaisesRegex(ValueError, "server log set mismatch"):
                analyze.validate_protocol(fixture.root)

    def test_wrong_server_backend_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = ProtocolFixture(Path(temp))
            fixture.write()
            first = analyze.expected_arms()[0]
            path = fixture.root / f"server-{first.label}.log"
            path.write_text(
                "requested_decode_storage_backend=dense\nfixture server output\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected backend marker"):
                analyze.validate_protocol(fixture.root)


class FakeHostgate:
    NVRM_WINDOW_MAX = 50
    NVRM_TOTAL_MAX = 400
    MEM_FLOOR_GIB = 32.0

    def __init__(self, *, always_dirty: bool = False) -> None:
        self.attempt = -1
        self.always_dirty = always_dirty

    def wait_quiet(self, *, log: object) -> int:
        del log
        self.attempt += 1
        return 0

    def driver_gate(self, *, log: object) -> tuple[int, int, int]:
        del log
        return 0, 0, 0

    def mem_gate(self, *, log: object) -> tuple[int, float]:
        del log
        return 0, 64.0

    def ci_job_running(self) -> bool:
        return False

    def busy_procs(self) -> list[str]:
        return ["rustc"] if self.always_dirty or self.attempt == 0 else []

    def foreign_model_procs(self) -> list[str]:
        return []

    def mem_available_gib(self) -> float:
        return 64.0

    def nvrm_counts(self) -> tuple[int, int]:
        return 0, 0


class QuietGateTests(unittest.TestCase):
    def test_post_wait_race_retries_until_clean(self) -> None:
        fake = FakeHostgate()
        record = gate_driver.wait_for_accepted_quiet(fake, max_post_wait_rechecks=2)
        self.assertTrue(record["accepted_quiet"])
        self.assertEqual(record["quiet_rechecks"], 1)
        self.assertEqual(fake.attempt, 1)

    def test_repeated_dirty_snapshots_fail_closed(self) -> None:
        fake = FakeHostgate(always_dirty=True)
        with self.assertRaisesRegex(RuntimeError, "refusing to start"):
            gate_driver.wait_for_accepted_quiet(fake, max_post_wait_rechecks=1)


if __name__ == "__main__":
    unittest.main()
