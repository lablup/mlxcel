#!/usr/bin/env python3
"""Run the repository's quiet-host gate and record per-arm evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import socket
import sys
from pathlib import Path


def assert_port_free(port: int) -> None:
    """Fail if the benchmark's IPv4 loopback port is already in use."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"benchmark port {port} is already in use: {exc}") from exc


def utc_now() -> str:
    """Return the current UTC time in the environment record's format."""

    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    """Gate one arm, fail closed on port conflicts, and append its evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hostgate-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--port-only", action="store_true")
    args = parser.parse_args()

    started = utc_now()
    assert_port_free(args.port)
    record: dict[str, object] = {
        "label": args.label,
        "started_utc": started,
        "port": args.port,
        "port_free_before_wait": True,
    }

    if not args.port_only:
        sys.path.insert(0, str(args.hostgate_dir))
        gate = importlib.import_module("hostgate")
        record["gate_wait_s"] = gate.wait_quiet(log=sys.stderr)
        assert_port_free(args.port)
        record.update(
            port_free_after_wait=True,
            ci_job_running=gate.ci_job_running(),
            busy_procs=gate.busy_procs(),
            foreign_model_procs=gate.foreign_model_procs(),
            load1=os.getloadavg()[0],
            mem_available_gib=gate.mem_available_gib(),
        )
        nvrm_window, nvrm_total = gate.nvrm_counts()
        record.update(nvrm_window_5m=nvrm_window, nvrm_total_since_boot=nvrm_total)

    record["completed_utc"] = utc_now()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
