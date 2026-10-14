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
from typing import Any


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


def quiet_snapshot(gate: Any) -> dict[str, object]:
    """Capture every host condition that can invalidate a timed arm."""

    nvrm_window, nvrm_total = gate.nvrm_counts()
    return {
        "ci_job_running": gate.ci_job_running(),
        "busy_procs": gate.busy_procs(),
        "foreign_model_procs": gate.foreign_model_procs(),
        "load1": os.getloadavg()[0],
        "mem_available_gib": gate.mem_available_gib(),
        "mem_floor_gib": gate.MEM_FLOOR_GIB,
        "nvrm_window_5m": nvrm_window,
        "nvrm_window_max": gate.NVRM_WINDOW_MAX,
        "nvrm_total_since_boot": nvrm_total,
        "nvrm_total_max": gate.NVRM_TOTAL_MAX,
    }


def snapshot_is_quiet(snapshot: dict[str, object]) -> bool:
    """Return whether a final post-wait snapshot satisfies every host gate."""

    return (
        snapshot["ci_job_running"] is False
        and snapshot["busy_procs"] == []
        and snapshot["foreign_model_procs"] == []
        and float(snapshot["mem_available_gib"]) >= float(snapshot["mem_floor_gib"])
        and int(snapshot["nvrm_window_5m"]) < int(snapshot["nvrm_window_max"])
        and int(snapshot["nvrm_total_since_boot"]) < int(snapshot["nvrm_total_max"])
    )


def wait_for_accepted_quiet(
    gate: Any,
    *,
    max_post_wait_rechecks: int,
) -> dict[str, object]:
    """Wait for quiet and retry when activity appears in the post-wait race."""

    quiet_wait_s = 0
    driver_wait_s = 0
    mem_wait_s = 0
    for recheck in range(max_post_wait_rechecks + 1):
        quiet_wait_s += gate.wait_quiet(log=sys.stderr)
        waited, _window, _total = gate.driver_gate(log=sys.stderr)
        driver_wait_s += waited
        waited, _available = gate.mem_gate(log=sys.stderr)
        mem_wait_s += waited
        snapshot = quiet_snapshot(gate)
        if snapshot_is_quiet(snapshot):
            snapshot.update(
                accepted_quiet=True,
                quiet_rechecks=recheck,
                gate_wait_s=quiet_wait_s,
                driver_wait_s=driver_wait_s,
                mem_wait_s=mem_wait_s,
            )
            return snapshot
        print(
            "[gate] activity appeared after the sustained wait; retrying: "
            f"ci={snapshot['ci_job_running']} busy={snapshot['busy_procs']} "
            f"foreign={snapshot['foreign_model_procs']} "
            f"mem={float(snapshot['mem_available_gib']):.1f}GiB "
            f"nvrm={snapshot['nvrm_window_5m']}/{snapshot['nvrm_total_since_boot']}",
            file=sys.stderr,
            flush=True,
        )
    raise RuntimeError(
        "host became nonquiet after every sustained wait; refusing to start a timed arm"
    )


def main() -> int:
    """Gate one arm, fail closed on port conflicts, and append its evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hostgate-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--port-only", action="store_true")
    parser.add_argument("--max-post-wait-rechecks", type=int, default=12)
    args = parser.parse_args()

    if args.max_post_wait_rechecks < 0:
        parser.error("--max-post-wait-rechecks must be non-negative")

    started = utc_now()
    assert_port_free(args.port)
    record: dict[str, object] = {
        "label": args.label,
        "started_utc": started,
        "port": args.port,
        "port_free_before_wait": True,
    }

    if args.port_only:
        assert_port_free(args.port)
        record["port_free_after_wait"] = True
    else:
        sys.path.insert(0, str(args.hostgate_dir))
        gate = importlib.import_module("hostgate")
        record.update(
            wait_for_accepted_quiet(
                gate,
                max_post_wait_rechecks=args.max_post_wait_rechecks,
            )
        )
        assert_port_free(args.port)
        record["port_free_after_wait"] = True

    record["completed_utc"] = utc_now()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
