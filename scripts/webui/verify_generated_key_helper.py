#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import json, os, platform, pty, select, signal, subprocess, sys, time
from pathlib import Path
from typing import Any
import verify_installed_artifact as inst

def close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass

def terminate_process(proc: subprocess.Popen[bytes], sigint_timeout: float = 15, sigkill_timeout: float = 5) -> dict[str, Any]:
    forced = False
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=sigint_timeout)
        except subprocess.TimeoutExpired:
            forced = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=sigkill_timeout)
            except subprocess.TimeoutExpired:
                return {"returncode": proc.returncode, "forced": forced, "cleanup_error": "generated-key server did not reap after SIGKILL deadline"}
    return {"returncode": proc.returncode, "forced": forced}

def run(config: dict[str, Any]) -> dict[str, Any]:
    if platform.system() == "Windows":
        return {"status": "unsupported", "reason": "pty unavailable on Windows"}
    work = Path(config["work"]); port = int(config["port"]); cmd = list(config["cmd"])
    env = os.environ.copy(); env["RUST_LOG"] = "info"
    master_fd: int | None = None; slave_fd: int | None = None; proc: subprocess.Popen[bytes] | None = None
    transcript = bytearray(); key: str | None = None; cleanup: dict[str, Any] = {"returncode": None}
    try:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, start_new_session=True, close_fds=True)
        close_fd(slave_fd); slave_fd = None
        os.set_blocking(master_fd, False)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise inst.generated_key_error("exited-before-key", proc.returncode, transcript)
            if select.select([master_fd], [], [], 0.1)[0]:
                try:
                    transcript.extend(os.read(master_fd, 65536))
                except (BlockingIOError, OSError):
                    pass
                if len(transcript) > 1024 * 1024:
                    raise AssertionError("generated-key terminal output exceeded 1 MiB")
            match = inst.GENERATED_KEY_RE.search(transcript)
            if match:
                key = match.group(1).decode()
                break
        if not key:
            raise inst.generated_key_error("missing-recognized-key", None, transcript)
        base = f"http://127.0.0.1:{port}/gen/ui-api/v1/bootstrap"
        assert inst.request(base, key=key)[0] == 200
        assert inst.request(base)[0] == 401
    finally:
        if proc is not None:
            cleanup = terminate_process(proc)
        close_fd(slave_fd); close_fd(master_fd)
        if cleanup.get("cleanup_error"):
            raise AssertionError(cleanup["cleanup_error"])
    if cleanup["returncode"] != 0:
        raise AssertionError(f"generated-key server exited {cleanup['returncode']}")
    return {"tty_key_authenticated": True, "key_in_argv_or_env": False, "returncode": cleanup["returncode"], "forced": cleanup["forced"]}

def main() -> int:
    config = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(run(config), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
