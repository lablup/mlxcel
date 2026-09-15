#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Installed WebUI artifact smoke for issue #1848.

This script launches the shipped Rust server binary from an isolated working
folder, not Vite preview, and verifies model-free WebUI serving, auth, CSP,
prefix handling, hostile-origin rejection, and UI-disabled behavior. It never
loads a checkpoint; hardware/model inference gates stay separate.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def request(url: str, *, key: str | None = None, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    all_headers = dict(headers or {})
    if key is not None:
        all_headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, method=method, headers=all_headers)
    try:
        with OPENER.open(req, timeout=5) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def wait_ready(base: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 20
    last = b""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}: {last.decode(errors='replace')}")
        try:
            status, _, body = request(f"{base}/health")
            last = body
            if status == 200:
                return
        except Exception as exc:  # noqa: BLE001 - diagnostic for startup polling
            last = str(exc).encode()
        time.sleep(0.1)
    raise RuntimeError(f"server did not become ready: {last.decode(errors='replace')}")


def clean_env(cwd: Path, store: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "SystemRoot",
        "WINDIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "TMPDIR",
        "TMP",
        "TEMP",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed and not key.lower().endswith("proxy")}
    env["HOME"] = str(cwd / "home")
    env["MLXCEL_MODELS_DIR"] = str(store)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def launch(command: list[str], cwd: Path, key_file: Path, port: int, models_dir: Path, store: Path, webui: bool) -> tuple[subprocess.Popen[bytes], object, Path]:
    args = command + [
        "--host", "127.0.0.1",
        "--port", str(port),
        "--models-dir", str(models_dir),
        "--model-store-root", str(store),
        "--api-key-file", str(key_file),
        "--api-prefix", "/lab",
        "--no-models-autoload",
        "--settings",
        "--props",
        "--metrics",
        "--webui" if webui else "--no-webui",
    ]
    log_path = cwd / ("server-webui-on.log" if webui else "server-webui-off.log")
    log_file = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(args, cwd=cwd, env=clean_env(cwd, store), stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path


def stop_proc(proc: subprocess.Popen[bytes], log_file: object, log_path: Path) -> None:
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait(timeout=5)
                raise RuntimeError(f"server did not stop after SIGINT and was force-killed; log: {log_path}") from exc
        if proc.returncode not in (0, None):
            raise RuntimeError(f"server exited with {proc.returncode}; log: {log_path}")
    finally:
        close = getattr(log_file, "close", None)
        if close is not None:
            close()


def run_one(label: str, command: list[str], root: Path) -> dict[str, object]:
    work = root / label
    work.mkdir(parents=True)
    (work / "home").mkdir()
    models = work / "empty-models"
    store = work / "store"
    models.mkdir()
    store.mkdir()
    key = secrets.token_urlsafe(32)
    key_file = work / "api-key.txt"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fp:
        fp.write(key)
    if stat.S_IMODE(key_file.stat().st_mode) != 0o600:
        raise AssertionError("api key file must be private 0600")

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc, log_file, log_path = launch(command, work, key_file, port, models, store, True)
    try:
        wait_ready(base, proc)
        status, headers, body = request(f"{base}/lab/webui/")
        assert status == 200, (status, body[:200])
        csp = headers.get("content-security-policy", "")
        assert "style-src 'self'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp, csp
        status, _, body = request(f"{base}/lab/ui-api/v1/bootstrap", key=key)
        assert status == 200, (status, body[:200])
        bootstrap = json.loads(body)
        assert bootstrap["server"]["api_base"] == "/lab", bootstrap["server"]
        status, _, _ = request(f"{base}/lab/ui-api/v1/catalog")
        assert status == 401, status
        status, _, _ = request(f"{base}/lab/ui-api/v1/model-actions", key=key, method="POST", data=b"{}", headers={"Host": "foreign.invalid", "Origin": "http://foreign.invalid", "Sec-Fetch-Site": "cross-site", "Content-Type": "application/json"})
        assert status == 403, status
        status, _, _ = request(f"{base}/lab/v1/chat/completions", key=key, method="POST", data=b"{}", headers={"Host": "foreign.invalid", "Origin": "http://foreign.invalid", "Sec-Fetch-Site": "cross-site", "Content-Type": "application/json"})
        assert status == 403, status
    finally:
        try:
            stop_proc(proc, log_file, log_path)
        finally:
            key_file.unlink(missing_ok=True)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fp:
        fp.write(key)
    proc, log_file, log_path = launch(command, work, key_file, port, models, store, False)
    try:
        wait_ready(base, proc)
        status, _, _ = request(f"{base}/health")
        assert status == 200, status
        status, _, _ = request(f"{base}/lab/webui/")
        assert status in (401, 404), status
        status, _, _ = request(f"{base}/lab/ui-api/v1/bootstrap", key=key)
        assert status in (400, 404), status
    finally:
        try:
            stop_proc(proc, log_file, log_path)
        finally:
            key_file.unlink(missing_ok=True)
    return {"label": label, "command": command, "webui_on": "passed", "webui_off": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", default=os.environ.get("WEBUI_SERVER_BIN"), help="Path to installed mlxcel-server artifact")
    parser.add_argument("--cli-bin", default=os.environ.get("WEBUI_CLI_BIN"), help="Optional path to installed mlxcel CLI artifact; exercises `mlxcel serve`")
    parser.add_argument("--evidence", default=os.environ.get("WEBUI_INSTALLED_EVIDENCE", "webui-installed-evidence.json"))
    args = parser.parse_args()
    if not args.server_bin:
        parser.error("--server-bin or WEBUI_SERVER_BIN is required; installed-artifact CI must not fall back to Vite")
    with tempfile.TemporaryDirectory(prefix="mlxcel-webui-installed-") as tmp:
        root = Path(tmp)
        results = [run_one("mlxcel-server", [str(Path(args.server_bin).resolve())], root)]
        if args.cli_bin:
            results.append(run_one("mlxcel-serve", [str(Path(args.cli_bin).resolve()), "serve"], root))
    evidence = {"scope": "installed Rust artifact, model-free WebUI serving/security smoke", "results": results}
    Path(args.evidence).write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
