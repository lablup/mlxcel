#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Measure empty-router production startup for issue #1848.

This helper launches the already-built Rust artifacts only; it does not use Vite,
load a model, or drive a browser. For each entrypoint it records serial startup
observations for both --webui and --no-webui: elapsed time from process launch to
root /health returning 200, and RSS sampled immediately after that readiness
point. The script records observations, not pass/fail budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_BODY = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    label: str
    binary: Path
    command_prefix: list[str]
    sha256: str
    build_source_head: str
    features: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(value: Any, secrets_to_hide: list[str]) -> Any:
    if isinstance(value, str):
        text = value
        for secret in secrets_to_hide:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text
    if isinstance(value, list):
        return [redact(item, secrets_to_hide) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, secrets_to_hide) for key, item in value.items()}
    return value


def write_private(path: Path, body: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fp:
        fp.write(body)
    assert_private_file(path)


def assert_private_file(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise AssertionError(f"{path} mode is {mode:o}, want 600")


def mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o700)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o700:
        raise AssertionError(f"{path} mode is {mode:o}, want 700")


def open_private_binary(path: Path) -> Any:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    file = os.fdopen(fd, "ab", buffering=0)
    assert_private_file(path)
    return file


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def clean_env(home: Path, store: Path) -> dict[str, str]:
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
    env.update(
        {
            "HOME": str(home),
            "MLXCEL_MODELS_DIR": str(store),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    return env


def request_health(base: str) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{base}/health", method="GET")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2) as resp:
            return resp.status, resp.read(MAX_BODY + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_BODY + 1)


def wait_root_health(base: str, proc: subprocess.Popen[bytes], timeout_secs: float) -> None:
    start = time.perf_counter()
    deadline = start + timeout_secs
    last = ""
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health became ready with {proc.returncode}: {last}")
        try:
            status, body = request_health(base)
            if len(body) > MAX_BODY:
                raise RuntimeError("/health response exceeded bounded buffer")
            last = body[:256].decode(errors="replace")
            if status == 200:
                return
            last = f"status {status}: {last}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.05)
    raise TimeoutError(f"/health did not return 200 within {timeout_secs:.1f}s: {last}")


def sample_rss_kib(pid: int, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> int | None:
    result = runner(["ps", "-o", "rss=", "-p", str(pid)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    if result.returncode != 0:
        return None
    text = result.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return int(text[-1].strip())
    except ValueError:
        return None


def build_command(artifact: Artifact, port: int, models: Path, store: Path, key_file: Path, webui: bool) -> list[str]:
    return [
        *artifact.command_prefix,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--models-dir",
        str(models),
        "--model-store-root",
        str(store),
        "--api-key-file",
        str(key_file),
        "--api-prefix",
        "/lab",
        "--no-models-autoload",
        "--webui" if webui else "--no-webui",
    ]


def public_command(command: list[str], key_file: Path) -> list[str]:
    rendered: list[str] = []
    skip = False
    for item in command:
        if skip:
            rendered.append("<private-key-file>")
            skip = False
            continue
        rendered.append(item)
        if item == "--api-key-file":
            skip = True
    return [Path(rendered[0]).name, *rendered[1:]] if rendered else rendered


def stop_proc(proc: subprocess.Popen[bytes], log_file: Any, log_path: Path, timeout_secs: float) -> dict[str, Any]:
    forced = False
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                forced = True
                proc.kill()
                proc.wait(timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"server exited with {proc.returncode}; log: {log_path}")
        if forced:
            raise RuntimeError(f"server required forced shutdown; log: {log_path}")
        return {"exit_code": proc.returncode, "forced": forced, "log": str(log_path)}
    finally:
        close = getattr(log_file, "close", None)
        if close:
            close()


def launch_process(command: list[str], work: Path, env: dict[str, str], log_file: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, cwd=work, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def run_one(
    artifact: Artifact,
    *,
    webui: bool,
    repeat_index: int,
    root: Path,
    timeout_secs: float,
    shutdown_timeout_secs: float,
    secrets_to_hide: list[str],
    launcher: Callable[[list[str], Path, dict[str, str], Any], subprocess.Popen[bytes]] = launch_process,
    waiter: Callable[[str, subprocess.Popen[bytes], float], Any] = wait_root_health,
    rss_sampler: Callable[[int], int | None] = sample_rss_kib,
    port_picker: Callable[[], int] = free_port,
) -> dict[str, Any]:
    mode = "webui-on" if webui else "webui-off"
    work = root / artifact.label / mode / f"repeat-{repeat_index:02d}"
    for dirname in ("home", "empty-models", "store"):
        mkdir_private(work / dirname)
    key = secrets.token_urlsafe(32)
    secrets_to_hide.append(key)
    key_file = work / "api-key.txt"
    write_private(key_file, key + "\n")
    port = port_picker()
    command = build_command(artifact, port, work / "empty-models", work / "store", key_file, webui)
    log_path = work / "server.log"
    log_file = open_private_binary(log_path)
    proc: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] = {
        "entrypoint": artifact.label,
        "mode": mode,
        "repeat": repeat_index,
        "command": public_command(command, key_file),
        "health_url": f"http://127.0.0.1:{port}/health",
        "models_dir": str(work / "empty-models"),
        "model_store_root": str(work / "store"),
        "measurement": "elapsed milliseconds from process launch to root /health 200; RSS sampled immediately after readiness before any model load",
    }
    try:
        launch_started = time.perf_counter()
        proc = launcher(command, work, clean_env(work / "home", work / "store"), log_file)
        result["pid"] = proc.pid
        waiter(f"http://127.0.0.1:{port}", proc, timeout_secs)
        result["startup_ms_to_root_health_200"] = round((time.perf_counter() - launch_started) * 1000.0, 3)
        rss_kib = rss_sampler(proc.pid)
        if not isinstance(rss_kib, int) or rss_kib <= 0:
            raise AssertionError(f"RSS measurement unavailable for pid {proc.pid}")
        result["rss_kib_after_health_200"] = rss_kib
    finally:
        try:
            if proc is not None:
                result["shutdown"] = stop_proc(proc, log_file, log_path, shutdown_timeout_secs)
            else:
                log_file.close()
        finally:
            key_file.unlink(missing_ok=True)
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if key in log_text:
        raise AssertionError(f"private API key leaked to {log_path}")
    if key_file.exists():
        raise AssertionError(f"private API key file was not removed: {key_file}")
    return redact(result, secrets_to_hide)


def git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_artifacts(args: argparse.Namespace) -> list[Artifact]:
    artifacts = []
    for label, raw, subcommand in [("mlxcel-server", args.server_bin, []), ("mlxcel-serve", args.cli_bin, ["serve"])]:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            Artifact(
                label=label,
                binary=path,
                command_prefix=[str(path), *subcommand],
                sha256=sha256(path),
                build_source_head=args.build_source_head,
                features=args.features,
            )
        )
    return artifacts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True, help="Path to the production mlxcel-server artifact")
    parser.add_argument("--cli-bin", required=True, help="Path to the production mlxcel CLI artifact used as `mlxcel serve`")
    parser.add_argument("--evidence", default="webui-startup-measurements.json")
    parser.add_argument("--repeats", type=int, default=3, help="Serial repeats per entrypoint/mode; minimum 3")
    parser.add_argument("--timeout", type=float, default=45.0, help="Seconds to wait for root /health 200")
    parser.add_argument("--shutdown-timeout", type=float, default=15.0, help="Seconds to wait for graceful SIGINT shutdown")
    parser.add_argument("--build-source-head", default=os.environ.get("WEBUI_BUILD_SOURCE_HEAD", "unknown"))
    parser.add_argument("--features", default=os.environ.get("WEBUI_BUILD_FEATURES", "unknown"), help="Feature provenance for the measured binaries")
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 for issue #1848 startup measurements")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    evidence_path = Path(args.evidence).resolve()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = evidence_path.parent / f"{evidence_path.stem}-artifacts-{stamp}"
    mkdir_private(root)
    secrets_to_hide: list[str] = []
    source_root = Path(__file__).resolve().parents[2]
    artifacts = build_artifacts(args)
    evidence: dict[str, Any] = {
        "result": "fail",
        "scope": "production empty-router startup measurements to root /health 200; no model load, no browser, no startup budget threshold",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {"platform": platform.platform(), "system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
        "script_worktree_head": git_head(source_root),
        "artifact_dir": str(root),
        "parameters": {"repeats": args.repeats, "timeout_secs": args.timeout, "shutdown_timeout_secs": args.shutdown_timeout},
        "artifacts": [
            {
                "label": artifact.label,
                "binary": str(artifact.binary),
                "sha256": artifact.sha256,
                "command_shape": [Path(artifact.command_prefix[0]).name, *artifact.command_prefix[1:]],
                "build_source_head": artifact.build_source_head,
                "features": artifact.features,
            }
            for artifact in artifacts
        ],
        "results": [],
    }
    try:
        for artifact in artifacts:
            for webui in (True, False):
                for repeat in range(1, args.repeats + 1):
                    evidence["results"].append(
                        run_one(
                            artifact,
                            webui=webui,
                            repeat_index=repeat,
                            root=root,
                            timeout_secs=args.timeout,
                            shutdown_timeout_secs=args.shutdown_timeout,
                            secrets_to_hide=secrets_to_hide,
                        )
                    )
                    evidence_path.write_text(json.dumps(redact(evidence, secrets_to_hide), indent=2, sort_keys=True) + "\n")
        evidence["result"] = "pass"
        evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        evidence_path.write_text(json.dumps(redact(evidence, secrets_to_hide), indent=2, sort_keys=True) + "\n")
        print(json.dumps(redact(evidence, secrets_to_hide), indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        evidence["result"] = "fail"
        safe_error = {"type": type(exc).__name__, "message": redact(str(exc), secrets_to_hide)}
        evidence["error"] = safe_error
        evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        evidence_path.write_text(json.dumps(redact(evidence, secrets_to_hide), indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"result": "fail", "error_type": safe_error["type"], "message": "startup measurement failed; see sanitized evidence", "evidence": str(evidence_path)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
