"""Lifecycle tests: spawn the stdlib fake_server via ManagedServer.

These exercise binary discovery, spawn, /health readiness polling, ready state,
stderr log capture, graceful shutdown, and the early-exit failure path, all
without the real Rust binary or a model.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
import time
from pathlib import Path

import httpx
import pytest

import mlxcel
from mlxcel._server import ManagedServer
from mlxcel.errors import MlxcelServerError

HERE = Path(__file__).resolve().parent
FAKE_SERVER = HERE / "fake_server.py"


@pytest.fixture()
def fake_binary(tmp_path: Path) -> str:
    """An executable shim that forwards args to the stdlib fake server.

    ManagedServer invokes ``<binary> serve -m <model> --host <path> --port 0``.
    The shim re-execs the fake server with the same argv tail so the fake server
    sees the real flag layout the manager produces.
    """
    shim = tmp_path / "mlxcel-shim"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, {str(FAKE_SERVER)!r}] + sys.argv[1:])\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(shim)


def _short_socket(tmp_path: Path) -> str:
    # Keep well under the sun_path limit; tmp_path can be long, so use /tmp.
    return f"/tmp/mlxcel-test-{os.getpid()}-{tmp_path.name[:6]}.sock"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="UDS not supported on Windows")
def test_managed_uds_lifecycle(
    fake_binary: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sock = _short_socket(tmp_path)
    caplog.set_level(logging.INFO, logger="mlxcel.server")

    server = ManagedServer(
        "fake/model",
        binary=fake_binary,
        socket_path=sock,
        extra_args=["--ready-after", "0.3"],
        startup_timeout=15.0,
    )
    server.start()
    try:
        assert server.uds_path == sock
        assert os.path.exists(sock)
        # /health must answer 200 after readiness.
        transport = httpx.HTTPTransport(uds=sock)
        with httpx.Client(transport=transport, base_url="http://mlxcel") as client:
            resp = client.get("http://mlxcel/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        # stderr log forwarding captured the startup line.
        assert any("Starting mlxcel server" in rec.message for rec in caplog.records)
        server.ensure_alive()
    finally:
        server.close()

    # Socket file removed and process gone after shutdown.
    assert not os.path.exists(sock)
    assert server._proc is not None and server._proc.poll() is not None


def test_managed_tcp_lifecycle(fake_binary: str) -> None:
    server = ManagedServer(
        "fake/model",
        binary=fake_binary,
        host="127.0.0.1",
        extra_args=["--ready-after", "0.2"],
        startup_timeout=15.0,
    )
    server.start()
    try:
        assert server.port is not None and server.port > 0
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://127.0.0.1:{server.port}/health")
            assert resp.status_code == 200
    finally:
        server.close()
    assert server._proc is not None and server._proc.poll() is not None


def test_early_exit_raises_with_stderr(fake_binary: str, tmp_path: Path) -> None:
    sock = _short_socket(tmp_path)
    server = ManagedServer(
        "fake/model",
        binary=fake_binary,
        socket_path=sock,
        extra_args=["--crash"],
        startup_timeout=15.0,
    )
    with pytest.raises(MlxcelServerError) as excinfo:
        server.start()
    # The captured stderr tail must be attached.
    assert "simulated startup failure" in str(excinfo.value)
    server.close()


def test_binary_not_found() -> None:
    server = ManagedServer("fake/model", binary="/nonexistent/mlxcel-binary-xyz")
    with pytest.raises(MlxcelServerError):
        server.start()


def test_socket_path_too_long(tmp_path: Path) -> None:
    long_path = "/tmp/" + ("x" * 200) + ".sock"
    with pytest.raises(MlxcelServerError):
        ManagedServer("fake/model", socket_path=long_path)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="UDS not supported on Windows")
def test_llm_managed_mode_end_to_end(fake_binary: str, tmp_path: Path) -> None:
    sock = _short_socket(tmp_path)
    with mlxcel.LLM(
        "fake/model",
        binary=fake_binary,
        socket=sock,
        extra_args=["--ready-after", "0.2"],
        startup_timeout=15.0,
    ) as llm:
        assert llm.model == "fake-model"
        assert llm.generate("hello") == "hello world"
        assert llm.chat([{"role": "user", "content": "hi"}]) == "hi there"
        assert "".join(llm.stream("x")) == "hello world"
        assert llm.tokenize("ab") == [ord("a"), ord("b")]
        assert llm.detokenize([104, 105]) == "hi"
    # After the context manager exits the socket is cleaned up.
    assert not os.path.exists(sock)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="UDS not supported on Windows")
def test_connect_mode_to_running_uds_server(fake_binary: str, tmp_path: Path) -> None:
    # Start a server with the manager, then connect to it without managing it.
    sock = _short_socket(tmp_path)
    server = ManagedServer(
        "fake/model",
        binary=fake_binary,
        socket_path=sock,
        extra_args=["--ready-after", "0.2"],
        startup_timeout=15.0,
    )
    server.start()
    try:
        client = mlxcel.LLM(socket=sock)
        try:
            assert client.model == "fake-model"
            assert client.generate("hi") == "hello world"
        finally:
            client.close()
        # The connect-mode client must NOT have stopped the server it joined.
        server.ensure_alive()
    finally:
        server.close()


def test_api_key_not_in_argv(fake_binary: str) -> None:
    # Regression: the API key must never appear on the command line (visible via
    # ps / /proc/<pid>/cmdline). It is passed through LLAMA_API_KEY instead.
    server = ManagedServer("fake/model", binary=fake_binary, host="127.0.0.1", api_key="secret")
    cmd = server._build_command()
    assert "--api-key" not in cmd
    assert "secret" not in " ".join(cmd)


def test_api_key_passed_via_env(
    fake_binary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The key reaches the child through LLAMA_API_KEY, not argv. Capture what
    # Popen receives without launching a real process.
    captured_env = {}
    captured_cmd = []

    class _StubPopen:
        def __init__(self, cmd, *args, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            captured_cmd.extend(cmd)
            # Empty stderr stream so log forwarding starts and finishes cleanly.
            self.stderr = iter(())

        def poll(self):
            # Report an immediate non-None exit so _wait_until_ready raises a
            # clean MlxcelServerError instead of polling /health forever.
            return 1

    monkeypatch.setattr("mlxcel._server.subprocess.Popen", _StubPopen)

    sock = _short_socket(tmp_path)
    server = ManagedServer("fake/model", binary=fake_binary, socket_path=sock, api_key="secret")
    # start() raises because the stubbed process "exits" immediately; we only
    # care that Popen was handed the key via env and not via argv.
    with pytest.raises(MlxcelServerError):
        server.start()

    assert captured_env.get("LLAMA_API_KEY") == "secret"
    joined = " ".join(captured_cmd)
    assert "secret" not in joined
    assert "--api-key" not in joined


def test_ensure_alive_after_crash(fake_binary: str, tmp_path: Path) -> None:
    server = ManagedServer(
        "fake/model",
        binary=fake_binary,
        host="127.0.0.1",
        extra_args=["--ready-after", "0.2"],
        startup_timeout=15.0,
    )
    server.start()
    proc = server._proc
    assert proc is not None
    proc.kill()
    proc.wait(timeout=5.0)
    # Give the manager's view a moment; ensure_alive should now raise.
    time.sleep(0.05)
    with pytest.raises(MlxcelServerError):
        server.ensure_alive()
    server.close()
