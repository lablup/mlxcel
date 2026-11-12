"""Awaited shutdown must yield without abandoning a real supervised child."""

from __future__ import annotations

import asyncio
import select
import signal
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest

from mlxcel import AsyncLLM
from mlxcel._server import ManagedServer


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX child signal handshake")
@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "context",
        "http-error",
        "http-cancel",
        "cancel",
        "cancel-twice",
        "kill",
        "worker-error",
    ],
)
def test_async_close_yields_and_finishes_reaping(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminated = tmp_path / "terminated"
    release = tmp_path / "release"
    # The event loop, not a timer, releases the child's SIGTERM handler. A
    # blocking close cannot finish gracefully: its grace period expires first.
    program = """
import pathlib, signal, sys, time
terminated, release = map(pathlib.Path, sys.argv[1:3])
def stop(*args):
    terminated.touch()
    while not release.exists():
        time.sleep(0.005)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
print('READY', flush=True)
while True:
    time.sleep(0.01)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", program, str(terminated), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = ManagedServer("fake/model", host="127.0.0.1", port=1, shutdown_grace=1.0)
    server._proc = proc
    calls = []
    real_close = server.close

    def record_close():
        calls.append(threading.get_ident())
        real_close()
        if mode == "worker-error":
            raise RuntimeError("managed close failed")

    monkeypatch.setattr(server, "close", record_close)

    async def run():
        llm = AsyncLLM(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        llm._server = server
        assert llm._http_client is not None
        real_http_close = llm._http_client.aclose
        http_entered = asyncio.Event()

        async def http_close():
            await real_http_close()
            if mode == "http-error":
                raise RuntimeError("HTTP close failed")
            if mode == "http-cancel":
                http_entered.set()
                await asyncio.Future()

        monkeypatch.setattr(llm._http_client, "aclose", http_close)

        async def close():
            if mode == "context":
                async with llm:
                    pass
            else:
                await llm.close()

        task = asyncio.create_task(close())
        try:
            if mode == "http-cancel":
                await asyncio.wait_for(http_entered.wait(), 3.0)
                task.cancel()

            deadline = asyncio.get_running_loop().time() + 3.0
            while not terminated.exists() and not task.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)

            assert terminated.exists(), "managed child was never terminated"
            assert not task.done(), "close blocked the loop until the child exited"
            assert proc.poll() is None, "child exited before the event loop could run"

            if mode in ("cancel", "cancel-twice"):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done(), "cancellation abandoned the cleanup worker"
                if mode == "cancel-twice":
                    task.cancel()
                    await asyncio.sleep(0.02)
                    assert not task.done(), "second cancellation abandoned cleanup"
            if mode != "kill":
                release.touch()

            if mode in ("http-cancel", "cancel", "cancel-twice"):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 4.0)
            elif mode == "http-error":
                with pytest.raises(RuntimeError, match="HTTP close failed"):
                    await asyncio.wait_for(task, 4.0)
            elif mode == "worker-error":
                with pytest.raises(RuntimeError, match="managed close failed"):
                    await asyncio.wait_for(task, 4.0)
            else:
                await asyncio.wait_for(task, 4.0)

            assert proc.returncode == (-signal.SIGKILL if mode == "kill" else 0)
            assert len(calls) == 1
            assert calls[0] != threading.get_ident()
            assert server._log_thread is not None and not server._log_thread.is_alive()
            await llm.close()
            assert len(calls) == 1, "idempotent close scheduled another worker"
        finally:
            release.touch()
            if not task.done():
                # Test cleanup also waits for the worker; never leak a child on
                # an assertion failure in a deliberately broken implementation.
                try:
                    await asyncio.wait_for(asyncio.shield(task), 4.0)
                except (asyncio.CancelledError, RuntimeError, asyncio.TimeoutError):
                    pass
            elif not task.cancelled():
                task.exception()
            await real_http_close()

    try:
        assert proc.stdout is not None
        assert select.select([proc.stdout], [], [], 5.0)[0], "child did not become ready"
        assert proc.stdout.readline().strip() == "READY"
        server._start_log_forwarding()
        asyncio.run(run())
    finally:
        release.touch()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5.0)
        if server._log_thread is not None:
            server._log_thread.join(timeout=2.0)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def test_async_connect_close_does_not_start_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run():
        loop = asyncio.get_running_loop()

        def unexpected_worker(*args):
            pytest.fail("connect-mode close must not schedule a managed cleanup worker")

        monkeypatch.setattr(loop, "run_in_executor", unexpected_worker)
        async with AsyncLLM(transport=httpx.MockTransport(lambda _: httpx.Response(200))):
            pass

    asyncio.run(run())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX child signal cleanup")
@pytest.mark.parametrize("mode", ["executor-shutdown", "submit-refused"])
@pytest.mark.parametrize("http_error", [False, True])
def test_close_reaps_when_the_executor_takes_no_work(
    mode: str, http_error: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A running loop whose executor refuses the cleanup worker: the default
    # executor was shut down first, or submission fails as it does while the
    # interpreter exits. The client is already marked closed at that point, so
    # a close that only raised would leave the child running with no retry.
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = ManagedServer("fake/model", host="127.0.0.1", port=1)
    server._proc = proc
    calls = []
    real_close = server.close

    def record_close():
        calls.append(threading.get_ident())
        real_close()

    monkeypatch.setattr(server, "close", record_close)

    async def run():
        loop = asyncio.get_running_loop()
        llm = AsyncLLM(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        llm._server = server
        assert llm._http_client is not None
        real_http_close = llm._http_client.aclose

        async def failing_close():
            await real_http_close()
            raise RuntimeError("HTTP close failed")

        if http_error:
            monkeypatch.setattr(llm._http_client, "aclose", failing_close)
        if mode == "executor-shutdown":
            await loop.shutdown_default_executor()
        else:

            def refuse(*args):
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

            monkeypatch.setattr(loop, "run_in_executor", refuse)

        if http_error:
            with pytest.raises(RuntimeError, match="HTTP close failed"):
                await llm.close()
        else:
            await llm.close()
        assert proc.returncode == -signal.SIGTERM, "the child was left running"
        assert calls == [threading.get_ident()], "cleanup did not run exactly once, inline"
        await llm.close()
        assert len(calls) == 1

    try:
        assert proc.stdout is not None
        assert select.select([proc.stdout], [], [], 5.0)[0]
        assert proc.stdout.readline().strip() == "READY"
        asyncio.run(run())
    finally:
        real_close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX child signal cleanup")
@pytest.mark.parametrize("http_error", [False, True])
def test_close_without_asyncio_preserves_cleanup_and_http_error(
    http_error: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This coroutine has no suspension points in its unused HTTP transport,
    # so drive it without installing an asyncio loop. A real Trio probe is
    # also retained outside the suite; Trio is not a required dependency.
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = ManagedServer("fake/model", host="127.0.0.1", port=1)
    server._proc = proc
    llm = AsyncLLM(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    llm._server = server
    assert llm._http_client is not None
    real_http_close = llm._http_client.aclose

    async def failing_close():
        await real_http_close()
        raise RuntimeError("HTTP close failed")

    if http_error:
        monkeypatch.setattr(llm._http_client, "aclose", failing_close)
    coroutine = llm.close()
    try:
        assert proc.stdout is not None
        assert select.select([proc.stdout], [], [], 5.0)[0]
        assert proc.stdout.readline().strip() == "READY"
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        if http_error:
            with pytest.raises(RuntimeError, match="HTTP close failed"):
                coroutine.send(None)
        else:
            with pytest.raises(StopIteration):
                coroutine.send(None)
        assert proc.returncode == -signal.SIGTERM
        assert llm._http_client.is_closed
    finally:
        coroutine.close()
        server.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
