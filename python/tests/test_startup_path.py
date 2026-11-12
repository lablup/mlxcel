"""Managed startup replaces a stale socket and nothing else at its path."""

from __future__ import annotations

import asyncio
import os
import socket
import stat
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

import mlxcel
from mlxcel._server import ManagedServer

FAKE_SERVER = Path(__file__).resolve().parent / "fake_server.py"

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix socket lifecycle")

KEPT = ["regular", "directory", "symlink", "live_socket", "datagram_socket", "no_permission"]
if sys.platform == "linux":
    # A full accept queue answers EAGAIN on Linux. macOS answers ECONNREFUSED,
    # the same as for a stale socket, so the case cannot be recognised there.
    KEPT.append("full_backlog")


@pytest.fixture
def root() -> Iterator[Path]:
    # Keep sun_path short even when pytest's own temporary directory is long.
    with tempfile.TemporaryDirectory(prefix="mlxcel-start-", dir="/tmp") as directory:
        yield Path(directory)


def _executable(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _start(caller: str, binary: str, path: Path) -> None:
    if caller == "manager":
        manager = ManagedServer(
            "unused/model",
            binary=binary,
            socket_path=str(path),
            startup_timeout=2.0,
            shutdown_grace=0.1,
        )
        try:
            manager.start()
        finally:
            manager.close()
    elif caller == "sync_constructor":
        mlxcel.LLM(
            "unused/model", binary=binary, socket=str(path), startup_timeout=2.0, shutdown_grace=0.1
        )
    else:
        mlxcel.AsyncLLM(
            "unused/model", binary=binary, socket=str(path), startup_timeout=2.0, shutdown_grace=0.1
        )


@pytest.mark.parametrize("path_kind", KEPT)
@pytest.mark.parametrize("caller", ["manager", "sync_constructor", "async_constructor"])
def test_startup_keeps_an_entry_that_is_not_a_stale_socket(
    path_kind: str, caller: str, root: Path
) -> None:
    path = root / "existing.sock"
    marker = root / "child-started"
    # The child only records that it ran. It never binds or unlinks, so what
    # happens to the path below is the Python client's doing alone.
    child = _executable(
        root / "child",
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('started')\n"
        "raise SystemExit(1)\n",
    )
    owner: Optional[socket.socket] = None
    queued: List[socket.socket] = []
    if path_kind == "regular":
        path.write_text("another owner's file")
    elif path_kind == "directory":
        path.mkdir()
    elif path_kind == "symlink":
        path.symlink_to(root / "absent-target")
    elif path_kind == "datagram_socket":
        owner = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        owner.bind(str(path))
    elif path_kind == "no_permission":
        # A leftover socket this user may not connect to: the probe is denied,
        # so nothing is known about its owner.
        leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        leftover.bind(str(path))
        leftover.close()
        path.chmod(0)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as check:
            try:
                check.connect(str(path))
            except PermissionError:
                pass
            except OSError:
                pytest.skip("this user is not denied by file permissions")
    else:
        owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        owner.bind(str(path))
        owner.listen(0 if path_kind == "full_backlog" else 1)
        owner.settimeout(1.0)
        if path_kind == "full_backlog":
            # Fill the accept queue until the kernel stops taking connections.
            for _ in range(16):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.setblocking(False)
                queued.append(client)
                try:
                    client.connect(str(path))
                except BlockingIOError:
                    break
            else:
                pytest.fail("the accept queue never filled")

    before = path.lstat()
    try:
        with pytest.raises(mlxcel.MlxcelServerError) as caught:
            _start(caller, child, path)

        message = str(caught.value)
        assert not marker.exists(), f"the child was started: {message}"
        assert os.path.lexists(path), f"the entry was removed: {message}"
        after = path.lstat()
        assert (after.st_dev, after.st_ino, after.st_mode) == (
            before.st_dev,
            before.st_ino,
            before.st_mode,
        )
        assert str(path) in message
        if path_kind in ("regular", "directory", "symlink"):
            assert "not a socket" in message
        elif path_kind in ("datagram_socket", "no_permission"):
            assert "Cannot tell whether" in message
        else:
            assert "in use by a running listener" in message
        if path_kind == "regular":
            assert path.read_text() == "another owner's file"
        elif path_kind == "symlink":
            assert os.readlink(path) == str(root / "absent-target")
        elif path_kind == "live_socket":
            assert owner is not None
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(1.0)
                client.connect(str(path))
                accepted, _ = owner.accept()
                accepted.close()
    finally:
        for client in queued:
            client.close()
        if owner is not None:
            owner.close()


@pytest.mark.parametrize("caller", ["manager", "sync_constructor", "async_constructor"])
def test_startup_replaces_a_socket_nobody_listens_on(caller: str, root: Path) -> None:
    path = root / "stale.sock"
    bound = root / "child-bound"
    # This child stands in for a server that does not clear the path itself:
    # its bind fails with "address already in use" unless the client removed
    # the leftover socket first.
    child = _executable(
        root / "child",
        "import socket, sys, time\n"
        "from pathlib import Path\n"
        "path = sys.argv[sys.argv.index('--host') + 1]\n"
        "server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "server.bind(path)\n"
        "server.listen(1)\n"
        f"Path({str(bound)!r}).write_text('bound')\n"
        "time.sleep(0.3)\n"
        "raise SystemExit(1)\n",
    )
    leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leftover.bind(str(path))
    leftover.close()
    before = path.lstat()
    assert stat.S_ISSOCK(before.st_mode)

    # The stand-in never answers /health and exits, so startup still reports
    # an error; what matters is that it could bind the path.
    with pytest.raises(mlxcel.MlxcelServerError) as caught:
        _start(caller, child, path)
    assert bound.exists(), f"the child could not bind the stale path: {caught.value}"


@pytest.mark.parametrize("use_async", [False, True])
def test_managed_client_serves_on_a_path_with_a_stale_socket(use_async: bool, root: Path) -> None:
    path = root / "stale.sock"
    shim = _executable(
        root / "mlxcel-shim",
        "import os, sys\n"
        f"argv = [{sys.executable!r}, {str(FAKE_SERVER)!r}] + sys.argv[1:]\n"
        f"os.execv({sys.executable!r}, argv)\n",
    )
    leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leftover.bind(str(path))
    leftover.close()

    if use_async:

        async def exercise() -> str:
            llm = mlxcel.AsyncLLM("fake/model", binary=shim, socket=str(path))
            try:
                return await llm.generate("hello")
            finally:
                await llm.close()

        assert asyncio.run(exercise()) == "hello world"
    else:
        with mlxcel.LLM("fake/model", binary=shim, socket=str(path)) as llm:
            assert llm.generate("hello") == "hello world"
    assert not os.path.lexists(path)
