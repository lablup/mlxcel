"""A manager that never spawned a child must not remove another owner's path."""

from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

import pytest

import mlxcel
from mlxcel._server import ManagedServer


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix socket lifecycle")
@pytest.mark.parametrize("path_kind", ["regular", "live_socket"])
@pytest.mark.parametrize("caller", ["manager_close", "sync_constructor", "async_constructor"])
def test_unstarted_manager_preserves_existing_path(path_kind: str, caller: str) -> None:
    # Keep sun_path short even when pytest's own temporary directory is long.
    with tempfile.TemporaryDirectory(prefix="mlxcel-owner-", dir="/tmp") as directory:
        root = Path(directory)
        path = root / "existing.sock"
        live = None
        if path_kind == "regular":
            path.write_text("owned by the test, not the manager")
        else:
            live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            live.bind(str(path))
            live.listen(1)
            live.settimeout(1.0)

        try:
            if caller == "manager_close":
                manager = ManagedServer("unused/model", socket_path=str(path))
                assert manager._proc is None
                manager.close()
                manager.close()  # idempotent, including before start
            else:
                cls = mlxcel.LLM if caller == "sync_constructor" else mlxcel.AsyncLLM
                with pytest.raises(mlxcel.MlxcelServerError, match="binary not found"):
                    cls("unused/model", binary=str(root / "absent-binary"), socket=str(path))

            assert path.exists()
            if live is None:
                assert path.read_text() == "owned by the test, not the manager"
            else:
                # Check the listener is still reachable, not merely its path.
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(1.0)
                    client.connect(str(path))
                    accepted, _ = live.accept()
                    accepted.close()
        finally:
            if live is not None:
                live.close()
