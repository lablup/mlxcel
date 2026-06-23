"""Lifecycle management for a local ``mlxcel serve`` subprocess.

``ManagedServer`` discovers the binary, picks a transport (Unix domain socket on
POSIX, TCP elsewhere or on request), spawns the server, waits until ``/health``
reports ready, forwards the server's stderr to the ``mlxcel.server`` logger, and
shuts the process down cleanly. The server loads and warms up the model *before*
it binds its listener, so a first-run weight download can keep the socket
unavailable for minutes; readiness is therefore driven by polling ``/health``
with a generous timeout while watching the child process for an early exit.
"""

from __future__ import annotations

import atexit
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import httpx

from .errors import MlxcelServerError, MlxcelTimeoutError

logger = logging.getLogger("mlxcel.server")

# Conservative Unix domain socket path limit. The kernel struct sun_path is
# 104 bytes on macOS and 108 on Linux; 100 leaves headroom for the trailing NUL
# and keeps a single rule across platforms.
_MAX_SOCKET_PATH = 100

# Number of trailing stderr lines retained for crash/error reporting.
_STDERR_TAIL_LINES = 50

_IS_WINDOWS = sys.platform.startswith("win")


def _find_binary(binary: Optional[str]) -> str:
    """Resolve the ``mlxcel`` executable.

    Order: explicit ``binary`` argument, then the ``MLXCEL_BIN`` environment
    variable, then ``mlxcel`` on ``PATH``.

    Raises:
        MlxcelServerError: if no usable binary is found.
    """
    candidate = binary or os.environ.get("MLXCEL_BIN")
    if candidate:
        resolved = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if resolved:
            return resolved
        raise MlxcelServerError(
            f"mlxcel binary not found at {candidate!r}. "
            "Pass binary=... or set MLXCEL_BIN to the built executable."
        )

    found = shutil.which("mlxcel")
    if found:
        return found

    raise MlxcelServerError(
        "Could not find the 'mlxcel' executable. Install it (e.g. via Homebrew: "
        "'brew install lablup/tap/mlxcel'), build it from source with "
        "'make release' and add target/release to PATH, pass binary=..., or set "
        "the MLXCEL_BIN environment variable."
    )


def _free_tcp_port() -> int:
    """Bind an ephemeral TCP port, then release it so the child can claim it.

    There is an unavoidable race between releasing the port and the child
    rebinding it, but it is small for a local single-launch flow.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _default_socket_path() -> str:
    """A short, unique Unix socket path under ``/tmp``.

    ``/tmp`` is used deliberately rather than ``tempfile.gettempdir()``: on
    macOS the latter resolves to a long ``/var/folders/...`` path that can blow
    past the sun_path limit.
    """
    return f"/tmp/mlxcel-{os.getpid()}-{secrets.token_hex(3)}.sock"


class ManagedServer:
    """Spawn and supervise a local ``mlxcel serve`` process.

    Either a Unix domain socket (default on POSIX) or TCP is used. After
    construction, call :meth:`start` to launch and block until ready, then read
    :attr:`base_url`, :attr:`uds_path`, and :attr:`api_key` to build a client.
    """

    def __init__(
        self,
        model: str,
        *,
        binary: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        socket_path: Optional[str] = None,
        api_key: Optional[str] = None,
        ctx_size: Optional[int] = None,
        n_predict: Optional[int] = None,
        alias: Optional[str] = None,
        warmup: Optional[bool] = None,
        extra_args: Optional[List[str]] = None,
        startup_timeout: float = 600.0,
        shutdown_grace: float = 5.0,
    ) -> None:
        self.model = model
        self._binary_arg = binary
        self.api_key = api_key
        self._ctx_size = ctx_size
        self._n_predict = n_predict
        self._alias = alias
        self._warmup = warmup
        self._extra_args = list(extra_args or [])
        self._startup_timeout = startup_timeout
        self._shutdown_grace = shutdown_grace

        self._proc: Optional[subprocess.Popen[str]] = None
        self._log_thread: Optional[threading.Thread] = None
        self._stderr_tail: Deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._closed = False
        self._atexit_registered = False

        # Transport resolution. TCP is forced on Windows or when host/port given.
        self.uds_path: Optional[str] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None

        use_tcp = _IS_WINDOWS or host is not None or port is not None
        if use_tcp:
            self.host = host or "127.0.0.1"
            self.port = port if port is not None else _free_tcp_port()
        else:
            path = socket_path or _default_socket_path()
            if len(path) > _MAX_SOCKET_PATH:
                raise MlxcelServerError(
                    f"Unix socket path is too long ({len(path)} > {_MAX_SOCKET_PATH} bytes): "
                    f"{path!r}. Pass a shorter socket=... path (e.g. under /tmp)."
                )
            self.uds_path = path

    @property
    def base_url(self) -> str:
        """OpenAI base URL (with the ``/v1`` suffix) for this transport.

        For a Unix socket the host portion is a placeholder; the actual routing
        happens through the httpx ``uds`` transport.
        """
        if self.uds_path is not None:
            return "http://mlxcel/v1"
        return f"http://{self.host}:{self.port}/v1"

    @property
    def _health_url(self) -> str:
        if self.uds_path is not None:
            return "http://mlxcel/health"
        return f"http://{self.host}:{self.port}/health"

    def _build_command(self) -> List[str]:
        """Build the ``mlxcel serve`` argv.

        The API key is intentionally NOT placed on the command line: argv is
        world-readable via ``ps`` / ``/proc/<pid>/cmdline`` on multi-user hosts.
        It is passed to the child through the ``LLAMA_API_KEY`` environment
        variable in :meth:`start` instead.
        """
        binary = _find_binary(self._binary_arg)
        cmd: List[str] = [binary, "serve", "-m", self.model]

        if self.uds_path is not None:
            # UDS mode: --port 0 and --host reinterpreted as the socket path.
            cmd += ["--host", self.uds_path, "--port", "0"]
        else:
            assert self.host is not None and self.port is not None
            cmd += ["--host", self.host, "--port", str(self.port)]

        if self._ctx_size is not None:
            cmd += ["--ctx-size", str(self._ctx_size)]
        if self._n_predict is not None:
            cmd += ["--n-predict", str(self._n_predict)]
        if self._alias is not None:
            cmd += ["-a", self._alias]
        if self._warmup is True:
            cmd += ["--warmup"]
        elif self._warmup is False:
            cmd += ["--no-warmup"]
        cmd += self._extra_args
        return cmd

    def start(self) -> None:
        """Spawn the server and block until ``/health`` returns HTTP 200.

        Raises:
            MlxcelServerError: if the binary cannot be found, the process exits
                before becoming ready, or another launch failure occurs.
            MlxcelTimeoutError: if readiness is not reached within
                ``startup_timeout``.
        """
        if self._proc is not None:
            raise MlxcelServerError("Server already started.")

        cmd = self._build_command()
        # Safe to log: the API key is passed via the environment, not argv.
        logger.debug("Launching mlxcel server: %s", " ".join(cmd))

        # Pass the API key through the environment so it never appears in argv
        # (visible via ps / /proc/<pid>/cmdline) or in the launch log above.
        env = os.environ.copy()
        if self.api_key:
            env["LLAMA_API_KEY"] = self.api_key

        # Remove a stale socket file so bind does not fail on EADDRINUSE.
        if self.uds_path is not None and os.path.exists(self.uds_path):
            try:
                os.unlink(self.uds_path)
            except OSError:
                pass

        try:
            self._proc = subprocess.Popen(  # noqa: S603 - command built from trusted args
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise MlxcelServerError(f"Failed to launch mlxcel server: {exc}") from exc

        self._start_log_forwarding()
        self._register_atexit()

        try:
            self._wait_until_ready()
        except BaseException:
            # Any failure (including KeyboardInterrupt) during startup must not
            # leak the child process.
            self.close()
            raise

    def _start_log_forwarding(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        stderr = proc.stderr

        def _pump() -> None:
            try:
                for raw in stderr:
                    line = raw.rstrip("\n")
                    self._stderr_tail.append(line)
                    logger.info("%s", line)
            except (ValueError, OSError):
                # Pipe closed during shutdown; expected, never raise.
                return

        thread = threading.Thread(target=_pump, name="mlxcel-server-logs", daemon=True)
        thread.start()
        self._log_thread = thread

    def _stderr_snapshot(self) -> str:
        return "\n".join(self._stderr_tail)

    def _make_probe_client(self) -> httpx.Client:
        if self.uds_path is not None:
            transport = httpx.HTTPTransport(uds=self.uds_path)
            return httpx.Client(transport=transport, timeout=5.0)
        return httpx.Client(timeout=5.0)

    def _wait_until_ready(self) -> None:
        proc = self._proc
        assert proc is not None
        deadline = time.monotonic() + self._startup_timeout
        backoff = 0.05
        url = self._health_url

        with self._make_probe_client() as client:
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    raise MlxcelServerError(
                        f"mlxcel server exited with code {exit_code} before becoming ready.",
                        stderr=self._stderr_snapshot(),
                    )

                ready = self._probe_once(client, url)
                if ready:
                    return

                if time.monotonic() >= deadline:
                    raise MlxcelTimeoutError(
                        f"mlxcel server did not become ready within {self._startup_timeout:.0f}s.",
                        stderr=self._stderr_snapshot(),
                    )

                time.sleep(backoff)
                backoff = min(backoff * 1.5, 1.0)

    @staticmethod
    def _probe_once(client: httpx.Client, url: str) -> bool:
        """Return True once ``/health`` answers HTTP 200.

        Connection errors (socket file not yet present, connection refused) and
        HTTP 503 (loading / no slot) are treated as "still starting".
        """
        try:
            resp = client.get(url)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError):
            return False
        if resp.status_code == 200:
            return True
        if resp.status_code == 503:
            return False
        # Any other status is unexpected for /health; keep polling rather than
        # failing hard, the next iteration's process-liveness check will catch a
        # dead server.
        return False

    def ensure_alive(self) -> None:
        """Raise if the supervised process has died.

        Call before delegating work so a crashed server surfaces a clear
        ``MlxcelServerError`` (with stderr context) instead of an opaque
        connection error.
        """
        proc = self._proc
        if proc is None:
            raise MlxcelServerError("Server is not running.")
        exit_code = proc.poll()
        if exit_code is not None:
            raise MlxcelServerError(
                f"mlxcel server has exited with code {exit_code}.",
                stderr=self._stderr_snapshot(),
            )

    def close(self) -> None:
        """Terminate the process (SIGTERM, then SIGKILL) and clean up the socket."""
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self._shutdown_grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=self._shutdown_grace)
                except subprocess.TimeoutExpired:
                    logger.warning("mlxcel server did not exit after SIGKILL.")

        if self._log_thread is not None and self._log_thread.is_alive():
            self._log_thread.join(timeout=1.0)

        if self.uds_path is not None and os.path.exists(self.uds_path):
            try:
                os.unlink(self.uds_path)
            except OSError:
                pass

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def __del__(self) -> None:
        # Best-effort cleanup for a leaked handle. Interpreter shutdown may have
        # already torn down modules, so swallow everything.
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ManagedServer"]
