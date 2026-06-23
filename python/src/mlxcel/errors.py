"""Exception types raised by the mlxcel client.

Lifecycle concerns (binary discovery, server startup, readiness, crash) raise
the ``Mlxcel*`` exceptions defined here. HTTP and API errors from the server
propagate as native ``openai`` SDK exceptions so callers keep full visibility
into status codes and response bodies.
"""

from __future__ import annotations


class MlxcelError(Exception):
    """Base class for all mlxcel client errors."""


class MlxcelServerError(MlxcelError):
    """A managed ``mlxcel serve`` process failed to launch, become ready, or stay alive.

    When the failure was observed against a running child process, ``stderr``
    holds the tail of the captured server log to aid debugging.
    """

    def __init__(self, message: str, stderr: str | None = None) -> None:
        self.stderr = stderr
        if stderr:
            message = f"{message}\n\n--- mlxcel server stderr (tail) ---\n{stderr}"
        super().__init__(message)


class MlxcelTimeoutError(MlxcelServerError):
    """The managed server did not become ready within the configured timeout."""


__all__ = ["MlxcelError", "MlxcelServerError", "MlxcelTimeoutError"]
