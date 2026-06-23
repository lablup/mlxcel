"""Shared helpers for the sync and async clients.

Mode selection, base-URL normalization, message-type narrowing, and the
transport constants live here so :mod:`._client` (sync ``LLM``) and
:mod:`._async_client` (``AsyncLLM``) stay small and share one source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, cast

import httpx

from .errors import MlxcelError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Placeholder host used for Unix-socket transports; routing is via the uds
# transport, so the host name is never resolved over the network.
UDS_BASE = "http://mlxcel"

# Default timeouts applied when a custom http_client is injected, because the
# SDK's built-in timeouts are dropped in that case. Generous read timeout so a
# slow first token does not abort a long generation.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)

ChatMessages = Iterable[Mapping[str, Any]]


def is_managed(
    model: Optional[str],
    base_url: Optional[str],
    socket: Optional[str],
    transport: Optional[object] = None,
) -> bool:
    """Decide managed vs connect mode and validate the argument combination.

    A ``model`` means managed mode (spawn a server); ``socket=`` may accompany it
    as the bind path. A connect target (``base_url=`` or ``socket=`` without a
    model, or an injected ``transport=``) means connect mode (no subprocess).

    Returns:
        True for managed mode, False for connect mode.

    Raises:
        MlxcelError: if the arguments are ambiguous or insufficient.
    """
    if model is not None:
        # Managed mode. A base_url or injected transport would be contradictory.
        if base_url is not None or transport is not None:
            raise MlxcelError(
                "Pass either a model (managed mode) or a connect target "
                "(base_url=/transport=), not both. In managed mode, socket= is "
                "the bind path for the spawned server."
            )
        return True

    # No model: connect mode, which needs a target.
    if base_url is not None and socket is not None:
        raise MlxcelError("Pass either base_url= or socket=, not both.")
    if base_url is None and socket is None and transport is None:
        raise MlxcelError(
            "A model is required in managed mode. Pass model=, or use connect "
            "mode with base_url=, socket=, or transport=."
        )
    return False


def normalize_base_url(base_url: str) -> str:
    """Ensure the OpenAI base URL ends with ``/v1``."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def native_base_url(openai_base_url: str) -> str:
    """Derive the server root (no ``/v1``) for native routes like ``/tokenize``."""
    return openai_base_url[: -len("/v1")] if openai_base_url.endswith("/v1") else openai_base_url


def connect_base_url(base_url: Optional[str]) -> str:
    """Resolved OpenAI base URL for a connect-mode client (UDS placeholder otherwise)."""
    return normalize_base_url(base_url) if base_url else f"{UDS_BASE}/v1"


def as_messages(messages: ChatMessages) -> "List[ChatCompletionMessageParam]":
    """Materialize caller messages as the OpenAI message-param list type.

    The public API accepts loosely-typed mappings (plain dicts) for ergonomics;
    the OpenAI SDK wants its precise TypedDict union. The shapes are compatible
    at runtime, so this only narrows the static type.
    """
    return cast("List[ChatCompletionMessageParam]", list(messages))


__all__ = [
    "UDS_BASE",
    "DEFAULT_TIMEOUT",
    "ChatMessages",
    "is_managed",
    "normalize_base_url",
    "native_base_url",
    "connect_base_url",
    "as_messages",
]
