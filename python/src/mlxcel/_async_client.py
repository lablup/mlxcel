"""Asynchronous client (:class:`AsyncLLM`) for an mlxcel server.

``AsyncLLM`` mirrors :class:`mlxcel.LLM` with ``async``/``await`` and async
iterators, backed by ``openai.AsyncOpenAI`` and ``httpx.AsyncClient``. The
managed-server lifecycle (spawn, readiness polling) runs synchronously inside
``__init__`` because it is a one-time blocking setup; every generation method is
async. The model id resolves lazily on the first request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterable, List, Optional, cast

import httpx
from openai import AsyncOpenAI

from ._common import (
    DEFAULT_TIMEOUT,
    UDS_BASE,
    ChatMessages,
    as_messages,
    connect_base_url,
    is_managed,
    native_base_url,
    normalize_base_url,
)
from ._sampling import build_params
from ._server import ManagedServer
from .errors import MlxcelError

if TYPE_CHECKING:
    from openai import AsyncStream
    from openai.types.chat import ChatCompletionChunk
    from openai.types.completion import Completion


class AsyncLLM:
    """Asynchronous mlxcel client mirroring :class:`mlxcel.LLM`.

    Use ``async with`` or call :meth:`close` (awaitable).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        socket: Optional[str] = None,
        api_key: Optional[str] = None,
        binary: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[httpx.Timeout] = None,
        startup_timeout: float = 600.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        **server_kwargs: Any,
    ) -> None:
        # Set _closed before any call that can raise so __del__ never sees a
        # missing attribute even when __init__ fails early (e.g. bad arg combo).
        self._closed = False
        self._server: Optional[ManagedServer] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._model: Optional[str] = None
        # Resolved API key, used to authorize native (/tokenize, /detokenize)
        # routes that bypass the OpenAI SDK's own Authorization injection. The
        # empty-or-None case stays unauthenticated to preserve the no-auth path.
        self._api_key = api_key
        timeout = timeout or DEFAULT_TIMEOUT

        managed = is_managed(model, base_url, socket, transport)
        model_override = server_kwargs.pop("model", None)

        try:
            if not managed:
                resolved_base = self._connect(base_url, socket, transport, timeout)
            else:
                assert model is not None
                resolved_base = self._spawn(
                    model,
                    binary,
                    host,
                    port,
                    socket,
                    api_key,
                    startup_timeout,
                    timeout,
                    server_kwargs,
                )

            self._client = AsyncOpenAI(
                base_url=resolved_base,
                api_key=api_key or "-",
                http_client=self._http_client,
            )
            self._base_url = resolved_base
            self._model_override = model_override
        except BaseException:
            # close() is a coroutine and cannot be awaited from __init__, so
            # perform synchronous cleanup directly. The managed subprocess is
            # the critical resource; stopping it here ensures deterministic
            # reaping even when the caller never awaits close().
            if self._server is not None:
                self._server.close()
            # Drop the async http client reference. It has never been used so
            # no active connections exist; the pool will be released when the
            # object is collected.
            self._http_client = None
            raise

    # -- setup -------------------------------------------------------------

    def _connect(
        self,
        base_url: Optional[str],
        socket: Optional[str],
        transport: Optional[httpx.AsyncBaseTransport],
        timeout: httpx.Timeout,
    ) -> str:
        """Build the connect-mode async http client and return the base URL."""
        if transport is not None:
            resolved_base = connect_base_url(base_url)
            self._http_client = httpx.AsyncClient(
                transport=transport, base_url=resolved_base, timeout=timeout
            )
        elif socket is not None:
            uds_transport = httpx.AsyncHTTPTransport(uds=socket)
            self._http_client = httpx.AsyncClient(
                transport=uds_transport, base_url=UDS_BASE, timeout=timeout
            )
            resolved_base = f"{UDS_BASE}/v1"
        else:
            assert base_url is not None
            resolved_base = normalize_base_url(base_url)
            self._http_client = httpx.AsyncClient(base_url=resolved_base, timeout=timeout)
        return resolved_base

    def _spawn(
        self,
        model: str,
        binary: Optional[str],
        host: Optional[str],
        port: Optional[int],
        socket: Optional[str],
        api_key: Optional[str],
        startup_timeout: float,
        timeout: httpx.Timeout,
        server_kwargs: Dict[str, Any],
    ) -> str:
        """Spawn and wait for a managed server, build its async http client, return the base URL."""
        self._server = ManagedServer(
            model,
            binary=binary,
            host=host,
            port=port,
            socket_path=socket,
            api_key=api_key,
            startup_timeout=startup_timeout,
            **server_kwargs,
        )
        self._server.start()
        resolved_base = self._server.base_url
        if self._server.uds_path is not None:
            uds_transport = httpx.AsyncHTTPTransport(uds=self._server.uds_path)
            self._http_client = httpx.AsyncClient(
                transport=uds_transport, base_url=UDS_BASE, timeout=timeout
            )
        else:
            self._http_client = httpx.AsyncClient(base_url=resolved_base, timeout=timeout)
        return resolved_base

    async def _resolve_model(self) -> str:
        if self._model is not None:
            return self._model
        if self._model_override is not None:
            self._model = self._model_override
            return self._model
        data = await self._client.models.list()
        items = list(data.data)
        if not items:
            raise MlxcelError("Server reported no models from /v1/models.")
        self._model = items[0].id
        return self._model

    # -- properties --------------------------------------------------------

    @property
    def model(self) -> str:
        """The resolved model id.

        Raises:
            MlxcelError: if accessed before any request has resolved it. Await a
                generation call (or :meth:`models`) first, or pass an explicit
                ``model=`` override.
        """
        if self._model is None:
            raise MlxcelError("Model id not resolved yet. Await a request first, or pass model=.")
        return self._model

    @property
    def openai_client(self) -> AsyncOpenAI:
        """The configured ``openai.AsyncOpenAI`` instance."""
        return self._client

    # -- generation --------------------------------------------------------

    async def generate(self, prompt: str, **sampling: Any) -> str:
        """Generate a completion for ``prompt`` and return the text."""
        self._ensure_alive()
        model = await self._resolve_model()
        params = build_params(sampling, chat=False)
        resp = await self._client.completions.create(model=model, prompt=prompt, **params)
        return resp.choices[0].text or ""

    async def stream(self, prompt: str, **sampling: Any) -> AsyncIterator[str]:
        """Stream completion text deltas for ``prompt``."""
        self._ensure_alive()
        model = await self._resolve_model()
        params = build_params(sampling, chat=False)
        stream = cast(
            "AsyncStream[Completion]",
            await self._client.completions.create(
                model=model, prompt=prompt, stream=True, **params
            ),
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text

    async def chat(self, messages: ChatMessages, **sampling: Any) -> str:
        """Run a chat completion and return the assistant message content."""
        self._ensure_alive()
        model = await self._resolve_model()
        params = build_params(sampling, chat=True)
        resp = await self._client.chat.completions.create(
            model=model, messages=as_messages(messages), **params
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(self, messages: ChatMessages, **sampling: Any) -> AsyncIterator[str]:
        """Stream chat completion content deltas."""
        self._ensure_alive()
        model = await self._resolve_model()
        params = build_params(sampling, chat=True)
        stream = cast(
            "AsyncStream[ChatCompletionChunk]",
            await self._client.chat.completions.create(
                model=model, messages=as_messages(messages), stream=True, **params
            ),
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # -- model / tokenizer helpers ----------------------------------------

    async def models(self) -> List[str]:
        """List model ids advertised by the server."""
        self._ensure_alive()
        data = await self._client.models.list()
        return [m.id for m in data.data]

    async def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        """Tokenize ``text`` via the server's native ``/tokenize`` route."""
        self._ensure_alive()
        resp = await self._raw_post("/tokenize", {"content": text, "add_special": add_special})
        return list(resp["tokens"])

    async def detokenize(self, tokens: Iterable[int]) -> str:
        """Decode token ids back to text via the native ``/detokenize`` route."""
        self._ensure_alive()
        resp = await self._raw_post("/detokenize", {"tokens": list(tokens)})
        return str(resp["content"])

    async def _raw_post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        assert self._http_client is not None
        url = native_base_url(self._base_url) + path
        # The OpenAI SDK injects Authorization on /v1/* routes, but these native
        # routes go through the bare httpx client, so add the bearer token here
        # when a key was supplied. With no key, omit the header to keep the
        # no-auth path working against servers started without --api-key.
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        response = await self._http_client.post(url, json=json, headers=headers)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        return result

    # -- lifecycle ---------------------------------------------------------

    def _ensure_alive(self) -> None:
        if self._server is not None:
            self._server.ensure_alive()

    async def close(self) -> None:
        """Close the async HTTP client and stop the managed server, if any."""
        if self._closed:
            return
        self._closed = True
        if self._http_client is not None:
            await self._http_client.aclose()
        if self._server is not None:
            self._server.close()

    async def __aenter__(self) -> "AsyncLLM":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __del__(self) -> None:
        # Best-effort synchronous cleanup at GC time. The async connection pool
        # is released by dropping the reference; the managed subprocess is reaped
        # via ManagedServer.close() which is synchronous. Do NOT attempt to run
        # the event loop or await close() here: the loop may already be gone at
        # interpreter shutdown. The correct API remains `await close()` or
        # `async with`.
        if self._closed:
            return
        try:
            if self._server is not None:
                self._server.close()
            self._http_client = None
        except Exception:
            pass


__all__ = ["AsyncLLM"]
