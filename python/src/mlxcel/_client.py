"""Synchronous client (:class:`LLM`) for an mlxcel server.

``LLM`` wraps the OpenAI SDK with an httpx transport that works over either TCP
or a Unix domain socket. In *managed mode* it spawns and supervises a local
``mlxcel serve`` process; in *connect mode* it talks to an already-running
server. After the server is ready the resolved model id is discovered once from
``/v1/models`` and cached so callers never pass a model string. The raw OpenAI
client is exposed via ``openai_client`` for the full API surface (tools,
``response_format``, logprobs, multimodal). The async twin lives in
:mod:`._async_client`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, cast

import httpx
from openai import OpenAI

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
    from openai import Stream
    from openai.types.chat import ChatCompletionChunk
    from openai.types.completion import Completion


class LLM:
    """Synchronous mlxcel client.

    Examples:
        Managed mode (spawns a local server)::

            with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
                print(llm.generate("def fib(n):", max_tokens=128))

        Connect mode (existing server)::

            llm = mlxcel.LLM(base_url="http://localhost:8080/v1")
            llm = mlxcel.LLM(socket="/tmp/mlxcel.sock")
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
        transport: Optional[httpx.BaseTransport] = None,
        **server_kwargs: Any,
    ) -> None:
        """Create a client in managed or connect mode.

        Passing ``model`` selects managed mode: a local ``mlxcel serve`` process is
        spawned and supervised, and the constructor blocks until it reports ready.
        Passing ``base_url``, ``socket`` without a model, or ``transport`` selects
        connect mode: no subprocess, the client talks to an already-running server.
        Combining a model with ``base_url`` or ``transport`` raises
        :class:`~mlxcel.errors.MlxcelError`.

        Args:
            model: Model to serve in managed mode (passed to ``mlxcel serve -m``): a
                HuggingFace repo id or a local checkpoint path.
            base_url: Base URL of a running server for connect mode, e.g.
                ``"http://localhost:8080/v1"`` (a missing ``/v1`` suffix is added).
            socket: Unix domain socket path. With ``model`` it is the bind path for the
                spawned server (default: a short unique path under ``/tmp``); without
                one it is the connect target.
            api_key: API key sent as the bearer token on every request. In managed mode
                it is also handed to the spawned server via the ``LLAMA_API_KEY``
                environment variable (never argv).
            binary: Path to (or name of) the ``mlxcel`` executable. Falls back to the
                ``MLXCEL_BIN`` environment variable, then ``mlxcel`` on ``PATH``.
                Managed mode only.
            host: TCP host for the spawned server. Setting it (or ``port``) forces TCP
                instead of the default Unix socket. Managed mode only.
            port: TCP port for the spawned server; defaults to a free ephemeral port
                when TCP is in use. Managed mode only.
            timeout: httpx timeout for requests. Defaults to
                :data:`~mlxcel._common.DEFAULT_TIMEOUT` (generous 600-second read so a
                slow first token does not abort a long generation).
            startup_timeout: Seconds to wait for the spawned server's ``/health`` to
                report ready before raising
                :class:`~mlxcel.errors.MlxcelTimeoutError`. Managed mode only.
            transport: Injected httpx transport (e.g. ``httpx.MockTransport`` in
                tests); implies connect mode.
            **server_kwargs: Extra options for the spawned ``mlxcel serve`` process:
                ``ctx_size``, ``n_predict``, ``alias``, ``warmup`` (map to the
                matching CLI flags), ``extra_args`` (a list of raw CLI arguments
                appended verbatim), and ``shutdown_grace``. Managed mode only.
        """
        managed = is_managed(model, base_url, socket, transport)

        self._server: Optional[ManagedServer] = None
        self._http_client: Optional[httpx.Client] = None
        self._model: Optional[str] = None
        self._closed = False
        # Resolved API key, used to authorize native (/tokenize, /detokenize)
        # routes that bypass the OpenAI SDK's own Authorization injection. The
        # empty-or-None case stays unauthenticated to preserve the no-auth path.
        self._api_key = api_key
        timeout = timeout or DEFAULT_TIMEOUT

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

            self._client = OpenAI(
                base_url=resolved_base,
                api_key=api_key or "-",
                http_client=self._http_client,
            )
            self._base_url = resolved_base
            self._model = model_override or self._discover_model()
        except BaseException:
            self.close()
            raise

    # -- setup -------------------------------------------------------------

    def _connect(
        self,
        base_url: Optional[str],
        socket: Optional[str],
        transport: Optional[httpx.BaseTransport],
        timeout: httpx.Timeout,
    ) -> str:
        """Build the connect-mode http client (no subprocess) and return the base URL."""
        if transport is not None:
            # Injected transport (e.g. httpx.MockTransport in tests).
            resolved_base = connect_base_url(base_url)
            self._http_client = httpx.Client(
                transport=transport, base_url=resolved_base, timeout=timeout
            )
        elif socket is not None:
            uds_transport = httpx.HTTPTransport(uds=socket)
            self._http_client = httpx.Client(
                transport=uds_transport, base_url=UDS_BASE, timeout=timeout
            )
            resolved_base = f"{UDS_BASE}/v1"
        else:
            assert base_url is not None
            resolved_base = normalize_base_url(base_url)
            self._http_client = httpx.Client(base_url=resolved_base, timeout=timeout)
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
        """Spawn and wait for a managed server, build its http client, return the base URL."""
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
            uds_transport = httpx.HTTPTransport(uds=self._server.uds_path)
            self._http_client = httpx.Client(
                transport=uds_transport, base_url=UDS_BASE, timeout=timeout
            )
        else:
            self._http_client = httpx.Client(base_url=resolved_base, timeout=timeout)
        return resolved_base

    def _discover_model(self) -> str:
        data = self._client.models.list()
        items = list(data.data)
        if not items:
            raise MlxcelError("Server reported no models from /v1/models.")
        return items[0].id

    # -- properties --------------------------------------------------------

    @property
    def model(self) -> str:
        """The resolved model id, auto-discovered from ``/v1/models``."""
        assert self._model is not None
        return self._model

    @property
    def openai_client(self) -> OpenAI:
        """The configured ``openai.OpenAI`` instance for the full API surface."""
        return self._client

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, **sampling: Any) -> str:
        """Generate a completion for ``prompt`` and return the text."""
        self._ensure_alive()
        params = build_params(sampling, chat=False)
        resp = self._client.completions.create(model=self.model, prompt=prompt, **params)
        return resp.choices[0].text or ""

    def stream(self, prompt: str, **sampling: Any) -> Iterator[str]:
        """Stream completion text deltas for ``prompt``."""
        self._ensure_alive()
        params = build_params(sampling, chat=False)
        stream = cast(
            "Stream[Completion]",
            self._client.completions.create(model=self.model, prompt=prompt, stream=True, **params),
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text

    def chat(self, messages: ChatMessages, **sampling: Any) -> str:
        """Run a chat completion and return the assistant message content."""
        self._ensure_alive()
        params = build_params(sampling, chat=True)
        resp = self._client.chat.completions.create(
            model=self.model, messages=as_messages(messages), **params
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: ChatMessages, **sampling: Any) -> Iterator[str]:
        """Stream chat completion content deltas."""
        self._ensure_alive()
        params = build_params(sampling, chat=True)
        stream = cast(
            "Stream[ChatCompletionChunk]",
            self._client.chat.completions.create(
                model=self.model, messages=as_messages(messages), stream=True, **params
            ),
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # -- model / tokenizer helpers ----------------------------------------

    def models(self) -> List[str]:
        """List model ids advertised by the server."""
        self._ensure_alive()
        return [m.id for m in self._client.models.list().data]

    def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        """Tokenize ``text`` via the server's native ``/tokenize`` route."""
        self._ensure_alive()
        resp = self._raw_post("/tokenize", {"content": text, "add_special": add_special})
        return list(resp["tokens"])

    def detokenize(self, tokens: Iterable[int]) -> str:
        """Decode token ids back to text via the native ``/detokenize`` route."""
        self._ensure_alive()
        resp = self._raw_post("/detokenize", {"tokens": list(tokens)})
        return str(resp["content"])

    def _raw_post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        assert self._http_client is not None
        url = native_base_url(self._base_url) + path
        # The OpenAI SDK injects Authorization on /v1/* routes, but these native
        # routes go through the bare httpx client, so add the bearer token here
        # when a key was supplied. With no key, omit the header to keep the
        # no-auth path working against servers started without --api-key.
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        response = self._http_client.post(url, json=json, headers=headers)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        return result

    # -- lifecycle ---------------------------------------------------------

    def _ensure_alive(self) -> None:
        if self._server is not None:
            self._server.ensure_alive()

    def close(self) -> None:
        """Close the HTTP client and stop the managed server, if any."""
        if self._closed:
            return
        self._closed = True
        if self._http_client is not None:
            self._http_client.close()
        if self._server is not None:
            self._server.close()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["LLM"]
