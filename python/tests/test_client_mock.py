"""Unit tests for LLM / AsyncLLM using httpx.MockTransport (no real server)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

import mlxcel
from mlxcel._sampling import build_params

MODEL_ID = "mock-model"

# Captured request bodies, keyed by path, so tests can assert sampling mapping.
_LAST_BODY: Dict[str, Dict[str, Any]] = {}

# Captured request headers, keyed by path, so tests can assert auth behavior.
_LAST_HEADERS: Dict[str, Dict[str, str]] = {}


def _models_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": 1, "owned_by": "user"}],
        },
    )


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1,
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": "generated text", "finish_reason": "stop"}],
        },
    )


def _chat_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "chat reply"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _sse(chunks: List[Dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
    )


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    body: Dict[str, Any] = {}
    if request.content:
        try:
            body = json.loads(request.content)
        except json.JSONDecodeError:
            body = {}
    _LAST_BODY[path] = body
    _LAST_HEADERS[path] = dict(request.headers)
    stream = bool(body.get("stream"))

    if path == "/v1/models":
        return _models_response()
    if path == "/v1/completions":
        if stream:
            return _sse(
                [
                    {"choices": [{"index": 0, "text": "gen ", "finish_reason": None}]},
                    {"choices": [{"index": 0, "text": "text", "finish_reason": "stop"}]},
                ]
            )
        return _completion_response()
    if path == "/v1/chat/completions":
        if stream:
            return _sse(
                [
                    {"choices": [{"index": 0, "delta": {"content": "chat "}}]},
                    {
                        "choices": [
                            {"index": 0, "delta": {"content": "reply"}, "finish_reason": "stop"}
                        ]
                    },
                ]
            )
        return _chat_response()
    if path == "/tokenize":
        return httpx.Response(200, json={"tokens": [1, 2, 3]})
    if path == "/detokenize":
        return httpx.Response(200, json={"content": "decoded"})
    return httpx.Response(404, json={"error": "not found"})


def _error_handler(request: httpx.Request) -> httpx.Response:
    """Like ``_handler`` but returns HTTP 500 on chat completions."""
    if request.url.path == "/v1/models":
        return _models_response()
    if request.url.path == "/v1/chat/completions":
        return httpx.Response(500, json={"error": {"message": "boom"}})
    return httpx.Response(404, json={"error": "not found"})


@pytest.fixture()
def llm() -> Any:
    _LAST_BODY.clear()
    _LAST_HEADERS.clear()
    transport = httpx.MockTransport(_handler)
    client = mlxcel.LLM(transport=transport)
    yield client
    client.close()


def test_model_auto_discovery(llm: Any) -> None:
    assert llm.model == MODEL_ID


def test_generate(llm: Any) -> None:
    assert llm.generate("hello") == "generated text"


def test_generate_passes_sampling(llm: Any) -> None:
    llm.generate("hello", max_tokens=42, temperature=0.5)
    body = _LAST_BODY["/v1/completions"]
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.5
    assert body["model"] == MODEL_ID


def test_generate_extra_body_passthrough(llm: Any) -> None:
    llm.generate("hello", top_k=20, min_p=0.05, repetition_penalty=1.1)
    body = _LAST_BODY["/v1/completions"]
    # Server-specific knobs land in the request body (extra_body merged in).
    assert body["top_k"] == 20
    assert body["min_p"] == 0.05
    assert body["repetition_penalty"] == 1.1


def test_stream(llm: Any) -> None:
    assert "".join(llm.stream("hello")) == "gen text"


def test_chat(llm: Any) -> None:
    assert llm.chat([{"role": "user", "content": "hi"}]) == "chat reply"


def test_chat_stream(llm: Any) -> None:
    out = "".join(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert out == "chat reply"


def test_models(llm: Any) -> None:
    assert llm.models() == [MODEL_ID]


def test_tokenize(llm: Any) -> None:
    assert llm.tokenize("hi", add_special=True) == [1, 2, 3]
    assert _LAST_BODY["/tokenize"] == {"content": "hi", "add_special": True}


def test_detokenize(llm: Any) -> None:
    assert llm.detokenize([1, 2, 3]) == "decoded"
    assert _LAST_BODY["/detokenize"] == {"tokens": [1, 2, 3]}


def test_response_format_passthrough(llm: Any) -> None:
    schema = {"type": "json_object"}
    llm.chat([{"role": "user", "content": "hi"}], response_format=schema)
    body = _LAST_BODY["/v1/chat/completions"]
    assert body["response_format"] == schema


def test_response_format_on_generate_uses_extra_body(llm: Any) -> None:
    # response_format is chat-only for the OpenAI SDK; on the completions
    # endpoint it must be routed through extra_body (no TypeError) and still
    # reach the server inside the request body.
    schema = {"type": "json_object"}
    assert llm.generate("hi", response_format=schema) == "generated text"
    body = _LAST_BODY["/v1/completions"]
    assert body["response_format"] == schema


# -- native-route authorization (regression: missing Bearer on /tokenize) ----


def test_native_routes_carry_bearer_when_api_key_set() -> None:
    _LAST_BODY.clear()
    _LAST_HEADERS.clear()
    transport = httpx.MockTransport(_handler)
    client = mlxcel.LLM(transport=transport, api_key="secret")
    try:
        client.tokenize("hi")
        client.detokenize([1, 2, 3])
    finally:
        client.close()
    assert _LAST_HEADERS["/tokenize"].get("authorization") == "Bearer secret"
    assert _LAST_HEADERS["/detokenize"].get("authorization") == "Bearer secret"


def test_native_routes_omit_auth_without_api_key(llm: Any) -> None:
    # The default fixture client has no api_key; the no-auth path must stay
    # intact so servers started without --api-key keep working.
    llm.tokenize("hi")
    llm.detokenize([1, 2, 3])
    assert "authorization" not in _LAST_HEADERS["/tokenize"]
    assert "authorization" not in _LAST_HEADERS["/detokenize"]


def test_openai_client_escape_hatch(llm: Any) -> None:
    from openai import OpenAI

    assert isinstance(llm.openai_client, OpenAI)
    resp = llm.openai_client.completions.create(model=llm.model, prompt="x")
    assert resp.choices[0].text == "generated text"


def test_http_error_propagates_as_openai_exception() -> None:
    import openai

    transport = httpx.MockTransport(_error_handler)
    client = mlxcel.LLM(transport=transport)
    try:
        with pytest.raises(openai.APIStatusError):
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        client.close()


# -- mode-selection / validation --------------------------------------------


def test_both_model_and_connect_target_is_error() -> None:
    with pytest.raises(mlxcel.MlxcelError):
        mlxcel.LLM("some-model", base_url="http://localhost:8080/v1")


def test_no_args_is_error() -> None:
    with pytest.raises(mlxcel.MlxcelError):
        mlxcel.LLM()


def test_base_url_and_socket_is_error() -> None:
    with pytest.raises(mlxcel.MlxcelError):
        mlxcel.LLM(base_url="http://x/v1", socket="/tmp/x.sock")


# -- sampling unit ----------------------------------------------------------


def test_build_params_splits_extra_body() -> None:
    params = build_params({"max_tokens": 10, "top_k": 5, "unknown_knob": "v"})
    assert params["max_tokens"] == 10
    assert params["extra_body"]["top_k"] == 5
    assert params["extra_body"]["unknown_knob"] == "v"


def test_build_params_caller_extra_body_wins() -> None:
    params = build_params({"top_k": 5, "extra_body": {"top_k": 99}})
    assert params["extra_body"]["top_k"] == 99


def test_build_params_drops_none() -> None:
    params = build_params({"max_tokens": None, "temperature": 0.0})
    assert "max_tokens" not in params
    assert params["temperature"] == 0.0


def test_build_params_response_format_routing() -> None:
    schema = {"type": "json_object"}
    # Chat: top-level field.
    chat_params = build_params({"response_format": schema}, chat=True)
    assert chat_params["response_format"] == schema
    assert "extra_body" not in chat_params
    # Completions: routed through extra_body so completions.create won't raise.
    cmpl_params = build_params({"response_format": schema}, chat=False)
    assert "response_format" not in cmpl_params
    assert cmpl_params["extra_body"]["response_format"] == schema


# -- async ------------------------------------------------------------------


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def test_async_generate_and_discovery() -> None:
    _LAST_BODY.clear()
    transport = httpx.MockTransport(_handler)

    async def go() -> tuple[str, str, str]:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            text = await client.generate("hello", max_tokens=7)
            chat = await client.chat([{"role": "user", "content": "hi"}])
            return text, chat, client.model
        finally:
            await client.close()

    text, chat, model = _run(go())
    assert text == "generated text"
    assert chat == "chat reply"
    assert model == MODEL_ID
    assert _LAST_BODY["/v1/completions"]["max_tokens"] == 7


def test_async_stream_and_tokenize() -> None:
    _LAST_BODY.clear()
    transport = httpx.MockTransport(_handler)

    async def go() -> tuple[str, list[int], str]:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            deltas = [d async for d in client.stream("hi")]
            tokens = await client.tokenize("hi")
            decoded = await client.detokenize([1, 2, 3])
            return "".join(deltas), tokens, decoded
        finally:
            await client.close()

    streamed, tokens, decoded = _run(go())
    assert streamed == "gen text"
    assert tokens == [1, 2, 3]
    assert decoded == "decoded"


def test_async_native_routes_carry_bearer_when_api_key_set() -> None:
    _LAST_BODY.clear()
    _LAST_HEADERS.clear()
    transport = httpx.MockTransport(_handler)

    async def go() -> None:
        client = mlxcel.AsyncLLM(transport=transport, api_key="secret")
        try:
            await client.tokenize("hi")
            await client.detokenize([1, 2, 3])
        finally:
            await client.close()

    _run(go())
    assert _LAST_HEADERS["/tokenize"].get("authorization") == "Bearer secret"
    assert _LAST_HEADERS["/detokenize"].get("authorization") == "Bearer secret"


def test_async_native_routes_omit_auth_without_api_key() -> None:
    _LAST_BODY.clear()
    _LAST_HEADERS.clear()
    transport = httpx.MockTransport(_handler)

    async def go() -> None:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            await client.tokenize("hi")
            await client.detokenize([1, 2, 3])
        finally:
            await client.close()

    _run(go())
    assert "authorization" not in _LAST_HEADERS["/tokenize"]
    assert "authorization" not in _LAST_HEADERS["/detokenize"]


def test_async_chat_stream() -> None:
    transport = httpx.MockTransport(_handler)

    async def go() -> str:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            chunks = [d async for d in client.chat_stream([{"role": "user", "content": "hi"}])]
            return "".join(chunks)
        finally:
            await client.close()

    assert _run(go()) == "chat reply"


def test_async_models() -> None:
    transport = httpx.MockTransport(_handler)

    async def go() -> list[str]:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            return await client.models()
        finally:
            await client.close()

    assert _run(go()) == [MODEL_ID]


def test_async_openai_client_escape_hatch() -> None:
    from openai import AsyncOpenAI

    transport = httpx.MockTransport(_handler)

    async def go() -> bool:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            return isinstance(client.openai_client, AsyncOpenAI)
        finally:
            await client.close()

    assert _run(go())


def test_async_model_property_raises_before_resolution() -> None:
    transport = httpx.MockTransport(_handler)

    async def go() -> None:
        client = mlxcel.AsyncLLM(transport=transport)
        try:
            # model property raises before any request has resolved the id
            with pytest.raises(mlxcel.MlxcelError):
                _ = client.model
        finally:
            await client.close()

    _run(go())


def test_async_ambiguous_args_is_error() -> None:
    async def go() -> None:
        with pytest.raises(mlxcel.MlxcelError):
            mlxcel.AsyncLLM("some-model", base_url="http://localhost:8080/v1")

    _run(go())


def test_async_no_args_is_error() -> None:
    async def go() -> None:
        with pytest.raises(mlxcel.MlxcelError):
            mlxcel.AsyncLLM()

    _run(go())
