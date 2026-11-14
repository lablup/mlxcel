"""Streaming response ownership through the public sync/async generators."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterator, List

import httpx
import pytest

import mlxcel


def _event(chat: bool, text: str) -> bytes:
    choice: Any = {"index": 0, "delta": {"content": text}} if chat else {"index": 0, "text": text}
    return ("data: " + json.dumps({"choices": [choice]}) + "\n\n").encode()


class _SyncBody(httpx.SyncByteStream):
    def __init__(self, chat: bool) -> None:
        self.chunks = iter([_event(chat, "first"), _event(chat, "second"), b"data: [DONE]\n\n"])
        self.closed = False
        self.reads = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if chunk != b"data: [DONE]\n\n":
                self.reads += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, chat: bool) -> None:
        self.chunks = iter([_event(chat, "first"), _event(chat, "second"), b"data: [DONE]\n\n"])
        self.closed = False
        self.reads = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if chunk != b"data: [DONE]\n\n":
                self.reads += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _transport(body: Any, responses: List[httpx.Response]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "mock", "object": "model", "created": 1, "owned_by": "test"}]
                },
            )
        response = httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
        responses.append(response)
        return response

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("termination", ["close", "throw", "exhaust"])
def test_sync_stream_releases_response(chat: bool, termination: str) -> None:
    body = _SyncBody(chat)
    responses: List[httpx.Response] = []
    with mlxcel.LLM(transport=_transport(body, responses)) as client:
        stream: Any = (
            client.chat_stream([{"role": "user", "content": "hi"}]) if chat else client.stream("hi")
        )
        try:
            assert next(stream) == "first"
            assert not body.closed
            if termination == "close":
                stream.close()
            elif termination == "throw":
                with pytest.raises(ValueError, match="consumer stopped"):
                    stream.throw(ValueError("consumer stopped"))
            else:
                assert list(stream) == ["second"]
            # Check immediately, before closing the client or running GC.
            assert responses[0].is_closed
            assert body.closed
            assert body.reads == (2 if termination == "exhaust" else 1)
        finally:
            stream.close()
            for response in responses:
                response.close()


@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("termination", ["close", "throw", "exhaust"])
def test_async_stream_releases_response(chat: bool, termination: str) -> None:
    async def go() -> None:
        body = _AsyncBody(chat)
        responses: List[httpx.Response] = []
        async with mlxcel.AsyncLLM(transport=_transport(body, responses)) as client:
            stream: Any = (
                client.chat_stream([{"role": "user", "content": "hi"}])
                if chat
                else client.stream("hi")
            )
            try:
                assert await stream.__anext__() == "first"
                assert not body.closed
                if termination == "close":
                    await stream.aclose()
                elif termination == "throw":
                    with pytest.raises(ValueError, match="consumer stopped"):
                        await stream.athrow(ValueError("consumer stopped"))
                else:
                    assert [text async for text in stream] == ["second"]
                assert responses[0].is_closed
                assert body.closed
                assert body.reads == (2 if termination == "exhaust" else 1)
            finally:
                await stream.aclose()
                for response in responses:
                    await response.aclose()

    asyncio.run(go())
