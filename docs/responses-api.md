# OpenAI Responses API subset (`/v1/responses`)

`mlxcel serve` and `mlxcel-server` expose a Phase-1 subset of OpenAI's
Responses API. Basic `client.responses.create(...)` and streaming flows can be
used with the OpenAI Python SDK when `base_url` points at the mlxcel server, but
this is not a full implementation of every OpenAI Responses feature.

Implementation source map:

| Module | Responsibility |
|--------|----------------|
| `src/server/types/responses_request.rs` | Request types. |
| `src/server/types/responses_response.rs` | Response types. |
| `src/server/types/responses_stream.rs` | SSE event enum. |
| `src/server/responses_translator.rs` | Responses ↔ chat-completions translation. |
| `src/server/responses_store.rs` | In-memory response store. |
| `src/server/conversation_store.rs` | In-memory conversation transcript store. |
| `src/server/routes/responses.rs` | Route handlers. |

## Implemented endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/responses` | Create a response, either non-streaming or streaming. |
| GET | `/v1/responses/{id}` | Retrieve a stored response. |
| DELETE | `/v1/responses/{id}` | Delete a stored response. |
| POST | `/v1/responses/{id}/cancel` | Best-effort cancellation / cancellation marking. |

Aliases without `/v1` are also mounted for the same implemented routes:
`/responses`, `/responses/{id}`, and `/responses/{id}/cancel`.

The following OpenAI-style surfaces are **not mounted** in this implementation:

- `GET /v1/responses/{id}/input_items`
- `POST /v1/responses/compact`
- `POST /v1/responses/input_tokens`

## Quickstart

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")

resp = client.responses.create(
    model="qwen3-0.6b-4bit",
    input="Reply with: hello",
    max_output_tokens=64,
)
print(resp.status)
print(resp.output_text)

with client.responses.stream(
    model="qwen3-0.6b-4bit",
    input="Count to 5.",
    max_output_tokens=64,
) as stream:
    for event in stream:
        print(event.type, getattr(event, "delta", ""))
    final = stream.get_final_response()
    print(final.usage)
```

## Supported request fields

| Field | Status | Notes |
|-------|--------|-------|
| `model` | required | Must match the loaded model alias/path accepted by the server. |
| `input` | supported | String or typed input item array. |
| `instructions` | supported | Prepended as a system-style message; not inherited through `previous_response_id`. |
| `tools` | function-only | Only `{"type":"function", ...}` is accepted. |
| `tool_choice` | supported subset | String or named function choice compatible with chat-completions tooling. |
| `parallel_tool_calls` | accepted | Forwarded to existing tool-call handling. |
| `text.format` | supported subset | `text` and `json_schema` shapes are handled through existing structured-output code. |
| `reasoning` | echoed/advisory | Recorded and echoed; model-specific thinking behavior remains template/runtime dependent. |
| `conversation` | supported | String id or `{ "id": "..." }`; uses in-memory conversation store. |
| `previous_response_id` | supported | Rehydrates stored prior input/output items. Mutually exclusive with `conversation`. |
| `store` | supported | Defaults to `true`; `false` skips persistence. |
| `stream` | supported | Streams typed SSE events. |
| `max_output_tokens` | supported | Must be greater than zero. |
| `max_tool_calls` | supported | Soft cap on emitted function-call items. |
| `temperature`, `top_p`, `top_logprobs` | supported subset | Mapped to chat-completions sampling fields. |
| `metadata` | supported | Maximum 16 entries. |
| `prompt_cache_key` | accepted | Forwarded to prompt-cache plumbing. |
| `user`, `safety_identifier` | accepted | `user` is used when both are present; `safety_identifier` is used as a fallback. |
| `background` | rejected when `true` | Async polling is not implemented. |
| `truncation` | only `disabled` | Other values, including `auto`, return 400. |
| `service_tier` | accepted | Echoed/ignored; no scheduling tier is implemented. |

## Input items

Phase 1 supports these typed items:

```jsonc
[
  {"type":"message", "role":"user", "content":"hello"},
  {"type":"message", "role":"system", "content":[{"type":"text", "text":"sys"}]},
  {"type":"function_call", "call_id":"call_abc", "name":"f", "arguments":"{}"},
  {"type":"function_call_output", "call_id":"call_abc", "output":"ok"},
  {"type":"reasoning", "content":[{"type":"reasoning_text", "text":"..."}]}
]
```

`developer` role is treated like `system`. Reasoning input items are accepted and forwarded: the text content is buffered and attached to the parallel `reasoning` field of the following assistant turn. Chat templates that render `message.get('reasoning')` (such as Gemma 4) receive it there. The `preserve_thinking` kwarg controls whether the field survives the rolling-checkpoint strip: `false` (the default, unless the prompt cache is on) drops prior-turn reasoning along with any inline `<think>` blocks; `true` retains it.

Message content parts reuse mlxcel's chat-completions content part types. This
means `text`, `image_url`, `video_url`, and `input_audio` can deserialize, but
actual execution still depends on the loaded model's media support. OpenAI
Responses-specific `input_image` / `input_file` compatibility is not complete.

A request with no effective input is rejected with 400 before any model
dispatch: an empty `input` array, a blank/whitespace-only string or `text`
part, and no image/video/audio, tool call, or reasoning content anywhere in
the translated conversation (including history pulled in through
`previous_response_id` / `conversation`) all trigger the same
`invalid_request_error`. A blank `input` combined with non-empty
`instructions` still passes, since `instructions` becomes a real system
message in the rendered conversation. The same check applies to
`/v1/chat/completions`. `/v1/completions` (the raw-prompt legacy endpoint) is
a deliberate exception: it rejects a whitespace-only prompt with the same 400
`invalid_request_error`, but allows a fully empty prompt through, since
unconditional generation from BOS is a legitimate base-model use case on a
route that has no chat-template scaffolding to be empty around (issue #806).

## Response shape

Responses use an OpenAI-like object shape:

```jsonc
{
  "id": "resp_...",
  "object": "response",
  "created_at": 1234.0,
  "completed_at": 1235.0,
  "status": "completed",
  "model": "...",
  "output": [
    {"type":"reasoning", "id":"rs_...", "status":"completed", "content":[...]},
    {"type":"function_call", "id":"fc_...", "call_id":"call_...", "name":"...", "arguments":"{}", "status":"completed"},
    {"type":"message", "id":"msg_...", "role":"assistant", "status":"completed", "content":[...]}
  ],
  "output_text": "...",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 34,
    "total_tokens": 46,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

Several request fields are echoed back when present. Treat this as compatibility
surface, not as proof that every echoed field changes runtime behavior.

## Streaming events

SSE frames are typed and include a monotonic `sequence_number` per response.
Phase 1 emits events such as:

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.reasoning_text.delta`
- `response.reasoning_text.done`
- `response.completed`
- failure/incomplete/error events on error paths

## Response and conversation stores

The stores are in memory and are bounded by entry count and TTL.

| Flag | Default | Env var | Notes |
|------|---------|---------|-------|
| `--responses-store-max-entries` | `1024` | `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES` | `0` disables response persistence. |
| `--responses-store-ttl-secs` | `3600` | `LLAMA_ARG_RESPONSES_STORE_TTL_SECS` | `0` disables TTL. |
| `--conversation-store-max-entries` | `256` | `LLAMA_ARG_CONVERSATION_STORE_MAX_ENTRIES` | `0` disables conversations. |
| `--conversation-store-ttl-secs` | `3600` | `LLAMA_ARG_CONVERSATION_STORE_TTL_SECS` | `0` disables TTL. |

When response storage is disabled, retrieve/delete/cancel-by-id and
`previous_response_id` chaining return an error. When conversation storage is
disabled, requests using `conversation` return an error.

## Chaining semantics

- `previous_response_id` loads the stored response's input items and output
  items as prior conversation history, then appends the new input.
- `conversation` loads and appends to an in-memory transcript by id.
- The two fields are mutually exclusive.
- `instructions` from the referenced prior response are not carried over.

## Unsupported tool types

Only function tools are accepted. Built-in/external tool types such as
`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`,
`image_generation`, `mcp`, `custom`, `apply_patch`, and `function_shell` return
400 responses. `mlxcel` does not execute external tools for the Responses API.

## Differences from OpenAI's full API

Notable gaps:

- no background job mode;
- no input-items pagination endpoint;
- no server-side compaction endpoint;
- no token-count endpoint;
- no built-in tools or MCP connector execution;
- no disk-persisted response store;
- incomplete OpenAI Responses multimodal part compatibility.
