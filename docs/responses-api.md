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
| `src/server/routes/responses_run.rs` | One streamed generation, driven into an event sink. |
| `src/server/routes/responses_background.rs` | Background job admission and lifecycle. |
| `src/server/routes/responses_background_sink.rs` | Background event sink and terminal recording. |
| `src/server/responses_background_events.rs` | Replayable event log of a streamed background response. |
| `src/server/routes/responses_background_stream.rs` | SSE readers of that log: the streamed create and `GET ...?stream=true`. |
| `src/server/responses_compaction.rs` | Compaction capsule, replay, selection and summarization core. |
| `src/server/routes/responses_compact.rs` | `POST /v1/responses/compact` handler. |

## Implemented endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/responses` | Create a response: non-streaming, streaming, or as a background job (`background: true`, see [Background mode](#background-mode)). |
| GET | `/v1/responses/{id}` | Retrieve a stored response, or poll a background response. With `?stream=true` it replays and tails the events of a streamed background response; see [Streaming and resuming](#streaming-and-resuming). |
| DELETE | `/v1/responses/{id}` | Delete a stored response. A background response that has not finished returns `400`; cancel it first. |
| POST | `/v1/responses/{id}/cancel` | Best-effort cancellation of an in-flight streaming response. The stream ends with `response.completed` whose `response.status` is `cancelled`, and the stored copy has the same status. Cancelling a response already stored as `cancelled` is idempotent and returns the stored Response object. Other finished responses return `400 invalid_request_error` without changing the saved result. |
| POST | `/v1/responses/compact` | Summarize older turns into a `compaction` item; see [Conversation compaction](#conversation-compaction). |
| POST | `/v1/responses/input_tokens` | Count the prompt tokens a body would prefill; see `docs/llama-server-compat.md`. |

Aliases without `/v1` are also mounted for the same implemented routes:
`/responses`, `/responses/{id}`, `/responses/{id}/cancel`,
`/responses/compact`, and `/responses/input_tokens`.

The following OpenAI-style surface is **not mounted** in this implementation:

- `GET /v1/responses/{id}/input_items`

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
| `tool_choice` | supported | `auto`, `none`, `required`, or a named function. `required` and the named form are enforced, not just accepted; see [Tool choice enforcement](#tool-choice-enforcement). |
| `parallel_tool_calls` | accepted | Forwarded to existing tool-call handling. |
| `text.format` | supported subset | `text` and `json_schema` shapes are handled through existing structured-output code. |
| `reasoning` | supported subset | `reasoning.effort` is echoed unchanged and feeds the same template controls as chat-completions `reasoning_effort`: `none`, `off`, `disabled`, `false`, and `0` (case-insensitive after trimming) disable `enable_thinking`; other values enable thinking and are forwarded verbatim as `reasoning_effort`, or as `reasoning_strength` when that is the identifier the loaded template reads. The derived controls override server-wide template defaults per key. `summary` remains model/runtime dependent. |
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
| `background` | supported | `true` returns a `queued` response at once and runs the generation as a job; see [Background mode](#background-mode). With `stream: true` the events are streamed and can be resumed. |
| `truncation` | only `disabled` | Other values, including `auto`, return 400. |
| `service_tier` | accepted | Echoed/ignored; no scheduling tier is implemented. |
| `context_management` | supported subset | `[{"type":"compaction","compact_threshold":N}]` enables automatic compaction; see [Automatic compaction](#automatic-compaction). `[]` or `null` disables it. |

## Input items

Phase 1 supports these typed items:

```jsonc
[
  {"type":"message", "role":"user", "content":"hello"},
  {"type":"message", "role":"system", "content":[{"type":"input_text", "text":"sys"}]},
  {"type":"message", "role":"user", "content":[{"type":"input_image", "image_url":"data:image/png;base64,...", "detail":"high"}]},
  {"type":"function_call", "call_id":"call_abc", "name":"f", "arguments":"{}"},
  {"type":"function_call_output", "call_id":"call_abc", "output":"ok"},
  {"type":"function_call_output", "call_id":"call_img", "output":[{"type":"input_text", "text":"captured"}, {"type":"input_image", "image_url":"data:image/png;base64,..."}]},
  {"type":"reasoning", "content":[{"type":"reasoning_text", "text":"..."}]},
  {"type":"compaction", "id":"cmp_...", "encrypted_content":"..."}
]
```

A `compaction` item is the capsule `POST /v1/responses/compact` returns; see [Conversation compaction](#conversation-compaction) for how it replays.

`developer` role is treated like `system`. Reasoning input items are accepted and forwarded: the text content is buffered and attached to the parallel `reasoning` field of the following assistant turn. Chat templates that render `message.get('reasoning')` (such as Gemma 4) receive it there. The `preserve_thinking` kwarg controls whether the field survives the rolling-checkpoint strip: `false` (the default, unless the prompt cache is on) drops prior-turn reasoning along with any inline `<think>` blocks; `true` retains it.

Message items and array-valued `function_call_output.output` fields accept the Responses-native part spellings alongside mlxcel's existing chat-completions spellings:

| Part | Fields | Behavior |
|------|--------|----------|
| `input_text` | `text: string` | Mapped to a chat-completions `text` part. |
| `input_image` | `image_url: string`, optional `detail: "auto" \| "low" \| "high"` | Mapped to an `image_url` part; URLs and data URIs use the existing media pipeline. |
| `input_image` with only `file_id` | `file_id: string` | Rejected with 400 because this server has no uploads store; provide `image_url` instead. |
| `input_file` | any Responses file fields | Rejected with a named 400; PDF and document ingestion are not implemented. |
| `text`, `image_url`, `video_url`, `input_audio` | existing chat-completions shapes | Accepted unchanged; execution still depends on the loaded model's media support. |

A string-valued function output is unchanged. In an array-valued function output, text parts are joined with newlines in the tool message, non-text/non-image parts are retained as JSON text, and image parts are emitted in order as an immediately following user image turn. The tool message ends with `[Image output attached in the next message]` when that follow-up is present. Images on user message items remain in place; images on assistant, system, or developer message items are likewise moved to an immediately following user turn because supported chat templates render image placeholders on user turns.

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

## mlxcel extension fields

mlxcel adds a small number of non-OpenAI fields to its response bodies. Each is optional and omitted (`skip_serializing_if`) unless the loaded model produces it. The convention started with `reasoning_content` on Chat Completions (mirroring vLLM), which now also carries an identical `reasoning` alias by default for OpenRouter-style clients. The alias is limited to Chat Completions and can be disabled with `--reasoning-alias-field none` or `MLXCEL_REASONING_ALIAS_FIELD=none`; Responses API reasoning continues to use its native `response.reasoning_text.delta` events and is unchanged. The extension convention also covers:

- `choices[0].message.florence2_result` on non-streaming
  `POST /v1/chat/completions` (issue #1073): the structured form of a
  Florence-2 task answer, present only when the loaded model is Florence-2.
  `message.content` carries the same answer as the human-readable text the
  CLI prints; this field carries the parsed coordinates as JSON so a client
  does not have to re-parse the formatted string. The object is
  `{"task": "<OD>", "kind": "bboxes" | "text" | "quad_boxes" | "polygons" |
  "bboxes_or_polygons", ...}` with the coordinate arrays under upstream
  mlx-vlm / HuggingFace key names (`bboxes`, `quad_boxes`, `polygons`,
  `labels`, `bboxes_labels`, `polygons_labels`); coordinates are pixels in
  the original image extent. Streaming chat responses and `/v1/responses`
  return the rendered text only; the structured field is a non-streaming
  chat-completions surface.

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
- `response.completed` (a stream stopped by a cancel call or a client disconnect also ends with this event: `response.status` is `cancelled` and `completed_at` is null, so clients must check `status`)
- `mlx.compaction.started`, `mlx.compaction.completed` (MLX extension, only after an automatic compaction; see [Automatic compaction](#automatic-compaction))
- failure/incomplete/error events on error paths

Reasoning streams as `response.reasoning_text.delta` events inside a `reasoning` output item, ahead of the message item, and never as `output_text`. This holds when the chat template primes an open thinking block (for example a generation prompt ending in `<think>\n`), where the model writes only the close marker: the stream starts inside the reasoning item, as the non-streaming split does.

Harmony (GPT-OSS) replies surface their `analysis` channel as the same `reasoning` output item, with or without tool calls, and `usage.output_tokens_details.reasoning_tokens` is the token count of that reasoning text, identical for streaming and non-streaming requests. Multiple analysis or `<think>` segments are accumulated through the same streaming filter in both response modes, without parser-added separators or dropped later segments. It is 0 when the reply has no reasoning.

Function calls stream while they are generated for the formats `/v1/chat/completions` streams incrementally (Hermes / Qwen JSON, Mistral `[ARGS]` and JSON list, Llama 3 JSON, Qwen3-Coder XML; see `docs/server-features.md`). Once a call's name is complete, any open reasoning or message item is closed and a `function_call` item is added with `status: "in_progress"` and empty `arguments`; `response.function_call_arguments.delta` events then carry the argument fragments. After generation, `response.function_call_arguments.done` and `response.output_item.done` carry the authoritative parsed arguments, and the call keeps its streamed `call_id` in the completed response. For a streamed request that parses tool calls, `response.completed` (and the copy stored for `previous_response_id`) is built from the items the stream sent: same item ids and `output_index` order, each message segment as its own `message` item (prose before and after a call are two items), function calls with their streamed `fc_` id, `call_id` and parsed arguments, and `output_text` as the concatenation of the message segments. If the streamer panics on a fragment, it is contained and the remaining arguments arrive from the final parse. Non-streaming responses are unchanged. For Hermes, Granite, Mistral and Llama 3, object `arguments` are the model's raw JSON text, not a re-serialization. A streamed call that the final parse does not confirm is logged at `warn` with the response id and closed as `incomplete`; the parsed calls are then emitted as separate items. Other formats emit each call as one `output_item.added`, one argument delta, `.done`, and `output_item.done` after generation, as before.

## Background mode

`POST /v1/responses` with `"background": true` returns HTTP 200 at once with a `queued` response and runs the generation as a job on the server. Poll `GET /v1/responses/{id}` until `status` is terminal.

```bash
curl -s http://127.0.0.1:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"local","input":"Write a haiku about rain.","background":true}'
# {"id":"resp_...","status":"queued","background":true,"output":[],...}
curl -s http://127.0.0.1:8080/v1/responses/resp_...
# status: queued -> in_progress -> completed
```

| Status | Meaning |
|--------|---------|
| `queued` | Admitted; generation has not started. Media decoding, automatic compaction and the wait for a generation slot happen here. |
| `in_progress` | The model started prefilling or generating. |
| `completed`, `incomplete` | Finished, with the same `output`, `usage` and `incomplete_details` a foreground response carries. |
| `failed` | `error` holds `{code, message}`. `code` is `server_error`, `server_overloaded` (the queue was full when the job tried to start), or `invalid_prompt` (a media read, the prompt render or automatic compaction refused the request). |
| `cancelled` | `POST /v1/responses/{id}/cancel` stopped the job. |

Every object of a background response carries `"background": true`; a foreground response has no `background` key. Until the job is terminal, `output` is `[]` and `usage` is all zeros: polling does not expose partial output.

What create answers:

- `400 invalid_request_error` for everything a foreground request is refused for before generation (malformed fields, unsupported media, invalid sampling or tool parameters, an invalid schema, an unknown `previous_response_id`), and also when the response store is disabled, when `stream` is `true` and `--responses-background-event-log-bytes` is `0`, or when `--responses-background-max-jobs`, the response store entry limit, or `--responses-store-max-bytes` is `0`. No job is created.
- `429` with `error.code: "rate_limit_exceeded"` when `--responses-background-max-jobs` jobs are already active, or when the active jobs fill `--responses-store-max-bytes`. Retry after a job finishes.
- `503` when the generation queue is full, as for a foreground request.

Interactions:

- Dropping the create connection or a poll cancels nothing. Only the cancel endpoint stops a job.
- `store: false` is allowed. The final object is then readable by polling for `--responses-background-ephemeral-ttl-secs` and cannot be used as `previous_response_id`.
- `previous_response_id` that names a job that has not finished returns `400` with `previous response '<id>' has not finished (status: <status>)`. Once the job is terminal it chains like any stored response.
- `conversation` history is read when the job is created, and the turn is appended when the job finishes with output. A `failed` job appends nothing.
- `context_management` runs inside the job, so a compaction failure ends the job as `failed` instead of failing the create call.
- Media inputs are allowed. Whether the model supports a media kind is checked at create; reading and decoding happen inside the job.

### Streaming and resuming

`"background": true` with `"stream": true` answers with an SSE stream of the job's events, and the same events stay readable from the server afterwards. Every event carries a `sequence_number` that starts at 0 and increases by one, so a client that lost its connection resumes after the last one it received:

```bash
curl -N http://127.0.0.1:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"local","input":"Write a haiku about rain.","background":true,"stream":true}'
# event: response.created      (sequence_number 0, status queued; carries the id)
# event: response.queued       (sequence_number 1)
# event: response.in_progress  (when the model starts)
# ... the same item and delta events a foreground stream sends ...
# event: response.completed
curl -N 'http://127.0.0.1:8080/v1/responses/resp_...?stream=true&starting_after=7'
# events 8, 9, ... then the live ones, closing after the terminal event
```

With the OpenAI Python SDK this is `client.responses.create(..., background=True, stream=True)` and `client.responses.retrieve(id, stream=True, starting_after=n)`.

- The stream ends with exactly one terminal event carrying the stored object: `response.completed`, `response.incomplete`, or `response.failed`. A cancelled job ends with `response.completed` whose `response.status` is `cancelled`. The foreground `response.error` event is not used for a job failure.
- After an automatic compaction, `response.in_progress` is followed by the two `mlx.compaction.*` events and the compaction item, as in the foreground.
- Any number of readers can stream the same response, each with its own cursor. Closing a stream ends that reader only; the job keeps running, and only the cancel endpoint stops it. Idle streams get the SSE comment ping (`--sse-ping-interval`).
- `X-Conversation-Id` is ignored on a background create: `starting_after` is the resume cursor.
- `GET /v1/responses/{id}?stream=true` without `starting_after` replays from the first event. `stream=false` or no `stream` returns the JSON object as before.
- The server keeps about `--responses-background-event-log-bytes` of events per response (default 4 MiB) and drops the oldest when the log is full; the terminal event is never dropped. A resume that asks for dropped events (a `starting_after` below the oldest retained `sequence_number` minus one, or no `starting_after` once the first event is gone) returns `400 invalid_request_error` naming the oldest retained `sequence_number`; retrieve the response with a plain GET instead. A reader that falls behind the retained window while streaming receives one `response.error` event with `code: "stream_events_evicted"` and the same message, and its stream ends.
- Other `400` answers: a `starting_after` that is not a non-negative integer, a `stream` that is not a boolean, and a response that has no event log because it was not created with both `background: true` and `stream: true`. An unknown id returns `404`.
- The log lives as long as the response: it is counted in `--responses-store-max-bytes` and is freed when the response is evicted, deleted or expires (after `--responses-background-ephemeral-ttl-secs` for `store: false`). A log too large to fit the store budget beside its response is dropped when the job finishes; streams already open still end normally.

Jobs are held in memory. The single-model server does not drain them on exit: stopping or restarting the process loses queued and running jobs, and their ids return `404` afterwards. If the model worker exits while the process keeps running, its jobs end as `failed`.

## Response and conversation stores

The stores are in memory and are bounded by entry count, approximate retained
JSON bytes, and TTL.

| Flag | Default | Env var | Notes |
|------|---------|---------|-------|
| `--responses-store-max-entries` | `1024` | `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES` | `0` disables response persistence. |
| `--responses-store-max-bytes` | `268435456` | `MLXCEL_RESPONSES_STORE_MAX_BYTES` | Approximate retained-byte budget; `0` keeps the route enabled but immediately evicts stored responses. |
| `--responses-store-ttl-secs` | `3600` | `LLAMA_ARG_RESPONSES_STORE_TTL_SECS` | `0` disables TTL. |
| `--responses-background-max-jobs` | `64` | `MLXCEL_RESPONSES_BACKGROUND_MAX_JOBS` | Cap on concurrently active `background: true` responses; each holds one generation thread. `0` disables background mode. |
| `--responses-background-ephemeral-ttl-secs` | `600` | `MLXCEL_RESPONSES_BACKGROUND_EPHEMERAL_TTL_SECS` | How long a finished `background: true` response created with `store: false` stays readable by polling. Must be at least `1`; `0` fails startup. |
| `--responses-background-event-log-bytes` | `4194304` | `MLXCEL_RESPONSES_BACKGROUND_EVENT_LOG_BYTES` | Byte budget of the replayable event log of one `background: true` response created with `stream: true`; the oldest events are dropped when it is full. `0` refuses `stream: true` on background responses. |
| `--conversation-store-max-entries` | `256` | `LLAMA_ARG_CONVERSATION_STORE_MAX_ENTRIES` | `0` disables conversations. |
| `--conversation-store-max-bytes` | `67108864` | `MLXCEL_CONVERSATION_STORE_MAX_BYTES` | Approximate retained-byte budget; `0` keeps the route enabled but immediately evicts transcripts. |
| `--conversation-store-ttl-secs` | `3600` | `LLAMA_ARG_CONVERSATION_STORE_TTL_SECS` | `0` disables TTL. |

Active background jobs are tracked outside the TTL and LRU limits, so a running job is never evicted. Their approximate bytes count against `--responses-store-max-bytes` at admission (a job that would push the active total past the budget is refused) but never evict stored responses. When a job finishes, its final object becomes a stored response. If that object alone is larger than `--responses-store-max-bytes`, a `failed` response with `error.code: "server_error"` and empty output is kept instead, so a poll still gets an answer. The final object of a `store: false` job is kept only for `--responses-background-ephemeral-ttl-secs` and is never usable as `previous_response_id`. Background jobs are in memory only and a restart loses them. Background mode is disabled when the job limit, the entry limit, or the byte budget is `0`.

When response storage is disabled, retrieve/delete/cancel-by-id and
`previous_response_id` chaining return an error. When conversation storage is
disabled, requests using `conversation` return an error.

## Chaining semantics

- `previous_response_id` loads the stored response's input items and output
  items as prior conversation history, then appends the new input.
- `conversation` loads and appends to an in-memory transcript by id.
- The two fields are mutually exclusive.
- `instructions` from the referenced prior response are not carried over.
- A `compaction` item anywhere in the combined history (stored input, stored
  output, or the new input) is expanded after the history is assembled. A
  response whose history held a capsule stores its instructions, that capsule
  and everything after it as its input, so later `previous_response_id` hops
  keep replaying the summary. An automatically compacted response stores its
  instructions as input and the capsule as its first output item, which
  replays the same way.

## Conversation compaction

`POST /v1/responses/compact` (alias `/responses/compact`) replaces older turns
with a model-written summary so a long agent session fits the context window
again. It is non-streaming and stores nothing.

Request body: `model`, `input` (string or item array), and optionally
`instructions`, `tools`, `max_output_tokens` (summary budget, default 1024,
1 to 16384) and `keep_tokens` (budget of the turns kept verbatim, default 1/16
of the per-slot context window, at most 8192). Sampling fields are accepted
as on `/v1/responses`. `stream: true`, `previous_response_id` and
`conversation` return 400: send the full history in `input`.

```jsonc
{
  "id": "resp_...",
  "object": "response.compaction",
  "created_at": 1760000000,
  "model": "...",
  "output": [
    {"type":"message", "role":"system", "content":"..."},
    {"type":"message", "role":"user", "content":"latest question"},
    {"type":"compaction", "id":"cmp_...", "encrypted_content":"eyJ2IjoxLC..."}
  ],
  "usage": {"input_tokens": 1830, "output_tokens": 212, "total_tokens": 2042, ...}
}
```

`output` holds the system and developer messages, the retained turns, and
exactly one `compaction` item. Version 1 also stores retained turns inside the
capsule, so the returned body can be larger than the original media payload.
The server rejects output that cannot fit its replay request limit; leave
additional room for new items. `usage` counts the one summary generation.
Continue the session by sending `[...output, <new items>]` to `/v1/responses`.
The request-level `instructions` applies to the compaction pass; resend it on
continuation if needed, or put persistent instructions in system/developer
input messages.
The retained turns appear in `output` and again inside the capsule, so that
body carries them twice. A client may send only the system and developer
messages plus the `compaction` item, followed by its new items, for the same
replay at a smaller body size.

**Selection.** System and developer messages are always kept in their roles.
The latest user message is always kept. The newest whole turns that fit
`keep_tokens` are kept; the newest turn is kept even when it alone exceeds the
budget. The budget is measured on the rendered kept context, so it also counts
the system and developer messages, `instructions` and template overhead. When
those alone exceed `keep_tokens`, no earlier turn fits and only the newest turn
(plus the latest user message) is kept. The history is split only before a
user message or right after the
output that closes the last open function call, so a `function_call` is never
separated from its `function_call_output`. A history that ends in a call
without an output keeps that whole tool group. Everything before the split is
summarized.

**Summarization.** The summarized prefix (with the system and developer
messages and `instructions`) is rendered through the model's chat template with
a fixed handoff instruction appended as a user turn, and one generation runs on
the same worker path as `/v1/responses`, at temperature 0, with thinking off
and without tools. Token counts use the `/v1/responses/input_tokens` path.
Successfully fetched image URLs are cached per compaction and reused for every
count and for the summary prompt. Media capability and image-count checks run
before prefetch; failed reads are left to the ordinary renderer for its error. A body with media is still counted on the
worker, so each count takes a worker slot as `/v1/responses/input_tokens` does.

**Failures.** None of these replaces any context:

| Condition | Status | Error |
|-----------|--------|-------|
| Input longer than one context window (multi-pass summarization is not implemented) | 400 | `type: exceed_context_size_error`, `code: context_length_exceeded` |
| Summary prompt plus `max_output_tokens` exceeds the window | 400 | same |
| Summarized context is not shorter than the input | 400 | same |
| Compacted output exceeds the request body limit on replay | 400 | `invalid_request_error`; reduce retained media or `keep_tokens` |
| No safe split before the latest turn | 400 | `invalid_request_error` |
| Summary empty or stopped at `max_output_tokens` | 502 | `server_error` |

The error type matches the server's existing context-overflow error; the code
is OpenAI's, so either kind of client recognizes it.

**Capsule.** `encrypted_content` is base64url (no padding) of
`{"v":1,"summary":"...","retained":[<input items>]}`. Despite the OpenAI field
name it is not encrypted, by design: the client already holds the conversation
it summarizes, so confidentiality adds nothing, and an encryption or HMAC key
would not survive a server restart without new key storage. A capsule is
therefore treated as client data. It is bounded by the request body limit,
its retained items pass the same parser and validation as any `input` item,
and a capsule that nests another capsule is refused. A capsule that does not
decode, or whose `v` is not 1, returns 400 `invalid_request_error`.

**Replay.** When `input` (or chained history) contains a `compaction` item,
everything before the newest one is dropped except system and developer
messages (exact resent duplicates are kept once), and the capsule's summary (as an assistant message) and retained
items take its place, followed by the items after it. This applies to
`/v1/responses`, `/v1/responses/input_tokens`, and `/v1/responses/compact`
itself, so compacting an already compacted session folds the previous summary
into the new one.

Known limits:

- The split may summarize the most recent completed tool group away; upstream mlx-vlm always keeps that group verbatim.
- The summary replays as an assistant turn directly after the system and developer messages, so a chat template that requires the first non-system turn to be a user turn (Gemma, older Mistral) can reject the replayed prompt; upstream mlx-vlm uses the same ordering.

### Automatic compaction

A `/v1/responses` request (also `/responses`) opts in with:

```json
{"context_management": [{"type": "compaction", "compact_threshold": 24000}]}
```

Before generation the server counts the rendered input on the
`/v1/responses/input_tokens` path. The input is the full history: chained
`previous_response_id` or `conversation` items plus `input`, with any capsule
already expanded. When the count reaches `compact_threshold`, or leaves no room
for output in the context window, the history is compacted exactly as
`POST /v1/responses/compact` does (same selection, default `keep_tokens` and
failures), and generation runs on the compacted context. The summary budget is
1/16 of the context window, between 128 and 1024 tokens. Unused
`max_output_tokens` does not count toward the threshold. Compaction never runs
during generation.

The response `output` then starts with the `compaction` item, followed by the
generated items. `usage` counts the final generation only, not the summary
pass. A stored response keeps the system and developer messages as its input
and the capsule as its first output item, so a `previous_response_id`
continuation replays the summary and the retained turns, and compacts again
only when that continuation reaches the threshold. A `conversation` transcript
appends the new input and the output, capsule included, and replays the same
way. A client that resends its history can append the output normally, or keep
only the newest `compaction` item and everything after it.

For every request that carries `context_management`, compacted or not,
generation is capped at the smaller of the output limit (`max_output_tokens`
or the server default) and the context left after the prompt. A reply that
stops at that cap, or at `max_output_tokens`, returns `status: "incomplete"`
with `incomplete_details: {"reason": "max_output_tokens"}`, no `completed_at`,
and the cut-off message item marked `status: "incomplete"` (streaming ends with
`response.incomplete`). A limited reasoning budget above the capped limit is
lowered to one token below it, so the forced end of thinking still leaves room
for visible output. A server with no free slot answers 503 before any count.
Requests without `context_management` are unchanged:
no extra count, no cap, and a `length` stop still reports `completed`.

A streamed request that was compacted sends two MLX extension events after
`response.in_progress` and before the compaction item's
`response.output_item.added` (at `output_index` 0; generated items follow from
index 1, and `response.completed` lists the same items):

```text
event: mlx.compaction.started
data: {"type":"mlx.compaction.started","sequence_number":2,"response_id":"resp_...","input_tokens":5786}

event: mlx.compaction.completed
data: {"type":"mlx.compaction.completed","sequence_number":3,"response_id":"resp_...","input_tokens_before":5786,"input_tokens_after":854}
```

The counts describe the rendered input before and after compaction. Compaction
finishes before the stream opens, so both events arrive together after it;
they carry the counts, not live progress. No compaction means no events.

**Failures.** A compaction failure returns the same HTTP error as
`POST /v1/responses/compact` (see the table above) before any output is
generated, for streaming requests too, since the stream has not opened yet:
for example 400 `exceed_context_size_error` with code
`context_length_exceeded` for a history over one window or a summary that does
not shorten it. Nothing is generated from a context that would overflow. When
the history has no safe split (a single turn, or one unfinished tool group),
the request proceeds uncompacted if the input still leaves room for output, as
upstream mlx-vlm does, and fails with `context_length_exceeded` otherwise. A
malformed `context_management` (not an array, more than one entry, a `type`
other than `compaction`, or a `compact_threshold` that is not an integer from
1 to 100000000) returns 400 `invalid_request_error`.

Not implemented: automatic compaction on `/v1/chat/completions` and
`/v1/messages`, the Codex `compaction_trigger` input item, and multi-pass
summarization of a history longer than one window.

## Tool choice enforcement

`tool_choice` is forwarded to the chat-completions pipeline unchanged, so the
four modes behave exactly as they do on `/v1/chat/completions`:

| `tool_choice` | tools rendered | prompt instruction | constraint | 400 when |
|---|---|---|---|---|
| absent / `"auto"` | all | none | none | never |
| `"none"` | none | none | none | never |
| `"required"` | all | "You must call one or more of the available functions ..." | forced call on grammar-capable formats | `tools` absent or empty |
| `{"type":"function","function":{"name":"f"}}` | only `f` | "You must call the 'f' function ..." | forced call to `f` on grammar-capable formats | `f` not declared, `type` not `function`, or empty name |

The instruction is appended to the first system message (this includes the
`instructions` field, which becomes the leading system message), otherwise to
the last user message, otherwise inserted as a new leading system message. The
stored request and the echoed fields are not modified.

Grammar-capable formats are the ones whose emitted call is a JSON object in a
fixed wrapper that can be read off the loaded chat template: Hermes / Qwen
(`<tool_call>`), Mistral Nemo (`[TOOL_CALLS]`), and Llama 3 (`<|python_tag|>`
or `"parameters":`). There the call is forced through the structured-output
grammar built from the tool schemas, and the response carries a call to a
declared function with `finish_reason: "tool_calls"` (a `function_call` output
item). Templates without a JSON wire shape (ATEM, Gemma 4, XML dialects such as
Qwen3-Coder and GLM, Kimi K2, pythonic) get the instruction and the narrowed
tool list only, and a forced choice that ends without a call is logged at
`warn`. A tool schema the grammar engine cannot express falls back to that same
instruction-only path instead of failing the request.

A `call_id` the model emitted with the call is passed through to the
`function_call` item (and its streaming events). The server mints a `call_`
id only when the model gave none.

A forced `tool_choice` cannot be combined with `text.format` of type
`json_schema`: the two constraints would each claim the whole generation, so
the request returns 400.

## Unsupported tool types

Only function tools are accepted. Built-in/external tool types such as
`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`,
`image_generation`, `mcp`, `custom`, `apply_patch`, and `function_shell` return
400 responses. `mlxcel` does not execute external tools for the Responses API.

## Differences from OpenAI's full API

Notable gaps:

- background responses are lost on restart, and their event streams are retained up to a byte budget rather than in full;
- no input-items pagination endpoint;
- no built-in tools or MCP connector execution;
- no disk-persisted response store;
- no `input_file` ingestion or `file_id` resolution through an uploads store.
