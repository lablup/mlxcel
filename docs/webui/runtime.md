# Runtime observation semantics

`GET /ui-api/v1/runtime?model_id=…&autoload=false` is a cheap read-only projection. It never initializes a model, admits an inference request, touches the model LRU or prompt-cache entries, calls allocator/device getters, synchronizes/evaluates tensors, or changes the Prometheus scrape baseline. It reads existing CPU atomics and a bounded slot-registry snapshot. Worker counters can change between individual atomic reads; this is not a transaction across counters.

## Availability and scope

- Live counters require a loaded provider and the existing `--metrics` opt-in. Slots independently honor the existing slots endpoint option; disabling it does not become a WebUI permission bypass. Configuration observations retain the existing `server_startup` settings report and do not enable settings mutation.
- An unloaded/loading/failed provider has no live sample. Draining providers can remain observable until their worker is no longer loaded. Absent counters are `null` with a reason, not zero.
- Every measured number carries a unit, observation timestamp, model scope and source/window description. The timestamp means when this CPU sample was read, not when the scheduler last wrote its gauge. Prompt-cache gauges explicitly warn that publication can lag eviction.
- `server` memory scope means process-wide, never one selected model. Resident memory and allocator active/cache/peak remain unavailable because this projection has no existing cheap CPU source. Device total/utilization, resident weight estimates and exact KV bytes are also unavailable. Downloaded checkpoint bytes are not resident weights. Unified/shared memory categories must not be summed.

## Counters and timing

`active_requests` is the scheduler's active decode sequence gauge, not all HTTP requests or queued prefill. `queued_requests` is the existing pending prefill/single-stream queue gauge. Single-stream DiffusionGemma/LLaDA/Florence workers publish queue reservations but do not publish batch active/decode counters; the latter remain unavailable for these providers.

`completed_requests_total`, `completion_tokens_total` and `generation_time_ms_total` come from successfully recorded route results. Their window is the loaded provider's lifetime, and they can reset after model reload. They exclude errors not recorded by the route and are not admission counters. Generation time has route-specific semantics, including elapsed time for some non-chat routes; it is not a decode-only denominator.

A chat completion counts the same whether it streams or not. A streamed one, which is what the WebUI chat sends, is recorded once when its generation finishes, with the token counts its usage chunk reports, whether or not the request asked for that chunk. A stream the client cancelled is not a completion and is not counted: a disconnect of a stream sent without `X-Conversation-Id`, `DELETE /v1/stream`, or replacement of the session by a new request with the same `X-Conversation-Id`. A resumable stream (`X-Conversation-Id`) whose client only disconnects keeps generating into its session, so it is counted when it finishes.

`decode_tokens_total` and `decode_time_us_total` read the existing completed-request `BatchObservability` counters. They remain unavailable until a completed decode timing sample has been published. A backend that never publishes them is not shown as a measured zero. Independent atomic reads must not be converted into a claimed exact per-request or instantaneous rate. TTFT (request send to first output token, with thinking/content distinctions) and decode rate remain unavailable on this runtime surface; request-aware chat instrumentation is a separate source.

## Slots and context

Slots mirror the counters used by native `GET /slots`, without copying its request parameters, prompt text or generated text even when native debug retention is enabled. At most 256 slots are projected, with explicit truncation diagnostics. A poisoned registry is unavailable, not empty.

The historically named `prompt_tokens` field is **current context occupancy**, mirroring native `n_prompt_tokens`: processed prompt tokens during prefill, then prompt plus accepted decoded tokens. Do not add `decoded_tokens` again. Idle slots can retain their last task's counters; a never-used slot has null counters. These are observational slots, not scheduler admission permits.

`request_context_tokens` uses the same effective post-load context getter as native slots. `shared_pool_context_tokens` is only populated for a known unified shared token budget; it is not the sum of every slot's request window. Zero/unknown geometry is null, so no percentage is derived from a missing denominator. Token fraction never implies exact KV memory bytes.

Configured parallelism is not necessarily effective decode width. The existing #1815 publication proves width one when a non-batching worker restores a split context window; that is reported as one. Other workers currently have no general CPU-published effective-width fact, so this field remains null rather than inferring it from `--parallel` or `max_batch_size`. Future worker capabilities can fill it without changing the arithmetic contract.
