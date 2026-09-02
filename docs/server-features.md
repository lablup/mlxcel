# Server features and deployment guide

This document is the operator-facing map of the `mlxcel-server` and
`mlxcel serve` entry points. It expands the feature summary in the root README
without duplicating the field-by-field compatibility record in
[`llama-server-compat.md`](llama-server-compat.md).

Both entry points start the same HTTP server and keep their server options
aligned. `mlxcel serve` additionally exposes `--estimate-memory` and `--force`
for a one-shot memory preflight before startup.

## Model and store arguments

The meaning of `--models-dir` depends on whether an offline command or a server
entry point owns the flag:

| Surface | Option | Meaning |
| --- | --- | --- |
| `mlxcel generate`, `run`, `inspect`, `download`, `list`, `rm` | `--models-dir PATH` | Override the mlxcel-managed model store. |
| `mlxcel-server`, `mlxcel serve` | `--model-store-root PATH` | Override the mlxcel-managed model store used for repository resolution and downloads. |
| `mlxcel-server`, `mlxcel serve` | `--models-dir PATH` | Start multi-model router mode and discover direct checkpoint subdirectories. |

`MLXCEL_MODELS_DIR` remains the server-side environment override for the model
store. On a server command line, `--models-dir` cannot be combined with
`-m/--model`; use `--model-store-root` when starting a single model from a
non-default store.

Single-model examples:

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080

mlxcel-server -m mlx-community/Qwen3.5-0.8B-4bit \
  --model-store-root /var/lib/mlxcel/models --port 8080
```

Repository ids, bare names under `MLXCEL_DEFAULT_ORG`, and local checkpoint
paths are accepted. `--revision` resolves a HuggingFace revision, but the
mlxcel store is not revision-namespaced; use a separate `--model-store-root`
for deployments that must retain several revisions of the same repository.

## HTTP surfaces

Routes are mounted even when the loaded model cannot serve their modality. In
that case the handler returns a structured capability error rather than making
route discovery depend on the checkpoint.

| Surface | Main routes | Notes |
| --- | --- | --- |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models` | Responses also supports retrieve, delete, and cancel by id. |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` | Uses the loaded chat template and the shared token accounting path. |
| Retrieval | `/v1/rerank`, `/v1/reranking` | Cohere/Jina-compatible reranking surface. |
| Audio | `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations` | Kokoro TTS, audio-capable chat models, and Whisper are described in [`audio-api.md`](audio-api.md). |
| Native llama-server | `/completion`, `/completions`, `/embedding`, `/embeddings`, `/tokenize`, `/detokenize`, `/infill`, `/apply-template` | Native response schemas are intentionally distinct from the OpenAI `/v1/...` schemas. |
| Prompt inspection | `/chat/completions/input_tokens`, `/responses/input_tokens` and `/v1` aliases | Renders and tokenizes through the same path as generation. |
| Operations | `/health`, `/props`, `/slots`, `/metrics`, `/lora-adapters` | Some mutation or inspection details are enabled by their corresponding flags. |
| mlxcel extensions | `/v1/cache/stats`, `/v1/cache/reset`, `/v1/internal/mtp-policy` | Prompt-cache and adaptive speculative-decoding state. |

Most non-`/v1` convenience aliases remain available. Two aliases are important
migration exceptions: `/completions` and `/embeddings` now return the native
`llama-server` schemas. Clients expecting OpenAI objects must use
`/v1/completions` and `/v1/embeddings`.

The checked-in b10621 manifest under `compat/llama-server/b10621/` is the
authoritative compatibility inventory. It classifies every pinned option,
route, and native completion field as supported, aliased, not applicable, or
different by design; no entry is deferred. See
[`llama-server-compat.md`](llama-server-compat.md) for exact schemas,
divergences, and migration notes.

## Context, batching, and caching

`--ctx-size` is the total context budget shared by the parallel request slots.
The default `0` derives the budget from the model. `--parallel -1` is automatic
and currently resolves to four slots; an explicit positive value sets the
maximum concurrent decode batch. Families whose cache layout cannot batch are
clamped to one slot.

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --ctx-size 65536 --parallel 4 --port 8080
```

Continuous batching and prompt-prefix caching are enabled by default. The
native `cache_prompt` request field can opt a request out of both lookup and
donation. `/v1/cache/stats` reports the cache state, while
`POST /v1/cache/reset` clears it. Context shifting is disabled by default to
match b10621; enable `--context-shift` and configure `--keep` when an
application intentionally wants to retain part of an overlong conversation.

See [`CONTINUOUS_BATCHING.md`](CONTINUOUS_BATCHING.md) for scheduler and
disaggregated-serving details and [`turbo-kv-cache.md`](turbo-kv-cache.md) for
KV-cache modes.

## Multi-model router mode

Start router mode without `-m`:

```bash
mlxcel-server --models-dir /srv/models \
  --model-store-root /var/lib/mlxcel/models \
  --models-max 4 --port 8080
```

The registry reconciles three sources in increasing precedence:

1. Cache entries under the model store.
2. Direct checkpoint subdirectories under `--models-dir`.
3. Sections from an optional `--models-preset` INI file.

Requests select an entry with the JSON `model` field, or the `model` query
parameter on GET routes. `--models-max` bounds the resident set with LRU
eviction. Autoload is on by default and can be disabled globally with
`--no-models-autoload` or controlled on a management request with `autoload`.

Router management uses `GET /models` or `/v1/models` for inventory and reload,
`POST /models` for a background download, `DELETE /models?model=...` for a
cache-owned model, and `GET /models/sse` for status events. Model-specific
`/props`, `/slots`, and similar GET routes accept `?model=...`. All management
routes pass through the ordinary API-key middleware.

## LoRA adapters

The default LoRA path keeps adapters unfused so scales can change without a
restart:

```bash
mlxcel-server -m models/base \
  --lora adapters/code,adapters/style \
  --lora-scaled adapters/domain:0.5

curl http://localhost:8080/lora-adapters

curl http://localhost:8080/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id":0,"scale":0.8},{"id":2,"scale":0.25}]'
```

Each request snapshots its adapter scales at admission, so a live update does
not alter an in-flight generation. Native completion requests may also provide
their own `lora` selection. `--lora-init-without-apply` starts every adapter at
scale zero.

`--lora-fuse` bakes adapters into the base weights and removes their decode
overhead, but the scales then remain fixed for the life of the process. Live
updates and per-request selection are refused in fused mode. Distributed
tensor and pipeline loaders currently imply fused behavior.

An adapter has to map onto the checkpoint it is loaded with, on the fused and
the unfused path alike. Every tensor in `adapters.safetensors` must be one half
of a `<layer>.lora_a` / `<layer>.lora_b` pair whose base weight the model
actually holds. An adapter trained for another architecture, a renamed
projection, or a stray tensor fails the load with every offending name listed,
rather than starting a server that answers from the base weights while
reporting the adapter as loaded. DoRA adapters are refused outright: applying
their low-rank pair without the magnitude vectors would produce weights that
match neither the base model nor the fine-tune. In a pipeline-parallel run a
stage that owns none of the adapter's layers still applies nothing, which stays
a valid load.

## Sampling, structured output, and reasoning

The server exposes the commonly used llama-server sampling stages, including
top-k/top-p/min-p, typical-p, XTC, DRY, repetition/frequency/presence penalties,
Mirostat, dynamic temperature, adaptive-p, logit bias, and probability reports.
Startup defaults can be overridden on supported request schemas.

`--grammar`, `--grammar-file`, `--json-schema`, and `--json-schema-file`
constrain output. Native completion and infill requests additionally accept
GBNF lazy-trigger fields. Grammars are parsed and enforced during sampling;
invalid constraints are rejected instead of being accepted as hints.

Reasoning-capable checkpoints support runtime placement through
`--reasoning-format`, optional aliasing, assistant prefill, token budgets, and
the native chat-control endpoint. Model-emitted tool calls can be parsed and
returned to the client, but mlxcel does not execute tools server-side.

### Tool calling and `tool_choice`

`tool_choice` on `/v1/chat/completions`, and on the Responses and Anthropic
Messages translations of it, is enforced rather than echoed:

- absent or `"auto"`: every declared tool is rendered into the prompt.
- `"none"`: no tool is rendered.
- `"required"`: every tool is rendered and an instruction is appended to the
  prompt. The request is rejected with 400 when `tools` is absent or empty.
- `{"type": "function", "function": {"name": "f"}}`: only `f` is rendered and
  the model is instructed to call it. The request is rejected with 400 when `f`
  is not declared, `type` is not `function`, or the name is empty.

The instruction is appended to the first system message, otherwise to the last
user message, otherwise inserted as a new leading system message. The request
object itself is never modified, and the prompt cache keys on the prompt that
was actually rendered.

On templates whose tool-call wire shape is a JSON object inside a fixed wrapper,
the call is additionally forced through the structured-output grammar built from
the tool schemas, so `required` always yields a call to a declared function and
the named form always yields that function, with `finish_reason: "tool_calls"`.
The wrapper is read off the loaded chat template: Hermes / Qwen (`<tool_call>`),
Mistral Nemo (`[TOOL_CALLS]`), and Llama 3 (`<|python_tag|>` or
`"parameters":`). Formats without a JSON wire shape (ATEM, Gemma 4, XML
dialects such as Qwen3-Coder and GLM, Kimi K2, pythonic) get the instruction and
the narrowed tool list only, and a forced choice that ends without a call is
logged at `warn` with the format name.

Tool schemas that the grammar engine cannot express fall back to that same
instruction-only path rather than failing the request: the compile failure is
logged at `warn` and generation runs unconstrained. Only a `response_format`
schema, which the client asked for explicitly, turns a compile failure into a
400.

A forced `tool_choice` cannot be combined with a `response_format` schema (400).
The Anthropic `tool_choice` values `{"type": "any"}` and
`{"type": "tool", "name": "f"}` map to `required` and the named form, so the
same rules apply on `/v1/messages`.

## Observability, persistence, and resumable streams

- `GET /health` and `/v1/health` provide readiness.
- `GET /props` reports resolved model, modality, cache, context, speculative,
  and endpoint configuration.
- `GET /slots` exposes live request-slot state. `--slot-save-path DIR` enables
  save, restore, and erase actions on `/slots/:id_slot`.
- `--metrics` enables Prometheus output at `GET /metrics`, including the
  b10621 `llamacpp:` families and mlxcel-specific families.

A streaming request with `X-Conversation-Id` is retained for resumable delivery.
`GET /v1/stream?conv_id=...&from=N` replays and follows it,
`POST /v1/streams/lookup` reports known ids, and
`DELETE /v1/stream?conv_id=...` cancels and removes it. Without that header, a
client disconnect continues to cancel generation normally.

`--sleep-idle-seconds N` frees the model worker and prompt cache after the idle
window. Health and inspection routes remain available while sleeping; the next
inference request reloads the model and performs its own prefill. Negative
values disable sleep, which is the default.

## Live settings

`--settings` opts into authenticated `GET` and `PATCH` on `/v1/settings` and
`/settings`. The endpoint is absent by default. Startup refuses to expose it on
a non-loopback TCP listener without an API key; loopback and Unix-socket
listeners may run it without authentication.

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --host 0.0.0.0 --api-key change-me --settings

curl http://localhost:8080/v1/settings \
  -H 'Authorization: Bearer change-me'

curl -X PATCH http://localhost:8080/v1/settings \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"default_temperature":0.2,"reasoning_budget":1024}'
```

GET returns the typed schema, current values, and a stable fingerprint. A plain
object PATCH performs a merge; the explicit form is
`{"op":"merge|replace","values":{...}}`. Valid independent settings are
published together through one atomic swap, while read-only or invalid names
are returned in `rejected`. Mutable groups include request timeouts, generation
defaults, DRY, language bias, reasoning budget, chat-template kwargs, loop
detection, and diffusion defaults.

## Transport and access control

Configure API keys with repeatable `--api-key` values, comma-separated values,
or `--api-key-file`. With keys enabled, only `/`, `/health`, and `/v1/health`
are public; model, metrics, settings, router, and generation routes require a
valid bearer key.

The default CORS mode follows b10621 and is permissive. Use
`--allowed-origins` for an explicit list, or the `--cors-*` compatibility flags
when migrating an existing llama-server command. TLS is enabled by supplying
both `--ssl-cert-file` and `--ssl-key-file`. A `--host` ending in `.sock` starts
a Unix-domain-socket listener, and `--api-prefix` nests all routes below a path.

With `AIP_MODE=PREDICTION`, the server also honors the Vertex AI custom-container
`AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, and `AIP_PREDICT_ROUTE` variables. See
[`environment-variables.md`](environment-variables.md) for the complete
environment map.

## Deliberate non-goals

mlxcel does not ship llama-server's embedded Web UI and does not execute tools,
MCP servers, an agent loop, or the UI CORS proxy. Enabling those compatibility
flags fails at startup instead of silently pretending that the capability is
active. `/tools` and `/cors-proxy` retain the disabled-feature response shape
expected by clients. Client-declared tools and parsed `tool_calls` remain
supported because execution stays with the client.

For every b10621 option and native field, including intentional differences
not repeated here, consult [`llama-server-compat.md`](llama-server-compat.md).
