# WebUI architecture contract

This document is the implementation contract for the first WebUI gate in epic #1834. It freezes the shared boundaries that backend, frontend, bundle and integration owners must use; it does not implement runtime routes or pages.

The implemented static-bundle boundary and contributor commands are documented in [Developing the bundled WebUI](bundling.md) ([한국어](bundling.ko.md)); production startup and control adapters remain assigned to downstream issues.

## Ownership and change rule

- The canonical machine contract is `docs/webui/api.yaml`, the generated strict DTO surface is `docs/webui/generated/ui-api.d.ts`, and executable examples live under `tests/fixtures/webui/`.
- A contract change must update the producer schema, generated DTOs, fixtures, and at least one consumer-facing requirement-map row in the same PR before dependent WebUI work resumes.
- `python3 scripts/ci/check_webui_contract.py` is the drift gate; `--fix` rewrites only generated DTOs from the schema.

## Server boundaries

`mlxcel-server --webui` and `mlxcel serve --webui` share the same server resolution. The UI control plane is added under `{api_prefix}/ui-api/v1`; existing OpenAI, llama-server and router compatibility endpoints keep their shapes and status codes. Browser-facing adapters call the existing owners: `RouterPool` for discovery/load/unload/download/remove, current settings routes for mutable/read-only settings, slot/cache/worker metrics for observation, and current chat/Responses routes for inference.

Do not add a second registry, a child-process supervisor, arbitrary path APIs, shell commands, SSR, a Node service, remote telemetry, CDN assets, or a service worker. `router_front.rs` remains out of scope because it is a distributed tokenizer-bearing service, not the local WebUI foundation.

## URL and authentication model

The shell is served at `{api_prefix}/webui/` by later bundle/startup issues. Redirect only the missing trailing slash and preserve `/`, `/health`, `/v1/health`, and every existing API path. `api_prefix` comes from the validated server config, never from browser input.

Every `/ui-api/v1` endpoint requires bearer authentication. In WebUI mode, shell/assets may be public but bootstrap, catalog, control, settings, runtime and events are administrator APIs. Loopback without a configured key may use a strong session-only terminal-presented key; non-loopback requires an explicit key and TLS or a documented loopback reverse proxy. No key goes in URLs, HTML, structured logs, diagnostics, IndexedDB, or SSE payloads.

Browser-accessible APIs must add Host, Origin and Fetch-Metadata checks when WebUI mode is enabled, including compatibility mutation routes (`POST /models/load`, `POST /models/unload`, `POST /models`, `DELETE /models`, settings, and `GET /models?reload=...`). Bearer-authenticated non-browser clients without `Origin` remain supported. Do not add CORS wildcards.

## Identity and catalog contract

A catalog entry has two identifiers: `identity.id` is the opaque stable UI model ID and `identity.inference_id` is the current request-facing model string accepted by compatibility routes. Stable ID input is the canonical JSON object `{"entry_key":<repo id, preset section, or basename>,"namespace_hash":<sha256(redacted canonical source key)>,"source":<cache|models_dir|preset|single_model>,"source_rank":<cache=0,models_dir=1,preset=2,single_model=3>,"version":1}` with object keys sorted, UTF-8 encoding and no whitespace. The raw source key is never serialized to clients; only `source_key_hash = sha256(redacted canonical source key)` is exposed. `identity.id` is `mdl_` plus unpadded base64url(SHA-256(canonical identity JSON)); `tests/fixtures/webui/identity-vectors.json` is the executable vector set, and collision fixtures must use those IDs rather than arbitrary strings. The same configured source/entry keeps the same stable ID across rescan and restart. `content_fingerprint` is separate from identity and is computed during bounded metadata projection or explicit refresh rather than on every GET; `revision` changes on every observable lifecycle/catalog mutation.

Source precedence mirrors the existing router: cache `<` models-dir `<` preset. Downloaded, architecturally supported, complete, runnable on this backend, loaded, and selected in the browser are separate facts. `mlxcel list` is not the WebUI catalog and `/v1/models` must not be extended with invented OpenAI semantics.

## Endpoint summary

| Endpoint | Contract |
|---|---|
| `GET /bootstrap` | Returns schema version, server instance ID, mode, build/features, canonical relative API base, action availability and redacted roots. It never initializes weights, tokenizers or arbitrary directory enumeration. |
| `GET /catalog` | Deterministic paged inventory. Query keys are `limit` (default 50, max 200), `cursor`, `q`, `source`, `task`, `lifecycle`, `support`, and `completeness`. Filtering never downloads or loads. |
| `GET /catalog/{id}` | Reads one opaque ID. Unknown IDs are 404 structured errors. |
| `POST /catalog/refresh` | Starts a bounded background rescan job. GET never mutates. |
| `POST /model-actions` | Accepts load/unload with opaque `model_id`, `expected_revision`, printable `idempotency_key`, optional `load_profile`, and optional explicit `eviction_target_id`. `load_profile` is limited to existing startup-backed fields only: `ctx_size` (1..262144), `n_parallel` (1..32), and `kv_cache_mode` using the same spellings accepted by `--kv-cache-mode`; no sampling or reasoning fields are part of the next-load contract. `202` means accepted, not ready. |
| `POST /downloads` | Accepts only bounded public `owner/name` repository IDs plus optional bounded revision and printable idempotency key. Dot-only path segments, URL syntax, control characters and filesystem path spellings are rejected before any network or cache write. Repo metadata/revision is pinned before writing weights. |
| `POST /model-removals` | Cache-only deletion after server checks; UI confirmation is additional, not security. Busy/loading/downloading/draining entries return 409. |
| `GET /operations`, `GET /operations/{id}`, `POST /operations/{id}/cancel` | Bounded active and terminal operation records. `target` and terminal `result` are discriminated DTOs, not arbitrary metadata bags. Cancellation is terminal only after the worker stops; unsupported cancellation is a typed 422. |
| `GET /events` | Authenticated SSE carrying `UiEvent` JSON with `event_id`, `emitted_at`, `server_instance_id`, monotonic `sequence`, concrete `type`, and typed `payload`. Last-Event-ID replay is valid only inside the same server instance and retained ring. |
| `GET /runtime?model_id=...` | Model-scoped observation from existing counters with null+reason for unsupported data. No autoload or sampling work. |

## Existing surface compatibility inventory

- `/settings` and `/v1/settings` are optional and may partially apply a PATCH: HTTP success is not an all-fields success. The UI contract reports `partial_errors` and `overridden_by_cli` so consumers show the effective value and the reason a field did not change.
- `/props`, `/slots`, and `/metrics` remain compatibility surfaces with existing gates and units. WebUI runtime snapshots project these facts instead of scraping or reinterpreting raw endpoint text.
- Chat streaming already carries content deltas, additive `reasoning_content` and optional `reasoning` alias fields, tool-call start/argument deltas, final finish chunks, and usage chunks when `stream_options.include_usage` is true. The UI must not treat an empty `content` delta with reasoning/tool data as a broken stream.
- Responses streaming uses typed `response.*` events including reasoning text deltas, output text deltas, function call argument deltas, item completion, and final usage. The WebUI chat page may display/copy tool calls but must not execute them.

## Limits and backpressure

All timestamps in API payloads are RFC3339 UTC strings with explicit offset unless a field name ends in `_ms`, in which case it is Unix milliseconds. Initial limits are deliberately conservative and part of the contract: catalog default page 50 and max 200; list responses max 200 items; cursor tokens max 512 bytes; JSON request body 2 MiB for UI API; metadata per catalog entry 16 KiB; settings and measurement maps max 64 fields; SSE ring 1,024 events retained for 10 minutes; terminal operation history 200 records for one hour; active operations max 64; concurrent loads 1; concurrent downloads 1; next-load `ctx_size` max 262144 and `n_parallel` max 32. Active operations are never silently dropped. Unknown total bytes means indeterminate progress (`total_bytes: null`, `indeterminate: true`), never 0% or 100%.

Slow SSE consumers are disconnected once they fall out of the retained ring. Because `/catalog`, `/operations` and `/runtime` can be fetched at different sequence positions, the client starts SSE replay from the minimum snapshot sequence it holds, then discards duplicate events per resource whose `sequence` is not newer than that resource snapshot. The SSE `id:` field and JSON `event_id` are the same opaque value; clients store it for reconnect but compare only `server_instance_id` and numeric `sequence`. A reconnect with an unknown event ID, gapped sequence, or changed `server_instance_id` receives a reset/server-restart signal and must refetch `/bootstrap`, `/catalog`, `/operations`, and the selected `/runtime` snapshot. The polling fallback uses one non-overlapping loop: 2 seconds visible, 30 seconds hidden, exponential backoff on failures.

## Settings precedence

Effective values resolve in this order: server startup defaults, per-model preset, next-load profile, then browser request overrides. Existing explicit CLI flags are authoritative; fields they override are reported in `overridden_by_cli`. Browser request overrides affect that request only and never mutate the loaded model or next-load profile. The next-load profile is intentionally not a generic settings bag: #1835 freezes only `ctx_size`, `n_parallel`, and `kv_cache_mode`, leaving request sampling and reasoning controls to the existing request path instead of inventing new runtime support.

Capabilities that are valid before load: stable identity, source, completeness, architectural support, declared input/output tasks, removability, download state, and action availability. Provider-ready-only capabilities: actual context window, tokenizer/template-derived chat behavior, image/audio support after processor readiness, live mutable settings, slot occupancy, TTFT/decode measurements, and worker-exit observations.

## Review checklist

Route owner: verify every response is produced from existing runtime truth, applies the documented status code, redacts paths/tokens, respects idempotency scope, never holds registry/settings locks across network/GPU waits, and updates fixtures when behavior changes.

Client owner: verify generated DTO drift gate passes, all actions use expected revisions and idempotency keys, reset/gap/server-restart events force resnapshot, indeterminate progress is rendered honestly, and request-only settings never become persistent UI state.

Integration owner: verify compatibility routes and UI routes use one coordinator, producer/consumer/fixtures land atomically, browser security checks cover mutation surfaces, screenshot/test IDs match `ux-contract.md`, and no downstream issue invents an API/state/UX decision already frozen here.

## Running the contract gate

Create an isolated Python environment, install `scripts/ci/webui_contract_requirements.txt`, and run `make verify-webui-contract WEBUI_CONTRACT_PY=/path/to/venv/bin/python`. The CLI delegates schema/fixture checks to `scripts/ci/webui_contract_checks.py` and independent negative cases to `scripts/ci/webui_contract_self_tests.py`. `--fix` changes generated TypeScript declarations only; it does not accept failing fixtures. The generated declaration file is exempt from hand-written module size limits because its source of truth is the schema.

Schema patterns use ECMAScript-compatible syntax. The strict end assertion `(?![\s\S])` is intentional: `$` also matches before a final newline in JavaScript and Python and therefore does not exclude trailing control characters. Rust implementations must use equivalent structural/full-string validation rather than copying lookahead expressions into the Rust `regex` crate, which does not support lookaround. Negative cases mutate one field at a time so an unrelated invalid field cannot mask a missing constraint.

The verifier registers an explicit date-time checker and refuses schema formats without an active checker. This avoids silently accepting malformed timestamps when optional `jsonschema` format dependencies are absent. These checks validate contract artifacts, not live Rust serialization or runtime behavior; route and integration owners must exercise actual producers against the same fixtures in their dependent issues.

## Catalog implementation boundary

The metadata-only catalog adapter and its current integration limits are documented in [catalog.md](catalog.md) ([한국어](catalog.ko.md)). It projects the existing router/provider authority; production startup mounting remains a separate integration step.

## Shared typed client and state authority

`webui/src/api` is the only browser transport layer. `WebUiApiClient` accepts an optional `apiBase` for tests and integration, but it must validate to a same-origin path prefix and every request stays under `/ui-api/v1`; pages must not construct absolute backend URLs or put credentials in query strings. The client keeps the bearer key in memory only, attaches it as `Authorization: Bearer ...`, clears it on 401, treats 403 as forbidden rather than a login retry, and sends model observation through `GET /runtime?model_id=<opaque id>&autoload=false` so browsing never loads a model.

`webui/src/state` is the shared headless state authority for downstream screens. Integrators wrap the app in `WebUiProvider({children, apiBase?})`, read `useWebUi()` for `{auth, connection, bootstrap, catalog, operations, runtimes, selectedModelId, lastUpdatedAt, error}`, and call `useWebUiActions()` for `login(token)`, `logout()`, `refresh()`, `selectModel(modelId)`, `loadModel(request)`, `unloadModel(request)`, `downloadModel(request)`, `removeModel(request)`, `refreshCatalog(idempotencyKey)`, `cancelOperation(operationId)`, and `refreshRuntime(modelId)`. Models, Chat, Activity and Settings pages must compose these hooks instead of creating private fetch wrappers, persistence, polling timers or model caches.

The reducer uses per-resource sequence fences: catalog snapshots, operation records, model lifecycle revisions and runtime snapshots each discard only events older than their own authoritative sequence. The reconnect cursor is the minimum held fence, not the newest global event, so independently fetched catalog/operations/runtime snapshots cannot hide a transition that occurred between them. Unknown POST outcomes are recorded by idempotency key and reconciled through catalog/operation refreshes; the client never blindly retries a load, download, removal, or refresh after a connection break, and unmatched outcomes expire as explicit stale/error state rather than disappearing silently.
