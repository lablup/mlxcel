# Activity and runtime observations

Activity consumes the authenticated shared WebUI provider; it does not create another fetch client, timer, or model registry. Loading a screen, opening a chart, or selecting a model does not load a provider or touch inference LRU state. Controls use the published `@lablup/ui-common@0.1.0-alpha.19` adapters. Native disclosure elements and a lazy SVG dot plot are product-specific compositions, not replacement control libraries.

## Sources, units and missing data

`GET /ui-api/v1/runtime?model_id=<opaque-id>&autoload=false` projects existing CPU atomics and a redacted slot registry snapshot. It never calls allocator/device getters, initializes MLX, evaluates an array or synchronizes the GPU. `--metrics` and `--slots` remain independent opt-ins. Disabled and unloaded groups return null values and actionable reasons. No process resident, allocator, device-total or exact KV memory reading is invented when no cheap existing snapshot exists. These memory scopes overlap on unified memory and are never summed.

Every measurement includes a unit, scope, timestamp and source/availability reason. Request counters are provider-lifetime recorded completions, not instantaneous throughput; resetting/reloading the provider can reset them. A token count without a matching valid timing source does not produce a decode rate. Server runtime TTFT remains unavailable: chat owns request-send to first observed reasoning/content deltas, which are not server-side model latency. Word counts are never token counts.

The merged #1800 source forwards live prefill observations into slots and counts current context occupancy in `n_prompt_tokens`, including accepted decoded tokens. The runtime `slots.items[].prompt_tokens` follows that existing counter exactly; the UI does not add decoded tokens again. Context capacity uses the effective per-request getter, not an estimate from checkpoint size. Shared-pool capacity is separate, nullable, and exposed only from configured unified-pool geometry. A missing or zero denominator displays N/A. Exact KV bytes cannot be inferred from the fraction. Effective parallelism remains unknown unless an existing published worker counter proves it; configured parallelism is explicitly separate. Later #1815 behavior is consumed through the existing worker-published context authority, not reimplemented arithmetic.

## Freshness and lifetime

One shared synchronizer samples visible selected-model observations at two-second intervals and bounds each snapshot request to ten seconds. Hidden tabs abort observation requests and the observation event stream, cancel timers, and resume with a fresh snapshot when visible; inference requests are not owned by that synchronizer. An SSE disconnect marks observation stale immediately. Changing selection cancels only observation through the provider’s `selectionChanged()` path. Authentication/logout still owns complete session cancellation.

The reducer keeps one selected-model five-minute ring with at most 150 two-second samples. Sparse dots do not interpolate polling gaps or fabricate progress. Model changes and server restart reset the history; it is never persisted. Repeated runtime polls at the same lifecycle sequence may refresh counters, while older sequence responses remain rejected. Complete unfiltered catalog snapshots clear a removed selection; partial pages and stale generations cannot do so.

Operation history belongs to the server session, with at most 200 terminal records for one hour. Active operations are never silently discarded. The UI reconciles reconnects through the shared provider and does not replay an uncertain POST. Cancellation is a request, not successful worker termination; cancelling remains visible until the server reports a terminal state. Failed operations show a typed error code without exposing arbitrary server error text.

Diagnostics export is a strict allowlist of connection state, counts, availability and numeric measurements. It excludes paths, model/repository identifiers, prompts, generated output, keys, operation errors and runtime free-text reasons, even when a debug slot endpoint would expose request data.

## Root-owned real acceptance

The script below is an acceptance harness, not a completed measurement. Start an already-loaded real model with WebUI, metrics and slots enabled, no unrelated GPU workload, and a private API-key file. From `webui/`:

```sh
WEBUI_PERF_BASE=http://127.0.0.1:8080/ \
WEBUI_PERF_KEY_FILE=/private/path/key \
WEBUI_PERF_INFERENCE_MODEL=model-id \
WEBUI_PERF_MODEL_ID=mdl_opaque \
WEBUI_PERF_OUTPUT=/tmp/activity-paired.json \
node scripts/activity-performance.mjs
```

The harness uses an actual headed browser and actual `document.hidden` after window minimization; it fails rather than substitutes a synthetic visibility event. One and two client windows render the real Activity screen with the shared observation schedule. Five alternating-order UI-off/treatment pairs per condition use the same native completion workload and backend `timings.predicted_per_second`; missing timings fail. It records median paired degradation, pair spread and baseline coefficient of variation. A median degradation above 2% or baseline CV above 5% is flagged for investigation, not silently accepted. The optional `WEBUI_PERF_PROMPT_FILE` supplies a fixed long prompt; prompts and model output are not written to the results.

A separate long-prefill acceptance must sample both native `/slots` and canonical runtime slots while an actual chat request runs, comparing current context and accepted decode counters against #1800 source semantics. Observation timings, browser status, models, flags and hardware must accompany results. Automated fixture/browser tests do not establish real-model behavior or the performance target. Native Safari/VoiceOver and real 200% page zoom remain deferred to the final integrated manual acceptance by maintainer decision.

For that long-request capture, use the same already-loaded server, key and model IDs, plus a fixed prompt file long enough to observe prefill and decode while staying below the configured context limit:

```sh
WEBUI_PERF_BASE=http://127.0.0.1:8080/ \
WEBUI_PERF_KEY_FILE=/private/path/key \
WEBUI_PERF_INFERENCE_MODEL=model-id \
WEBUI_PERF_MODEL_ID=mdl_opaque \
WEBUI_PERF_PROMPT_FILE=/private/path/long-prompt.txt \
WEBUI_SLOTS_OUTPUT=/tmp/activity-slots.json \
node scripts/activity-slots.mjs
```

Run from `webui/`. The script samples native slots before and after canonical runtime slots every two seconds during a real chat stream, with a ten-minute bound and owned-request cleanup on failure. It validates bounded UTF-8 SSE frames, rejects error payloads and malformed or truncated streams, and requires a successful finish reason followed by `[DONE]` without retaining generated text. Offline parser regressions run through `pnpm unit`. It also fails if no processing/decode phase is observed; a capture success is explicitly `captured-requires-comparison`, not a performance or correctness pass. Compare each sequential bracket and account for progress between reads. The result excludes prompts, generated output, debug slot parameters and credentials.

## Restricted desktop diagnostic mode

The default `WEBUI_PERF_MODE=full` requires a working headed Chromium layout/visual viewport and real visible/hidden transitions. Every requested browser condition is preflighted before warmup inference; zero or invalid geometry fails closed. On the September 15 remote host, even a blank headed page reported zero layout and visual viewport dimensions after native bounds changes. Headless Chromium had usable geometry, but native minimize and foreground-tab changes did not produce `document.hidden=true`. These observations are an environment limitation, not a UI or GPU root-cause finding.

`WEBUI_PERF_MODE=visible-only-headless` permits a limited diagnostic with five paired runs for each of one and two visible headless clients. It uses full Chromium, verifies actual visibility, never synthesizes hidden events, and always reports `status: incomplete` with `hidden_native: not-run`, even when both visible conditions meet the numeric target. Per-condition regressions remain marked `investigate`. Real native hidden-tab overhead must still be measured on a functioning interactive host during integrated acceptance; the original failed headed attempt remains evidence, not a passing measurement.

Runtime defaults to available primary request counters and slot occupancy. Cumulative counters are labeled totals; unavailable counts remain explicit. All metrics, individual availability reasons, scopes and sampling provenance remain accessible through the native “All measurements and sources” disclosure.
