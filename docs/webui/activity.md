# Activity and runtime observations

Activity consumes the authenticated shared WebUI provider; it does not create another fetch client, timer, or model registry. Loading a screen, opening a chart, or selecting a model does not load a provider or touch inference LRU state. Controls use the published `@lablup/ui-common@0.1.0-alpha.19` adapters. Native disclosure elements and the active-request sparkline (an always-rendered SVG dot plot drawn as a single path in the Active requests tile) are product-specific compositions, not replacement control libraries.

## Sources, units and missing data

`GET /ui-api/v1/runtime?model_id=<opaque-id>&autoload=false` projects existing CPU atomics and a redacted slot registry snapshot. It never calls allocator/device getters, initializes MLX, evaluates an array or synchronizes the GPU. `--metrics` and `--slots` remain independent opt-ins. Disabled and unloaded groups return null values and actionable reasons. No process resident, allocator, device-total or exact KV memory reading is invented when no cheap existing snapshot exists. These memory scopes overlap on unified memory and are never summed.

Every measurement includes a unit, scope, timestamp and source/availability reason. Request counters are provider-lifetime recorded completions, not instantaneous throughput; resetting/reloading the provider can reset them. A token count without a matching valid timing source does not produce a decode rate. Server runtime TTFT remains unavailable: chat owns request-send to first observed reasoning/content deltas, which are not server-side model latency. Word counts are never token counts.

The merged #1800 source forwards live prefill observations into slots and counts current context occupancy in `n_prompt_tokens`, including accepted decoded tokens. The runtime `slots.items[].prompt_tokens` follows that existing counter exactly; the UI does not add decoded tokens again. Context capacity uses the effective per-request getter, not an estimate from checkpoint size. Shared-pool capacity is separate, nullable, and exposed only from configured unified-pool geometry. A missing or zero denominator displays unknown and draws no bar. An idle slot (not processing) with a null count reads `0 / D tokens` with an empty bar when the denominator `D` is known: the server reports null for a slot without a task, not for a failed measurement, so an empty slot holds no context. This is the one documented exception to "null is never shown as zero" in `docs/webui/ux-contract.md`. A processing slot with a null count reads unknown, and no occupancy is derived from decoded tokens. Exact KV bytes cannot be inferred from the fraction. Effective parallelism remains unknown unless an existing published worker counter proves it; configured parallelism is explicitly separate. Later #1815 behavior is consumed through the existing worker-published context authority, not reimplemented arithmetic.

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

`WEBUI_PERF_MODE=visible-only-headless` permits a limited diagnostic with five paired runs for each of one and two visible headless clients. It uses the default Chromium headless shell without native-window CDP calls, verifies actual visibility, never synthesizes hidden events, and always reports `status: incomplete` with `hidden_native: not-run`, even when both visible conditions meet the numeric target. Per-condition regressions remain marked `investigate`. Real native hidden-tab overhead must still be measured on a functioning interactive host during integrated acceptance; the original failed headed attempt remains evidence, not a passing measurement.

The page reads as a status page: the header carries the model picker (ready models first), Refresh, Export and the last snapshot time; then four summary tiles in a fixed order (Active requests, Total completed requests, Total completion tokens, Queued requests); the slot table; and one line per operation. A tile whose value is null reads unknown, never 0 and never hidden, with a localized hint for the known server reasons; a measured tile's hint is its observation time. Until the first runtime sample arrives on a live connection, the tiles show decorative skeletons under one localized status. A badge with the unavailable count sits in the summary of the native “All measurements and sources” disclosure, which holds every measurement with its value, scope, observation time and the raw server reason, plus the recent active-request history as text; its content is built only while it is open, so a closed disclosure adds nothing to a runtime poll. Operation kind, state and age come from the catalog; operation and model identifiers appear only inside each row's “Operation details” disclosure, and a model missing from the catalog reads “Unknown model”, never its opaque ID. A failed operation's banner gives a localized reason for its error code, and the code itself appears only inside “Operation details”; the server's error message is never shown. More than eight slots scroll inside the table's named, focusable region rather than the page.

The first full-Chromium headless diagnostic subsequently crashed with SIGBUS during context replacement, before recording any inference sample; the actual browser process log included display-link and notification-center errors. That failed attempt is retained. The visible-only diagnostic therefore uses the same default headless shell as the passing production browser checks. This changes only diagnostic browser infrastructure, not UI code, inference code or the still-outstanding native hidden-host requirement.
