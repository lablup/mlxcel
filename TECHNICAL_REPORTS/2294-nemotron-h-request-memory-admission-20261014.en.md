# PR #2294: Admit Nemotron-H requests against a checked memory budget

**Date**: 2026-10-14
**Status**: Implementation and validation complete; remaining CI jobs pending
**Risk**: High because the guard sits between HTTP streaming and GPU execution and intentionally fails closed

## Executive summary

PR #2294 adds conservative request-memory admission for single-slot, text-only Nemotron-H serving on Metal. Requests whose checked allocation envelope exceeds the lower of the runtime limits and `MLXCEL_REQUEST_MEMORY_BUDGET` receive a typed HTTP 503 before GPU prefill or SSE headers; context overflow still receives HTTP 400, and unsupported models and execution modes keep their prior behavior.

## 1. Problem statement

PR #2290 bounded Nemotron-H SSD prefill scratch, but a valid in-context request could still exceed Metal memory through resident weights and caches, output KV, recurrent state, activations, logits, attention fallback, and bounded SSD workspace. Such an allocation failure can terminate serving rather than reject one predictable request. Startup memory estimates cannot decide this at the actual call site because the relevant prompt, planned prefill geometry, requested output, retained allocations, and configured runtime ceilings are only known immediately before prefill.

This guard is deliberately not a universal OOM guarantee. Other processes, driver allocations, fragmentation, unexpected kernel workspace, and allocations after admission can still consume memory.

## 2. Technical review

### 2.1 Admission boundary and error contract

The final check runs on the scheduler thread after preceding width-one work finishes and before request GPU work. The provider waits for a one-shot admission event before an HTTP route commits streaming headers. A rejection is returned as HTTP 503 with `error.type` and `error.code` equal to `request_memory_budget_exceeded`, plus `estimated_bytes` and `budget_bytes`. Existing context validation retains precedence and returns HTTP 400. Cancellation and worker-disconnect paths complete or cancel the handshake so an abandoned or unsupported request cannot leave a waiter hanging.

The review closed two route-level gaps. Guarded requests cannot enter batched prefill, whose sequential fallback bypassed the final single-request check, and unsupported paths acknowledge immediately instead of waiting for an admission event they never emit. Legacy `--no-batch` remains sequential but now uses the same size-one scheduler guard.

### 2.2 Checked estimate and budget

The pure estimator uses normalized loaded `NemotronHConfig`, the loaded embedding output dtype, actual cold `PrefillPlan` pieces, architecture-aware KV geometry, and a current MLX memory snapshot. All products and sums use checked arithmetic; unknown dimensions, unknown budgets, invalid dtypes, and overflow fail closed. The envelope charges current active allocations, cold attention KV with growth and snapshot ownership, recurrent and convolution state, per-layer activations, logits, MoE intermediates, conservative float32 attention scores, and float32 SSD buffers with the model's internal chunk size.

The default budget is the minimum positive finite value among the device recommendation, the wired cap actually applied by mlxcel, and the allocator cap. `MLXCEL_REQUEST_MEMORY_BUDGET` accepts the existing binary size grammar but can only lower that ceiling. Empty, zero, `none`, negative, nonfinite, and overflowing explicit values stop startup.

### 2.3 Residency and cache policy

Metal Nemotron-H weights are evaluated at the shared construction boundary, including when warmup is disabled, so lazy graphs cannot make the first request observe a false low resident baseline. The constructor also covers fused checkpoint adapters and embedded VLM or VoiceChat users, but only eligible text serving enables admission. Runtime unfused LoRA is rejected at startup and staged adapter state is drained because arbitrary-rank float32 workspace is not modeled.

Guarded requests prefill cold, do not restore or donate prompt snapshots before admission, and skip background prompt-cache warmups. Retained snapshots remain part of active memory. This conservative policy can reject a request that would fit with optimistic cache reuse or a fused kernel, but it avoids performing request work before the final measurement.

### 2.4 Scope and compatibility

The policy is restricted to local, non-speculative, text-only Metal Nemotron-H with effective width one. CUDA, ROCm, CPU, other families, multimodal input, tensor or pipeline parallel execution, disaggregated worker handoffs, speculative decoding, and multi-slot reservations retain their prior policy. The PR adds no dependency or data-format change and does not alter model numerics.

## 3. Technical decisions

| Decision | Chosen approach | Rejected alternative | Trade-off |
|----------|-----------------|----------------------|-----------|
| Check location | Final scheduler admission at the production prefill call site | Startup estimate or HTTP-only heuristic | Accurate request geometry and resident state, but requires an HTTP-to-worker handshake |
| Cache credit | Charge cold KV and retained active allocations | Assume a prompt snapshot hit reduces the request | Predictable safety with possible false rejection and lost cache-hit latency |
| Override | Lower-only cap over known runtime limits | Let an environment variable raise or create a budget | Operators can test stricter policy without bypassing device limits |
| Unknown values | Fail closed | Admit when geometry, budget, or arithmetic is unknown | Safer failure mode, but configuration or unsupported geometry can reject service |
| Runtime LoRA | Reject unfused adapters at construction | Guess arbitrary-rank workspace | Clear startup failure until the workspace can be modeled |

## 4. Verification

- Request-memory suites: 26 tests passed, covering budget boundaries, output growth, checked overflow, invalid overrides, cache accounting, context precedence, cancellation, zero-forward rejection, unsupported paths, batched-prefill bypass, legacy serving, and pre-SSE errors.
- Nemotron-H suite: 23 tests passed and 3 existing tests were ignored, including runtime-LoRA rejection and staging cleanup.
- Focused provider and native-route binaries passed 30 and 34 tests respectively; these overlap the 26 request-memory tests and are not added as unique cases. They cover all five generation dialects and the provider admission lifecycle.
- `cargo check --tests --features metal,accelerate` passed in 51.50 seconds, and `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` passed in 5 minutes 5 seconds. Known C++ unused-capture warnings did not fail the Rust Clippy gate.
- `cargo build --release --features metal,accelerate` passed on the measured source in 9 minutes 42 seconds and again after the in-place-setter correction in 8 minutes 59 seconds.
- Reviewed-source real-checkpoint lifecycle passed on three normal fresh servers, one `--no-warmup` server, and one legacy `--no-batch` server. Post-setter normal and legacy lifecycle smokes repeated the rejection/recovery behavior. A Qwen 2.5 7B second-family smoke remained admitted for streaming and non-streaming requests before and after the setter correction.
- GitHub's two-host logical pipeline job reported a stack overflow from one added by-value `BatchScheduler` builder frame in an unoptimized worker. Commit `9d2a1584` replaced it with an in-place setter without changing admission behavior, and the 26 targeted tests passed. The formerly failing logical job then passed on the corrected commit; the full workflow and corrected release build still had pending jobs and were not reported as fully green.

## 5. Benchmark evidence

The checked-in benchmark record is `docs/benchmark_results/nemotron-h-request-memory-m1ultra-2026-10-13.md`, with machine-readable artifacts under its matching `data/` directory. Initial calibration on an M1 Ultra 128 GB admitted a 6,001-token request under 32 GiB with a 27.34 GiB estimate and a measured 17.8 GiB process peak; the harness correctly refused to present that run as rejection evidence. Intermediate lower-cap runs are retained as calibration but predated the final routing and cache review fixes.

The final reviewed binary (`1c4673addeb123b8d3ef2758f69dc0c54811d1d061911dbcb86aa486cc6065c5`) ran three normal fresh-server repetitions. All 12 long streaming and non-streaming requests returned typed 503 before SSE, all 12 immediately following short requests and all 3 medium requests returned 200, all 3 beyond-context requests retained 400 precedence, and health remained 200. The medium estimate was 22.328 GiB in every repeat with a 17.6 GiB process physical peak; long estimates ranged from 27.337 to 27.351 GiB, short estimates from 17.813 to 17.865 GiB, and the long-rejection process peak was 16.9 GiB. One-minute host load was 6.44, 6.90, and 11.01, so this is lifecycle and safety evidence rather than a latency claim.

One no-warmup server repeated 4 long 503 responses, 4 following short 200 responses, a medium 200, a context 400, and healthy status. One legacy `--no-batch` server repeated 4 long 503 responses, 4 following short 200 responses, and a medium 200; its unchunked largest forwards produced intentionally conservative 45.506–45.634 GiB estimates. A Qwen 2.5 7B server returned 200 for both streaming and non-streaming requests under the same budget and remained healthy, confirming the shared route handshake did not apply Nemotron-H rejection policy to the second family.

The post-setter release artifact (`e7a5cc9997a8c0eeceba6a69769353eed35395e906b270d2c62a5f437e2a7062`) repeated one normal and one legacy lifecycle: each produced 4 typed long-request 503 responses before SSE, 4 completed short-request 200 responses, a medium 200, and health 200; the normal context check stayed 400. Estimates matched the measured artifact ranges. The Qwen control again returned 200 for streaming and non-streaming requests with health 200. These post-setter artifacts and the benchmark record were committed in `62c1cf03`.

## 6. Change summary

| Category | Summary |
|----------|---------|
| Safety | Adds a checked, fail-closed request envelope and lower-only runtime budget |
| HTTP | Returns typed 503 responses before SSE headers across supported generation routes |
| Scheduler | Serializes guarded admission at the final width-one prefill boundary and blocks batched-prefill bypass |
| Model loading | Materializes Metal Nemotron-H weights and rejects unmodeled runtime LoRA cleanly |
| Cache behavior | Disables pre-admission snapshot restore/donation and background warmup for guarded requests |
| Tests and docs | Adds estimator, scheduler, provider, route, and model regression tests plus operator documentation and a benchmark harness |

## 7. Follow-up and monitoring

- Confirm the corrected release build and remaining CI jobs before merge; do not claim a fully green workflow without evidence.
- Monitor false-rejection rates, estimate-to-observed-peak ratios, retained prompt-cache contribution, and post-admission allocation failures.
- A future policy may model additional families, runtime LoRA ranks, multi-slot reservations, distributed paths, or optimistic cache reuse, but none is promised by this PR.

## References

- Issue #2291: reject over-budget Nemotron-H requests before Metal prefill.
- PR #2290 and issue #2281: bound Nemotron-H SSD prefill scratch and establish the measured long-prompt baseline.
- `docs/request-memory-admission.md`: operator contract, limitations, and error shape.
