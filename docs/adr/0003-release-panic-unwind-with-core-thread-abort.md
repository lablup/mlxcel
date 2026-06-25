# ADR 0003: Release builds unwind, and core generation threads re-impose fail-fast with a targeted abort

**Status:** Accepted (2026-06-21). Resolves issue #375 (a synthesis panic aborts the server in release because `panic = "abort"` defeats the AudioWorker `catch_unwind` backstop added in PR #374). Names issue #382 as the documented residual abort vector (MLX C++ FFI exceptions become `std::terminate`, not a Rust panic).

## Context

PR #374 added the Kokoro text-to-speech provider, which runs the StyleTTS2 plus iSTFTNet forward pass on the shared single `AudioWorker` thread (`src/server/audio_worker.rs`) behind `POST /v1/audio/speech`. To keep one bad request from bricking all audio requests, that PR wrapped each engine call in a `catch_unwind` boundary (`run_guarded`): a synthesis panic becomes a per-request `Inference` error and the worker thread survives.

That backstop did not work in production. `Cargo.toml` set `[profile.release] panic = "abort"`, and the documented production build is `cargo build --release`. Under `panic = "abort"` a panic does not unwind, so `catch_unwind` never runs: in a release build a synthesis panic aborts the whole server process, not just the offending request. The audio forward path carries many `expect`/`unwrap` sites. Today's input guards (4096-char input cap, Kokoro-vocab-restricted g2p output, 510-token truncation, finite/positive `speed`, per-token frame counts clamped to [1,100], whitelisted voice names) make those largely unreachable, but the backstop is the safety net for a future regression that introduces a reachable panic, and that net was inert in release.

The audio worker `run_guarded` is the only deliberate `catch_unwind` containment in the server, and that release setting silenced it. The distributed pipeline stage has no production `catch_unwind` of its own (the only one under `src/distributed/` is a `#[cfg(test)]` assertion), so nothing there was being relied on either.

A subtlety that shapes the testing story: `cargo test` and `cargo bench` always build with `panic = "unwind"` regardless of the profile setting, because the test harness needs to unwind to report failures. So the `worker_survives_engine_panic_and_keeps_serving` test in `audio_worker.rs` passed even while the release binary would have aborted. The test asserted unwind behavior the shipped binary did not have.

## Options considered

### Option A (chosen): release `panic = "unwind"`, plus explicit core-thread fail-fast

Flip the release profile to `panic = "unwind"` so the deliberate audio worker `catch_unwind` request-isolation backstop works in production exactly as it does under `cargo test`. Then re-impose fail-fast where it is actually wanted, the core inference worker threads, with a targeted `catch_unwind` plus `std::process::abort()` wrapper (`run_core_thread_or_abort` in `src/worker_failfast.rs`). A panic in a core worker means a broken invariant, so the process aborts for a supervised restart into fresh state rather than unwinding and leaving the server alive but unable to generate.

The tradeoff `panic = "abort"` was chosen for is binary size and the removal of landing-pad code. For an inference server whose binary is dominated by the linked MLX C++ runtime and model weights, the unwinding-table overhead is negligible, and the decode hot path does not panic, so there is no measurable runtime cost. The correctness win (request and stage isolation that actually works in production) outweighs the size delta.

### Option B (rejected): keep `panic = "abort"`, make the synthesis path panic-free

Convert every attacker-reachable `expect`/`unwrap` in `src/models/kokoro/*` and `src/server/kokoro_tts.rs` to recoverable `Result` errors so no input can panic regardless of unwind policy. Rejected as disproportionate: it is a large, ongoing refactor of one model's forward path, it does not generalize (the next model added would reintroduce the same class of risk), and it leaves the broader `catch_unwind` containment (pipeline stages, any future request boundary) still inert in release. Option A fixes the category, not one instance.

### Option C (rejected): keep `panic = "abort"` and document the limitation only

Accept that the backstop is release-inert and rely solely on the input guards. Rejected because the guards are defense in depth, not the guarantee; the `catch_unwind` was added precisely so a future reachable panic degrades to a per-request error instead of a server-wide outage, and leaving it inert defeats the reason it exists.

## Decision

Set `[profile.release] panic = "unwind"` (the only profile that sets `panic`; no member crate overrides it). Add `run_core_thread_or_abort(label, body)`, a thin `catch_unwind` plus `process::abort()` wrapper that forwards the body's normal return value (a recoverable `Err` is not a panic and is propagated unchanged), in `src/worker_failfast.rs` so both the server and the distributed pipeline can reach it. Wrap every core inference worker thread body with it: the batched worker and the legacy sequential worker (the two `thread::spawn` sites in `src/server/model_worker.rs`), plus the remote pipeline stage service thread (`src/distributed/pipeline/remote_service.rs`), which runs core forward passes for a remote stage. The batch scheduler, the DiffusionGemma batch-1 loop, and the disaggregated serving-role loop all run inside the two server thread bodies, so those spawns plus the remote stage spawn cover every core generation path.

The audio worker `run_guarded` (`src/server/audio_worker.rs`) is the only boundary that deliberately contains panics: its own `catch_unwind` turns a synthesis or transcription panic into a per-request error, so it is not wrapped by `run_core_thread_or_abort`.

The distributed pipeline stage has no `catch_unwind` of its own. Its service thread runs core forward passes, so it now fails fast through `run_core_thread_or_abort` exactly like the local workers: an uncaught panic aborts the stage process for a supervised restart instead of unwinding into a zombie stage while `serve_remote_pipeline_stage` stays parked on `ctrl_c`. Pipeline-level fault handling lives at the coordinator, not in a stage `catch_unwind`: a stage error propagates as a `Result`, and a dead or failed stage is caught by stage timeout and health probing, then surfaced as a per-request error with the stage marked failed.

### The no-global-abort-hook trap

There is deliberately no `std::panic::set_hook` that aborts. The panic hook runs before unwinding, so a global abort hook would fire before any `catch_unwind` and kill the audio worker `run_guarded`, the exact backstop this change exists to restore. Fail-fast is imposed only by the targeted per-thread wrappers, never globally. The default panic hook (which prints the panic and backtrace) stays, so a caught panic is still logged on the way through.

`AssertUnwindSafe` inside `run_core_thread_or_abort` is correct precisely because the wrapper aborts and never continues on a caught panic, so no partially-torn state is ever observed by subsequent code.

## Consequences

- A synthesis-path panic in a release build now becomes a per-request error and the server keeps serving, matching the `worker_survives_engine_panic_and_keeps_serving` test. That test now represents release behavior, not just the always-unwind test profile.
- A panic on a core generation thread aborts the process cleanly (logged via `tracing` at `target: "mlxcel::worker"`, without request content) for a supervisor to restart, preserving the fail-fast posture `panic = "abort"` used to provide for those threads.
- The abort path of `run_core_thread_or_abort` cannot be unit-tested in-process (it terminates the test runner). It is covered by a happy-path unit test and verified manually in a release build; a cfg-gated subprocess re-exec test is possible but not worth the flakiness for a one-line abort.
- **Residual abort vector (#382) — resolved in PR #384; extended in PR #434 and PR #439.** An MLX C++ FFI exception thrown through a non-`Result` `cxx` op becomes `std::terminate`, not a Rust panic, so neither the `catch_unwind` backstops nor the unwind policy intercept it. PR #384 adds `try_matmul` and `try_array_to_raw_bytes` as `-> Result<..>` variants in the cxx bridge and routes the Kokoro alignment-expansion matmuls and the final PCM readback through them, so those throws become recoverable per-request `Err`s instead of `std::terminate`. PR #434 adds `try_conv2d` and `try_conv1d` by the same pattern, covering the Gemma 4 Conformer audio encoder's convolution path (issue #427). PR #439 applies the same `try_conv2d`/`try_conv1d` coverage to the Nemotron-H Nano Omni Conformer/Parakeet audio encoder (issue #435): all four data-dependent conv calls now return `Err` on shape faults instead of reaching `std::terminate`. Whisper relies on fixed-shape invariants so no fallible variants are needed there. `cxx` ops that throw a non-`std::exception` type still terminate (the cxx bridge only catches `std::exception`); no such throws exist on the current audio path.

## References

- Issue #375 (this decision), PR #374 (the Kokoro provider and the `run_guarded` backstop), issue #382 (the residual MLX FFI `std::terminate` vector, resolved by PR #384), PR #384 (`try_matmul` / `try_array_to_raw_bytes` fallible cxx variants, Kokoro path wiring), issue #427 / PR #434 (`try_conv2d` / `try_conv1d` fallible cxx variants, Gemma 4 Conformer audio encoder path), issue #435 / PR #439 (same fallible cxx variants extended to the Nemotron-H Nano Omni Conformer/Parakeet audio encoder).
- `Cargo.toml` `[profile.release]` `panic = "unwind"`.
- `src/worker_failfast.rs` (`run_core_thread_or_abort`, shared by the server and the distributed pipeline).
- `src/server/model_worker.rs` (the two wrapped core generation `thread::spawn` sites).
- `src/distributed/pipeline/remote_service.rs` (the wrapped remote pipeline stage service thread).
- `src/server/audio_worker.rs` (`run_guarded`, the only request-isolation boundary left contained).
- [ADR 0001](0001-paged-attention-gather-vs-fused-kernel.md) and [ADR 0002](0002-turbo-kv-split-dequant-vs-fused.md), the prior records in this series.
