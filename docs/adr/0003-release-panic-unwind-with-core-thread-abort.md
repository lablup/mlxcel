# ADR 0003: Release builds unwind, and core generation threads re-impose fail-fast with a targeted abort

**Status:** Accepted (2026-06-21). Resolves issue #375 (a synthesis panic aborts the server in release because `panic = "abort"` defeats the AudioWorker `catch_unwind` backstop added in PR #374). Names issue #382 as the documented residual abort vector (MLX C++ FFI exceptions become `std::terminate`, not a Rust panic).

## Context

PR #374 added the Kokoro text-to-speech provider, which runs the StyleTTS2 plus iSTFTNet forward pass on the shared single `AudioWorker` thread (`src/server/audio_worker.rs`) behind `POST /v1/audio/speech`. To keep one bad request from bricking all audio requests, that PR wrapped each engine call in a `catch_unwind` boundary (`run_guarded`): a synthesis panic becomes a per-request `Inference` error and the worker thread survives.

That backstop did not work in production. `Cargo.toml` set `[profile.release] panic = "abort"`, and the documented production build is `cargo build --release`. Under `panic = "abort"` a panic does not unwind, so `catch_unwind` never runs: in a release build a synthesis panic aborts the whole server process, not just the offending request. The audio forward path carries many `expect`/`unwrap` sites. Today's input guards (4096-char input cap, Kokoro-vocab-restricted g2p output, 510-token truncation, finite/positive `speed`, per-token frame counts clamped to [1,100], whitelisted voice names) make those largely unreachable, but the backstop is the safety net for a future regression that introduces a reachable panic, and that net was inert in release.

The same release setting silenced every other deliberate `catch_unwind` containment, including the pipeline stage isolation boundaries.

A subtlety that shapes the testing story: `cargo test` and `cargo bench` always build with `panic = "unwind"` regardless of the profile setting, because the test harness needs to unwind to report failures. So the `worker_survives_engine_panic_and_keeps_serving` test in `audio_worker.rs` passed even while the release binary would have aborted. The test asserted unwind behavior the shipped binary did not have.

## Options considered

### Option A (chosen): release `panic = "unwind"`, plus explicit core-thread fail-fast

Flip the release profile to `panic = "unwind"` so the deliberate `catch_unwind` request/stage backstops work in production exactly as they do under `cargo test`. Then re-impose fail-fast where it is actually wanted, the core generation worker threads, with a targeted `catch_unwind` plus `std::process::abort()` wrapper (`run_core_thread_or_abort` in `src/server/model_worker.rs`). A panic in a core worker means a broken invariant, so the process aborts for a supervised restart into fresh state rather than unwinding and leaving the server alive but unable to generate.

The tradeoff `panic = "abort"` was chosen for is binary size and the removal of landing-pad code. For an inference server whose binary is dominated by the linked MLX C++ runtime and model weights, the unwinding-table overhead is negligible, and the decode hot path does not panic, so there is no measurable runtime cost. The correctness win (request and stage isolation that actually works in production) outweighs the size delta.

### Option B (rejected): keep `panic = "abort"`, make the synthesis path panic-free

Convert every attacker-reachable `expect`/`unwrap` in `src/models/kokoro/*` and `src/server/kokoro_tts.rs` to recoverable `Result` errors so no input can panic regardless of unwind policy. Rejected as disproportionate: it is a large, ongoing refactor of one model's forward path, it does not generalize (the next model added would reintroduce the same class of risk), and it leaves the broader `catch_unwind` containment (pipeline stages, any future request boundary) still inert in release. Option A fixes the category, not one instance.

### Option C (rejected): keep `panic = "abort"` and document the limitation only

Accept that the backstop is release-inert and rely solely on the input guards. Rejected because the guards are defense in depth, not the guarantee; the `catch_unwind` was added precisely so a future reachable panic degrades to a per-request error instead of a server-wide outage, and leaving it inert defeats the reason it exists.

## Decision

Set `[profile.release] panic = "unwind"` (the only profile that sets `panic`; no member crate overrides it). Add `run_core_thread_or_abort(label, body)`, a thin `catch_unwind` plus `process::abort()` wrapper, and wrap the two core generation worker thread bodies with it: the batched worker and the legacy sequential worker, both `thread::spawn` sites in `src/server/model_worker.rs`. The batch scheduler, the DiffusionGemma batch-1 loop, and the disaggregated serving-role loop all run inside those two thread bodies, so wrapping the two spawns covers every core generation path.

Boundaries that deliberately contain panics are not wrapped and keep their own `catch_unwind`:

- the audio worker `run_guarded` (`src/server/audio_worker.rs`), which turns a synthesis or transcription panic into a per-request error, and
- the pipeline stage isolation (`src/distributed/pipeline/`), where a stage fault is contained to its request or connection.

### The no-global-abort-hook trap

There is deliberately no `std::panic::set_hook` that aborts. The panic hook runs before unwinding, so a global abort hook would fire before any `catch_unwind` and kill the audio worker `run_guarded` and the pipeline stage containment, the exact backstops this change exists to restore. Fail-fast is imposed only by the targeted per-thread wrappers, never globally. The default panic hook (which prints the panic and backtrace) stays, so a caught panic is still logged on the way through.

`AssertUnwindSafe` inside `run_core_thread_or_abort` is correct precisely because the wrapper aborts and never continues on a caught panic, so no partially-torn state is ever observed by subsequent code.

## Consequences

- A synthesis-path panic in a release build now becomes a per-request error and the server keeps serving, matching the `worker_survives_engine_panic_and_keeps_serving` test. That test now represents release behavior, not just the always-unwind test profile.
- A panic on a core generation thread aborts the process cleanly (logged via `tracing` at `target: "mlxcel::worker"`, without request content) for a supervisor to restart, preserving the fail-fast posture `panic = "abort"` used to provide for those threads.
- The abort path of `run_core_thread_or_abort` cannot be unit-tested in-process (it terminates the test runner). It is covered by a happy-path unit test and verified manually in a release build; a cfg-gated subprocess re-exec test is possible but not worth the flakiness for a one-line abort.
- **Residual abort vector (#382).** An MLX C++ FFI exception thrown through a non-`Result` `cxx` op becomes `std::terminate`, not a Rust panic, so neither the `catch_unwind` backstops nor the unwind policy intercept it: it still terminates the process. This change does nothing for that path; it is tracked separately as issue #382.

## References

- Issue #375 (this decision), PR #374 (the Kokoro provider and the `run_guarded` backstop), issue #382 (the residual MLX FFI `std::terminate` vector).
- `Cargo.toml` `[profile.release]` `panic = "unwind"`.
- `src/server/model_worker.rs` (`run_core_thread_or_abort` and the two wrapped `thread::spawn` sites).
- `src/server/audio_worker.rs` (`run_guarded`, the request-isolation boundary left contained).
- `src/distributed/pipeline/` (the stage isolation boundaries left contained).
- [ADR 0001](0001-paged-attention-gather-vs-fused-kernel.md) and [ADR 0002](0002-turbo-kv-split-dequant-vs-fused.md), the prior records in this series.
