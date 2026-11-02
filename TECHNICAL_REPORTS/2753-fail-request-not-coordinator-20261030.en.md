# PR #2753: Fail one request, not the coordinator, on remote pipeline stage errors

**Date**: 2026-10-30 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 8c1a9af3, with origin/main (99b3e563) merged in at c76b0ce0. The local gate in the PR body passed on GB10 (spark-102) at 8c1a9af3. At the time of writing, a second local gate was running on the merged head and hosted CI was partly pending (`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `WebUI installed artifact`)
**Languages**: Rust, Markdown
**Risk Level**: Low (no wire change; error text for two failure classes changes; `LanguageModel` gains three provided methods and `EngineError` two variants, so an exhaustive `match` on `EngineError` outside the workspace would need new arms)

## Executive Summary

PR #2753 (closes #2727) removes a coordinator-wide abort from the remote pipeline. When a remote stage failed a forward (an `Error` reply, a transport error, or a `stage_timeout`), `PipelineServerModel` panicked on the model worker thread, which runs under `run_core_thread_or_abort`, so the whole coordinator process died with every in-flight request. A timeout needs no cluster secret: an on-path party that drops or corrupts one frame is enough. The PR adds fallible forward hooks to `LanguageModel`, routes them through the engine as `EngineError::Forward`, and has the scheduler fail only the affected rows with the fixed text `model pipeline forward failed; see the server log for the stage error`. A refused `PrepareSequence`, which used to reach the client as `Server busy: open: ... remote stage <addr> failed request: ...`, now arrives as `model pipeline could not prepare this request; see the server log for the stage error`, without the capacity label or the stage address. Review added two runtime fixes on the same failure path: cancel and release now reach every stage even past one that does not answer, and a batched decode whose rows return different logits shapes fails the step instead of aborting inside MLX `concatenate`.

## 1. Problem Statement

#2727 collected two residuals of #2723 (issue #2714), verified against main 59c07652, and asked for both in one PR because they share the engine error path and the client-safe message. It was scheduled after the 0.8.0 release at LOW priority, with the note that item 1 is reachable without the secret.

### 1.1 A remote forward error aborted the coordinator (security, DoS)

The `LanguageModel` forward API was infallible: `forward_with_sequence_id` and `forward_batched_with_ids` return an `MlxArray`, not a `Result`. `PipelineServerModel` implements them over `RemotePipelineRuntime::forward_sequence` and `forward_batched`, which do return `Result`, and turned every `Err` into `panic!("pipeline forward failed for sequence ...")` (`server_runtime.rs:169`). The model worker thread runs under `run_core_thread_or_abort`, whose purpose is to make a panic mean "broken invariant, abort the process". So one failing stage took down the coordinator and all of its requests.

The runtime returns `Err` for a stage `Error` reply, a transport error and a `stage_timeout` expiry. The last one is the reason this is a security issue and not only a robustness one: a frame that fails AEAD authentication closes the connection, the exchange then waits until `stage_timeout` (30 s in `startup.rs`), and the coordinator aborts. An on-path party without the cluster secret can trigger that at will.

### 1.2 A refused prepare was labelled "Server busy" and leaked the stage address (security, info disclosure)

#2714 made a refused `PrepareSequence` fail one open instead of aborting. The message, however, was assembled from every layer it crossed: `broadcast_ack` produced `remote stage {addr} failed request: {message}`, `Engine::open` wrapped it as `EngineError::Open("prepare sequence state for {id}: ...")`, `allocate_sequence_state` flattened it to a `String`, and admission prefixed `Server busy: `. The client therefore saw an internal stage address and a remote `pp-seq-` id under a capacity label for a cause that has nothing to do with capacity.

### 1.3 Two further aborts on the same path (found during review)

- **Teardown stopped at the first silent stage.** `cleanup_request` sent `CancelRequest` and `ReleaseSequence` through `broadcast_ack`, which returns on the first error. A stage that timed out kept both commands from every stage after it. Remote stages have no reaper, so each such failure left the sequence, its KV caches and one of the 4096 prepared-sequence slots (#2746) on the later stages permanently. Enough failures would fill the cap and refuse every later request.
- **Mixed-shape batched logits aborted in MLX.** `forward_batched` concatenated per-row logits without comparing shapes. `validate_logits_shape` lets each row match only its own input (sequence axis of the input length or 1) and, without a configured vocabulary size, any width. Rows of different widths made MLX throw inside an FFI call that cannot report it, and the process aborted. Once 1.1 was fixed, this became the next way a stage could kill the coordinator through a forward.

## 2. Change Summary

| Area | Files | Change |
|---|---|---|
| Model trait | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `src/loaded_model.rs` | Three provided methods: `try_forward_with_sequence_id`, `try_forward_last_logits_with_sequence_id`, `try_forward_batched_with_ids`. Defaults wrap the infallible forwards in `Ok`, so no model file changes. `&M` and `LoadedModel` delegate them explicitly |
| Pipeline model | `pipeline/server_runtime.rs` | `PIPELINE_FORWARD_FAILED` and `PIPELINE_PREPARE_REFUSED` constants. `PipelineServerModel` overrides the three try hooks and `try_prepare_sequence_state`: on `Err` it logs `{err:#}` with the sequence id(s) at `warn` and returns the fixed text. The infallible forwards keep their panics for callers with no error path |
| Engine | `engine/mod.rs`, `prefill.rs`, `speculative.rs` | `EngineError::Forward(String)` (`forward: {0}`) and `EngineError::Prepare(String)` (`{0}`). `step`/`submit` (via `forward_rows`), the text branches of `prefill` and `score`, `prefill_cohort` and `verify` call the try hooks. `Engine::open` returns `Prepare` for a refused prepare and keeps `Open` for pool capacity |
| Scheduler | `decode_tick.rs`, `planned_prefill.rs`, `prefill.rs`, `prompt_lookup.rs`, `paged_layout.rs`, `admission.rs`, `queued_adoption.rs` | A failed decode step fails every row with the message verbatim and forces `Finished(Error)` (`fail_forward_rows`). A failed prefill piece becomes `PieceFailure::Abort(msg)` (`prefill_piece_failure`). A failed cohort prefill fails its rows instead of falling back to per-row prefill. A failed verify fails its request. `allocate_sequence_state` returns `EngineError`, and `allocation_failure_message` passes `Prepare` through while capacity keeps `Server busy: ...` |
| Runtime | `pipeline/runtime.rs` | `broadcast_best_effort` sends cancel and release to every stage, logging a stage that fails to acknowledge. Prepare rollback and `release_sequence_state_by_id` use it too. `forward_batched` refuses a row whose logits shape differs from the first row's |
| Tests | `stub_stage.rs` (new), `server_runtime_tests.rs` (new), `scheduler_pipeline_failure_tests.rs` (new), `engine/tests.rs`, `runtime.rs`, `paged_layout.rs`, `language_model_ref.rs` | The scripted stage moves to a shared `cfg(test)` module and learns to serve logits and fault one chosen forward. 11 new tests, 4 moved |
| Docs | `docs/distributed.md` | "A stage failure fails one request, not the coordinator": the two messages, the teardown, the mixed-shape refusal |

### Statistics

| Item | Value |
|---|---|
| Files changed | 22 |
| Lines | +1841 / -489 |
| New tests | 11 (4 scheduler, 2 runtime, 2 pipeline model, 2 allocation message, 1 engine), plus 4 moved and 1 extended |
| Commits | 5 plus a merge of main |

| Commit | Purpose |
|---|---|
| 26c21722 | Failing-first scheduler tests over a real `RemotePipelineRuntime` inside `run_core_thread_or_abort` |
| ba98789a | The fix: try hooks, engine errors, scheduler handling, client-safe messages |
| 7ca6a515 | Move the failure tests from fixed ports 20300-20307 to ephemeral `127.0.0.1:0` |
| 53cd14bd | Best-effort teardown on every stage; refuse mixed-shape batched logits |
| 8c1a9af3 | `allocation_failure_message` unit tests; docs for the teardown and mixed-shape behavior |
| c76b0ce0 | Merge of origin/main |

## 3. Technical Decisions

### 3.1 Fallible hooks beside the infallible API, not a changed signature

**Context.** The forward signatures are implemented by every model in the repository. Only one implementation, the remote pipeline coordinator, computes somewhere else and can fail there.

| Option | Assessment |
|---|---|
| Catch the panic around the forward | Rejected in the issue: the fail-fast wrapper exists so a panic means a broken invariant; catching it would also swallow real bugs and leave MLX state unknown |
| Return sentinel logits plus a side-channel error | Untyped and easy to miss at one of the many call sites |
| Change every forward to return `Result` | Touches every model file for a failure only one model can produce |
| **Chosen: provided `try_*` methods whose defaults wrap the infallible ones** | Zero model changes; the engine calls the try form everywhere it has an error path; only `PipelineServerModel` overrides them |

This mirrors `try_prepare_sequence_state`, which #2714 added for the same reason. The contract written on `try_forward_with_sequence_id` is that an `Err` means the sequence's state did not advance (or was discarded with the request), so the caller has nothing to unwind. That is what lets `EngineError::Forward` skip the `unwind_appends` path that `EngineError::Eval` needs.

**Trade-off.** Wrappers must forward the try hooks explicitly. `LoadedModel` is an enum wrapper; had it relied on the trait default, the default would have wrapped `LoadedModel`'s own infallible forward and the pipeline override would never run. The PR adds the explicit delegation and a comment explaining why; any future wrapper type has the same obligation and nothing enforces it.

### 3.2 A fixed client-safe message, with the detail in the log

**Context.** The stage's error is useful to an operator and harmful to a client: it names stage addresses, the remote `pp-seq-{n}-{hex}` id (whose suffix #2746 made secret precisely so it cannot be pre-claimed) and stage-side text.

`PipelineServerModel` is the last point that sees both. It logs `{err:#}` at `warn` with the sequence id and returns a constant. Every layer above passes the constant through unchanged: `EngineError::Prepare` displays as the bare message, `allocation_failure_message` returns it without a prefix, and `fail_sequence_with_message` sends it verbatim. The scheduler tests assert equality with the constant, and `assert_client_safe` additionally rejects any text containing the stage address, `127.0.0.1`, `pp-seq-`, `stub stage` or `Server busy`.

`Server busy: ...` stays for real capacity failures (pool slots, the paged budget), so clients and load balancers that treat it as retryable keep doing so for the cases where retrying helps.

### 3.3 Fail the rows; never retry a failed forward

A failed forward is not retried at the scheduler level. A decode step fails all its rows; a cohort prefill fails all its rows instead of taking the existing per-row fallback; a verify fails its request. Retrying would repeat the same remote exchange against the same stage, and for a timeout each attempt costs another `stage_timeout` on the single model worker thread. Before the forward returns `Err`, the runtime has already sent cancel and release for the sequence, so a retry would also run against a sequence the stages no longer hold.

The failed rows are forced to `Finished(Error)`, which overrides any finish recorded earlier in the same tick. That keeps the prompt cache from receiving a donation of state the failed forward never wrote, the same rule `fail_desynchronized_sequence` already applied.

One path does repeat a forward once: if the decode lookahead prime (`submit`) gets `Forward`, it returns `None` and the tick falls back to the synchronous step, which runs the forward again and then fails the rows. This keeps the prime's existing "fall back on any failure" shape; it is bounded to one extra exchange per tick.

### 3.4 Best-effort teardown on every stage

**Context.** Teardown commands carry no result anyone acts on, but they are the only thing that frees stage-side state.

`broadcast_ack` is right for commands whose failure must stop the caller (prepare, shutdown). For cancel and release, stopping at the first failure is exactly wrong: the stage that failed is the one least likely to need the command, and the stages after it are the ones that leak. `broadcast_best_effort` sends to each stage in order and logs per stage (an `Error` reply at `debug`, since a stage that never prepared the id answers that way; a missing acknowledgement at `warn`). `cleanup_request`, the prepare rollback and `release_sequence_state_by_id` all use it.

**Trade-off.** The sends are still sequential. A stage that stops answering costs one `stage_timeout` per teardown command addressed to it, on the model worker thread (section 5).

### 3.5 Check batched row shapes before `concatenate`

Logits validation is per row and, without a configured vocabulary size, cannot know the right width. The batch is the first place where rows can be compared, and MLX's own shape error at graph construction escapes through an FFI boundary that cannot carry it. Comparing each row's shape to the first row's before concatenating turns that into an ordinary `Err`, which then takes the 3.3 path and fails every row of the batch with `PIPELINE_FORWARD_FAILED`.

### 3.6 Last-logits on a stage that returns one position

The issue specified `try_forward_last_logits_with_sequence_id` as the full forward followed by `logits_at_position(last_pos)`. A stage may legitimately return only the last position (`[1, 1, vocab]`, accepted by the runtime's shape check), and indexing that at `last_pos > 0` would be out of range. The override uses position 0 when the returned sequence axis has length 1.

## 4. Verification

### 4.1 Failing-first evidence

26c21722 adds the scheduler tests on unmodified main. They build a `PipelineServerModel` over a real `RemotePipelineRuntime` (authenticated TCP, the real exchange, timeout and cleanup code) against a scripted stage, wrap it in `LoadedModel`, drive it through `BatchScheduler`, and run the whole test inside `run_core_thread_or_abort`. What the test process does is therefore what the coordinator process would do.

| Test | On main | Now |
|---|---|---|
| `remote_forward_error_fails_one_request_and_the_coordinator_serves_the_next` | panic at `server_runtime.rs:169:35`, binary exits 134 (SIGABRT) | First request gets exactly `PIPELINE_FORWARD_FAILED`; the second completes with `[3, 3]` |
| `remote_forward_timeout_fails_one_request_and_the_coordinator_serves_the_next` (300 ms timeout, stage silent on the decode forward) | same abort | same as above |
| `remote_prefill_failure_fails_one_request_and_the_coordinator_serves_the_next` | same abort | `PieceFailure::Abort(PIPELINE_FORWARD_FAILED)`, 0 tokens, next request completes |
| `refused_prepare_reaches_the_request_as_the_client_safe_message` | `Server busy: open: ... pp-seq-0-...: remote stage 127.0.0.1:20304 failed request: ...` | exactly `PIPELINE_PREPARE_REFUSED`; the stage sees `prepare, release` |
| `remote_runtime_teardown_reaches_every_stage_past_a_silent_one` (with the 53cd14bd fix disabled) | the stage after the silent one saw only `prepare` | it sees `prepare, cancel, release, release` |
| `remote_runtime_refuses_a_batch_whose_rows_disagree_on_logits_shape` (fix disabled) | SIGABRT in `concatenate` | `Err` containing `logits with shape` |

The forward tests also pin the stage's full command log, for example `prepare, run_entry, run_entry, cancel, release, release, prepare, run_entry, run_entry, release`, which shows the runtime tearing the failed sequence down before the next request starts.

### 4.2 Local gate (head 8c1a9af3)

Run on GB10 (spark-102) under `gpu-lock run --tag issue-2727`, `--profile test-fast`, `--test-threads=1`, with `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`. `cargo fmt --check` and clippy with `-D warnings` (mlxcel and mlxcel-core, lib and tests) were clean.

| Suite | Passed |
|---|---|
| mlxcel-core `engine::` and `language_model_ref::` | 73 |
| `distributed::` | 1557, 4 ignored |
| Scheduler modules touched (pipeline failure, allocation message, model-owned cache and lookahead, prompt lookup, prompt cache, VLM fallback, mrope) | 97 |
| `distributed_integration` (`cuda,test-utils`) | 20 |
| `pipeline_e2e` (`cuda,test-utils`) | 30 |
| `pipeline_remote_real_models` (`--include-ignored --skip glm_moe_dsa`) | 10 |
| `pipeline_server_remote_real_models` (same) | 7 |
| `pipeline_ci_multi_stage_real_models` (same) | 4 |
| `disaggregated_serving_e2e` (`--include-ignored`) | 2/2 |
| `disaggregated_router_e2e` (`--include-ignored`) | 7/7 |

### 4.3 Manual check

A `--pp-auto 2` coordinator and two `--pp-peer` processes served qwen3-0.6b on loopback; the last stage was killed with `kill -9` during a streamed chat request. The stream ended with `finish_reason: "error"` and `[DONE]`, the coordinator stayed up, and `/health` returned 200. The next request got `Generation error: model pipeline could not prepare this request; see the server log for the stage error`, because the stage was gone. No client text contained a stage address or a `pp-seq-` id.

### 4.4 After the merge of main

c76b0ce0 merges origin/main (99b3e563, which includes #2742 and #2746). The gate above predates it. At the time of writing a second local gate was running on the merged head and hosted CI was partly pending, so neither result is recorded here.

## 5. Remaining Gaps and Risks

- **Isolation is per request, not per latency.** The model worker is a single thread and every remote exchange is synchronous. A stage that stops answering costs one `stage_timeout` for the failed forward, then one each for cancel, release and the scheduler's later release, all addressed to that stage (the teardown test's silent stage receives `cancel, release, release`). With the 30 s default that can stall every other request on the coordinator for about two minutes per failure. The coordinator survives; its throughput does not. A shorter timeout for teardown commands, or skipping stages already known to be unreachable, would bound this.
- **Release is sent twice per failed request.** The runtime releases on a failed forward and keeps the id mapping, and the scheduler's close of the sequence releases again. Harmless against a healthy stage, but against a silent one it is one of the timeouts above.
- **The lookahead fallback repeats a forward once.** See 3.3. The pipeline failure tests pin command logs with no repeated `run_entry`, so this path is not exercised by them.
- **Streaming chat clients see no error text.** The chat SSE path ends the stream with `finish_reason: "error"` and does not carry the message; this is unchanged from main and outside this PR.
- **Wrapper types must delegate the try hooks.** A new `LanguageModel` wrapper that relies on the trait defaults silently restores the infallible path for the pipeline model. `LoadedModel` and `&M` are covered; the `&M` test checks the override is reached.
- **Not run.** `glm_moe_dsa` (`glm5-4bit` does not load on this 128 GB box), the Thunderbolt and RDMA transports, and macOS. No performance measurement: the success path gains one `Result` wrapper per forward and one shape comparison per batched row.

## 6. Learning Points

### 6.1 Add a fallible twin with a wrapping default

When one implementation of a widely implemented trait method gains a failure mode, a provided `try_*` method whose default wraps the infallible one localizes the change to that implementation and to the callers that can handle the error.

```rust
fn try_forward_with_sequence_id(
    &self,
    input_ids: &MlxArray,
    seq_id: Option<SequenceId>,
    caches: &mut [KVCache],
    mask: Option<&MlxArray>,
) -> Result<UniquePtr<MlxArray>, String> {
    Ok(self.forward_with_sequence_id(input_ids, seq_id, caches, mask))
}
```

The catch is delegation: every wrapper (`&M`, enum dispatchers) must forward the new method explicitly, or the default wraps the wrapper's infallible method and the override is never reached.

### 6.2 Choose the client message at the lowest layer that knows the cause

Error text that accumulates context at every layer ends up carrying whatever the lowest layer knew, including internal addresses and identifiers. Decide the client-visible text once, where the cause is known, log the detail there, and have every layer above pass it through without prefixes. Keep capacity labels for capacity, so a client's retry policy stays correct.

### 6.3 Teardown must not stop at the first failure

A broadcast that aborts on the first error is right for setup and wrong for cleanup. In cleanup, the failing peer is the one least likely to hold state, and the peers after it are the ones that leak. Send to all, log each failure, return nothing.

### 6.4 Test the process outcome, not only the return value

The scheduler tests run inside `run_core_thread_or_abort`, the same fail-fast wrapper as the real model worker. That makes "the coordinator aborts" a test outcome (exit 134) instead of something inferred from a returned panic payload, and it is why the mixed-shape `concatenate` abort, which no `Result` could have reported, was caught.

## 7. References

- Issue #2727: fail one request, not the coordinator, on remote pipeline errors
- PR #2723 (issue #2714): fallible `try_prepare_sequence_state` and per-sequence ownership; this PR closes two of its residuals
- PR #2746 (issue #2726): the 4096 prepared-sequence cap and CSPRNG `pp-seq-` suffixes that the teardown fix and the message redaction protect
- `src/worker_failfast.rs`: `run_core_thread_or_abort`
- `docs/distributed.md`, "What the secret protects"
