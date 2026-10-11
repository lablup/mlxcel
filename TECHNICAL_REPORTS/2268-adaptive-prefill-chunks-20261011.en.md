# PR #2268: Adaptive Server Prefill Chunks

**Date**: 2026-10-11
**Status**: Complete; performance acceptance and full integration validation passed
**Languages**: Rust, Markdown
**Risk**: Medium (changes live batch scheduling, prefill partitioning, and CLI precedence)

## Executive Summary

PR #2268 changes server prefill chunking from one fixed value to a scheduler-state policy. A request uses 2048-token chunks while no other sequence is decoding and 512-token chunks while the active batch contains decode work. Explicit native, llama-compatible, or environment configuration pins both states to the requested value.

The implementation also makes prefill plans resumable from their live cursor. This is required because a parked request can start under contention at a 512-token boundary and resume after the active batch drains, when a newly built 2048-token partition would otherwise have no piece at that cursor. Focused CUDA tests cover the actual forward lengths and completion behavior. Fresh-release quiet-host admission and TTFT acceptance passed on GB10, and the full integrated workspace and CUDA gates passed after the two preceding PRs merged.

## 1. Problem Statement

PR #2205 selected a fixed 2048-token prefill chunk from single-stream TTFT measurements. PR #2226 then measured a conflicting concurrent-serving result: with four streams decoding, admitting an 8192-token request raised live-stream ITL p95 to 127.9–128.9 ms at chunk 2048, while chunk 512 held it to 55.8–56.4 ms. The smaller chunk increased the admitted request's TTFT from 1903–2060 ms to 3768–4067 ms.

Neither fixed value serves both states well. Keeping 2048 harms active streams during admission; keeping 512 pays the longer TTFT even when no stream needs protection. Selecting a value once at admission is also insufficient because decode work can finish while a long prompt is parked between pieces.

The existing plan constructor made a live switch unsafe. Its piece boundaries were derived from the original adopted or history boundary, so a cursor reached on the 512 grid was not necessarily a valid start in a freshly constructed 2048 plan. The scheduler treated the missing piece as an abort condition.

## 2. Technical Decisions

### 2.1 Select the chunk at every scheduler tick

PrefillChunkPolicy holds alone and contended values and resolves the current chunk from whether active_batch is empty. The defaults are 2048 and 512. The parked prefill itself is kept separately from active_batch, so it does not incorrectly classify its own continuation as contended.

An explicit --prefill-chunk-size, its --batch-size / LLAMA_ARG_BATCH alias, or MLXCEL_PREFILL_CHUNK produces {n, n}. This preserves operator control and makes an explicit value equal to the default distinguishable from no flag. Native flag precedence remains above the compatibility alias, which remains above the environment default.

Fixing the policy per sequence was rejected because a sequence admitted next to a stream would stay on 512 after that stream completed. Switching only new requests was rejected for the same reason.

### 2.2 Resume from the live cursor

PrefillPlan::resume_at retains the metadata that a full plan would report, but cuts continuation pieces from the current prefill_offset. The first continuation therefore starts exactly at the cursor regardless of the prior chunk size. Padding, history-boundary behavior, prompt-cache metadata, and model capability checks share the same helpers as the original constructor.

For a 5000-token request, the tested state transition is:

    active decode present: 0..512
    active batch drained:  512..2560, 2560..4608, 4608..5000
    forward lengths:       [512, 2048, 2048, 392]

The continuation reports chunked progress even when its remaining suffix fits in one piece. This preserves reservation and progress accounting after an earlier piece established the chunked-prefill lifecycle.

### 2.3 Preserve alone-value behavior outside the batch scheduler

Direct engine generation, the engine benchmark, and speculative/MTP prefill paths continue to use the 2048-token alone value unless explicitly overridden. The existing batched-prefill token budget is also derived from the alone value; its per-row cap already limits each row to 512 tokens.

## 3. Implementation Details

The core planner now defines the two-value policy, a process-wide parsed environment override, shared metadata and piece construction, and cursor-resumable planning. An allocation-free next-piece accessor reuses the same chunk eligibility, boundary, range, and padding helpers for capacity checks that only need the next padded length. The scheduler stores the policy, reads live decode state before each start or continuation, builds the plan at the live cursor for execution, reserves the blocks for the piece it will actually execute, and records the selected chunk in its tracing span.

CLI fields changed from a defaulted usize to Option<usize>, allowing explicit 2048 to remain explicit. Both server binaries use the same precedence and conflict rules. Startup, runtime settings, probes, server configuration, model workers, and scheduler builders carry both policy values. Startup diagnostics report both values.

Documentation updates explain the adaptive default, override semantics, cursor repartitioning, prompt-cache partition implications, and the PR #2226 tradeoff that motivated the policy. No dependency, persistence schema, authentication, or network protocol changed.

## 4. Technical Review

### Correctness and compatibility

- Chunk 0 still disables chunking in both policy states.
- Prompts shorter than the selected chunk remain one forward.
- Embedding-input/VLM prefills and unsplittable history-boundary segments keep their existing behavior.
- Explicit native, compatibility-alias, and environment values pin both states.
- Existing direct and speculative call sites retain the alone value.
- No new dependencies or public response fields were added.

The highest-risk path is a chunk-size transition after a request has parked. The integration regression executes real scheduler prefill calls against the tiny test model, records lengths immediately before Engine::prefill, consumes every continuation, rejects error or timeout events, and requires the request to emit Done.

### Performance

The policy is intended to recover PR #2226's 512-token admission behavior while retaining 2048-token single-stream TTFT. Execution planning is linear in the pieces it will run and no longer creates and discards a full-prefix vector before rebuilding a suffix. Repeated paged-block capacity checks compute only the next piece in constant space instead of allocating the complete remaining suffix.

Fresh release measurements passed every acceptance gate:

| Gate | Acceptance | Result |
|---|---|---|
| Live-stream admission, 3 quiet-host rounds | ITL p95 at or below 62.0 ms in every round | **Passed:** 54.1, 54.8, and 55.2 ms |
| Qwen 3 1.7B server TTFT, 5 rounds | Policy delta inside explicit-2048 null range | **Passed:** policy median +0.01%; null range -4.73% to +3.26% |
| Llama 3.2 1B server TTFT, 5 rounds | Policy delta inside explicit-2048 null range | **Passed:** policy median -0.20%; null range -1.12% to -0.03% |

### Security

The change adds no external trust boundary. CLI and environment input use the existing positive-integer-or-zero parser and invalid environment input falls back to the adaptive defaults with a warning. Logs expose only numeric policy values.

## 5. Verification

Completed on the implementation worktree:

- cargo check --profile test-fast --features cuda --lib: passed.
- cargo fmt --all -- --check: passed.
- git diff --check: passed.
- CUDA mlxcel-core prefill-plan tests: 17 passed.
- CUDA server CLI tests: 148 passed.
- CUDA adaptive scheduler execution regression: passed.
- CUDA runtime-settings tests: 8 passed.
- CUDA props compatibility regression: passed.
- Independent security/performance review found that repeated paged-block capacity checks eagerly built every remaining plan piece. Commit 88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752 replaces that hot-path allocation with the O(1) accessor and adds equivalence coverage across boundaries, cursors, chunks, padding modes, model opt-outs, and embedding inputs.
- Workspace all-targets Clippy passed after the fix. Focused CUDA runtime checks passed: 17 prefill-plan tests, 20 block-reclaim tests, and the adaptive scheduler transition regression.
- The first full CUDA gate exposed a source-audit gap after capability-helper extraction. The audit now uses brace-counted interprocedural recognition, retains direct-guard detection, and includes a mutation fixture; its three tests pass.

The acceptance artifact used production commit `88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752` and measured clean worktree HEAD `b2394566a71fc1f6db83ad40b44380b9c129b57c`. The driver verified that only two test files differed. Raw records, server logs, the exact environment, a machine-readable summary, and the reproducible driver and analyzer are archived under `docs/benchmark_results/data/adaptive-prefill-chunk-gb10-2026-10-11/`.

The driver's default build commit is valid only on the unsquashed measured lineage. A replay from a squash-merged or newer clean checkout must rebuild both release binaries from that checkout and set `MLXCEL_BUILD_COMMIT="$(git rev-parse HEAD)"`; the benchmark report gives the complete command.

After PR #2270 and PR #2269 merged, the complete series passed workspace test compilation with `--no-run`, workspace all-targets Clippy, formatting, the kernel dtype-key, kernel port-dispatch, and llama-server compatibility contract suites, and the full single-threaded CUDA workspace run through `make verify-test-cuda`. Every CUDA test summary reported zero failures, including the BF16 and F32 SSM update parity tests. The validated pre-rebase tree `ab8ff77ade8d6b1b4fb494707320cebc95150c8e` remained identical after rebasing onto current `origin/main`; the final changes after that comparison are documentation and PR status only.

## 6. Change Summary

| Item | Value |
|---|---|
| Files changed | 47 |
| Lines added | 1879 |
| Lines deleted | 210 |
| Measured branch head | b2394566a71fc1f6db83ad40b44380b9c129b57c |

| Category | Summary |
|---|---|
| Core planning | Two-state policy, cached override parsing, resumable cursor-based plans |
| Scheduling | Live-state chunk selection, safe continuation/reservation, O(1) capacity checks |
| Configuration | Explicit-value detection and two-value propagation through both server front ends |
| Testing | Planner properties, CLI precedence, schema compatibility, actual scheduler forward sequences |
| Documentation | Adaptive defaults, override behavior, measurement rationale, cache partition note |

The initial implementation commit is 7c43726a0fb792594a69de0fda1e31e78eef4fa1. Commit 7b2d17241ea9c16612766ccf37524cbe8be9f913 satisfies the CI Clippy expression policy. Commit 88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752 removes the review-discovered reservation allocation, and commit 808bcccf387957a6da31b961666aadd5bf7cbe65 adds the test-only type annotation required by all-targets Clippy. Commits 6f07f7563e74c5466b5aace6f3e8ee98ec8347e2 and b2394566a71fc1f6db83ad40b44380b9c129b57c conservatively update and strengthen the source audit for the shared capability guard.

## 7. Completion

No implementation, acceptance, or integration-validation work remains for issue #2228.

## References

- Issue #2228: choose the prefill chunk by whether other sequences are decoding.
- PR #2226 and docs/benchmark_results/unified-engine-final-gb10-2026-10-08.md: motivating admission measurements.
- ADR 0007: unified batch/native engine decisions and the amended prefill-chunk policy.
- docs/CONTINUOUS_BATCHING.md: server policy, planning invariants, and operational controls.
- docs/benchmark_results/adaptive-prefill-chunk-gb10-2026-10-11.md: passing acceptance results, raw data, and reproduction protocol.
