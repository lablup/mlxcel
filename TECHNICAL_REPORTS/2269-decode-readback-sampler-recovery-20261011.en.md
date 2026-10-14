# Technical Report: PR #2269 — Contain Decode Readback Failures and Preserve Sampler State

**Date**: 2026-10-11

**Status**: Implementation and required local validation completed. CI remains pending at report preparation.

**Languages**: Rust, C++

**Risk Level**: High. Changes cross the lazy GPU graph, CXX error boundary, speculative KV teardown, and mutable sampler/RNG state.

## Executive Summary

Decode host reads now wait through fallible CXX entrypoints so backend evaluation errors can fail the affected request instead of escaping an infallible bridge. Failed lookahead submissions commit their already-drawn pending token before synchronous fallback, while failed row draws restore mirostat and adaptive-p feedback.

Seeded fused parity additionally requires restoring the RNG key consumed while constructing the failed next submission. A snapshot of the pinned MLX thread-local key sequence supplies that rollback without adding a GPU evaluation or scheduling call. Focused fault tests, the broader suites and the newly strengthened stopping-boundary mutation checks passed; the restored implementation passed all 13 fault regressions again.

## 1. Problem Statement

An asynchronous backend error may first surface when the host waits for a token, selected probability, or sorted sampler distribution. Several decode paths reached non-`Result` CXX reads before a fallible wait, allowing a C++ exception to terminate the process. Guarding only the final synchronous token read did not protect mirostat or adaptive-p, which read earlier inside sampling.

A second fault was state advancement after speculative failure. Dropping a pending token and sampling it again could consume another random key or advance feedback twice. The emitted stream could diverge from force-sync execution even though the request appeared to recover. Retiring the wrong number of speculative appends could also leave the pool offset and KV length inconsistent with the committed prefix.

## 2. Technical Review

The change confines evaluation errors to existing request-level error and teardown paths. Scheduler read failures record an evaluation failure and enter the established finishing teardown before retrying through guarded synchronous execution. This scheduler branch is reviewed in code, as the core test-only fault seam is not available to the server crate.

Token waits use per-array `try_eval`, preserving overlap with the next in-flight forward. Feedback snapshots copy only mirostat/adaptive-p state, not penalty history caches. The fused stochastic path adds a host-side snapshot allocation and key-array ownership copy; it adds no evaluation or scheduling operation. No successful-path throughput result is claimed in this draft.

Existing infallible public sampler entrypoints retain their signatures and wrap the fallible core with the documented panic message. Engine sampling uses the new fallible entrypoints. There are no new external dependencies or data migrations. CUDA is the locally validated backend; Metal and ROCm runtime validation is not claimed.

## 3. Technical Decisions

### Guard the actual host-read boundary

`try_tokens_to_host` and `try_read_token` first perform the fallible per-array wait. Adaptive-p and mirostat guard every required evaluation/copy and return errors before publishing feedback updates. `DrawError::Mask` retains structured-output classification, while `DrawError::Eval` becomes an evaluation error.

Eagerly evaluating all logits before sampling was rejected because it would serialize penalty and DRY draws behind the forward. It also would not cover every synchronous sampler entrypoint.

### Commit pending state before fallback

Both B=1 pipelines share their normal read-and-commit helper with the failed-submit branch. That branch unwinds only the failed submission's append, commits `pending` with no `next`, and enters synchronous fallback only if generation continues. EOS and callback stops therefore use the same stopping and cache-retirement rules as normal iterations. A readback failure retires the pending append plus any existing next append.

### Restore only the failed fused draw's RNG advancement

Fused graph construction consumes a random key before `schedule`. Moving the injected fault earlier would conceal the real ordering defect; adding another schedule would risk successful-path throughput. The selected implementation snapshots `random::KeySequence::default_()` after forward construction and restores it only when the existing submission schedule fails. Successful submissions keep their key advancement.

This relies on the public copy construction/assignment of `KeySequence` in MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`. The CXX snapshot owns its copied key array; its lifetime stays inside one synchronous call on the owning thread. A regression verifies restoration and thread isolation. Per-row failed draws retain the documented limitation that a discarded stochastic draw can shift the seeded stream by its consumed key.

## 4. Implementation Details

| Area | Result |
|---|---|
| Fused, per-row, plain-drafter and scheduler token reads | Backend read errors follow fallible request-error paths |
| Mirostat v1/v2/greedy and adaptive-p | All required sampler host reads propagate errors |
| In-loop submit failure | Retire failed append, commit pending token, then conditionally fall back |
| Row draw failure | Restore mirostat/adaptive-p feedback before caller teardown |
| Fused RNG | Restore failed submission's thread-local key without an extra schedule |
| Fault seam | One-shot, thread-local Submit/Draw/Readback/SamplerRead plans; constant `Ok(())` outside tests |

The production fault seam is `#[inline(always)]` and returns `Ok(())` under `cfg(not(test))`; it has no runtime fault plan. This conclusion is based on source configuration, not release-binary disassembly.

## 5. Learning Points

A lazy graph can mutate sampler state before GPU execution succeeds. Correct recovery must distinguish graph construction, scheduling, readback, and token commitment. A final output comparison alone can miss a discarded greedy draw, because redrawing may produce the same token; seeded failure tests and targeted mutations expose that gap.

## 7. Change Summary

Before report additions, the branch changes 16 files with 868 insertions and 116 deletions relative to `24e7e7ff`. Implementation commits are `6f3c384b` (readback/state handling), `4669eeec` (seeded stopping boundaries), and `7facb4fa` (sampler output type aliases).

## 8. Follow-up Actions

- Confirm PR CI results before central merge.
- Keep the separate plain-drafter failed-submit/retraction problem outside this change, as specified by #2258.

## Appendix: Validation Recorded So Far

| Check | Observed result |
|---|---|
| Formatting and diff checks | Passed |
| Core CUDA lib/tests clippy, `-D warnings` | Passed after factoring three complex return types into aliases |
| Core CUDA test-fast test build | Passed |
| Original engine fault suite | 11 passed, GPU lock held, one test thread |
| Plain-drafter readback fault | 1 passed |
| Thread-local fault-plan regression | 1 passed |
| Remove pending commitment | Four separate seeded fused/mirostat/adaptive/penalty stream tests failed as expected |
| Remove feedback restoration | Feedback regression failed as expected |
| Remove fused RNG restoration | Seeded fused regression failed as expected |
| Restored implementation after mutations | All 13 fault regressions passed again, including seeded EOS/callback tests |
| New seeded EOS/callback tests | Both passed in the engine suite; each failed when pending commitment was removed independently from either pipeline (four negative checks) |
| Full engine/sampling/server suites and contracts | Engine 63 passed; sampling 202 passed; server batch 526 passed, 9 ignored; three contracts passed |
| Successful-path throughput measurement | Not measured for this draft |

Focused evidence is in `/tmp/test2258-final.log`, `/tmp/plain2258.log`, `/tmp/seam2258.log`, and `/tmp/mutation2258-{discard-pending,feedback,fused-rng}.log`. Central verification writes `/tmp/mlxcel-auto-plan/2258-*.log`. Stopping-boundary mutations are recorded in `/tmp/mutation2258-stops-summary.log` and `/tmp/mutation2258-pending-{eos,callback}-{fused,row}.log`; the final positive rerun is `/tmp/test2258-restored-stops.log`. These are session-local evidence paths, not committed artifacts.

Related work: issue #2258; request-local backend error contract #822; per-row pipeline #2256 and #2229.
