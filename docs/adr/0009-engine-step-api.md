# ADR 0009: The engine step API and the batch-of-one rule

**Status:** Accepted (2026-10-07). Epic #2166 Phase 4b (issue #2172). Amends [ADR 0007](0007-unified-batch-native-engine.md): it settles the signatures the "Engine" row of the component table left to this phase and records how "a single sequence is a `StepBatch` of one" is implemented. ADR 0007's decision table stands unchanged.

## Context

ADR 0007 fixed the names `Engine`, `SequenceSpec`, `SequenceId`, `open`, `prefill`, `step` and `close` and deferred their signatures to the phase that introduces them. `BatchScheduler` ran model forwards, sampler draws and the finish step itself at nine sites, with the single-sequence path (`LanguageModel::forward`) and the batched path (`forward_batched_with_context_and_ids`) reached from different scheduler functions. The issue body also proposed making `LanguageModel::forward` a provided method that delegates to the batched entry with one row.

## Decision

### Signatures

`mlxcel_core::engine::Engine<M: LanguageModel>` owns the model and the `CachePool`. The scheduler holds one and reaches the model and the pool only through `model()`, `pool()` and `parts_mut()` for capability queries, snapshot and restore hooks, admission accounting, prompt-cache detach and handoff; every forward and every sampler draw goes through the entries below.

| Entry | Signature | Notes |
|---|---|---|
| open | `open(SequenceSpec { layout_override: Option<SequenceStateLayout> }) -> Result<SequenceId, EngineError>` | Pool allocation plus `prepare_sequence_state`. `adopt(DetachedCacheSet)` / `adopt_paged(DetachedPagedCacheSet)` open a sequence over a prompt-cache entry. |
| prefill | `prefill(&PrefillStep { seq_id, input, embeddings, mask, last_pos, trim_excess, eval }) -> Result<PrefillOutcome { logits, eval, trim }, EngineError>` | One `PrefillPlan` piece per call, so chunked prefill interleaves with decode. The forced eval and the pad trim are reported separately so the caller records the eval against its backend health counter before acting on the trim. `prefill_cohort(&[SequenceId], input, mask)` is the padded batched window; `complete_prefill(&logits, &mut StepRow<H>) -> RowOutcome` samples and finishes the first token. |
| step | `step(&StepBatch { seq_ids, input, context }, &mut [StepRow<'_, H>]) -> Result<StepOutput { rows: Vec<RowOutcome> }, EngineError>` | The one decode entry for every row count. `StepRow` carries the row's `RowSampler`, `SamplingConfig`, token history, generated stream, EOS set, `max_tokens`, `LogprobsConfig` and an `H: StepRowHooks` (logit mask, token override, matcher, first-token stamp, plus the `FinishHooks` half). The engine takes the fused `[B, vocab] -> [B]` draw when every row is fused-eligible and shares its parameters, else the per-row chain: mask, draw, eval, override, logprobs, matcher, `finish_step`. Rows that continue advance their pool offset. |
| submit / finish_rows | `submit(&StepBatch, &FusedSampleParams, &[&TokenBiasMap]) -> Result<UniquePtr<MlxArray>, EngineError>`; `finish_rows(&[i32], &mut [StepRow<'_, H>]) -> Vec<RowOutcome>` | The split form the lookahead pipeline needs: the speculative forward runs inside `DecodeLookaheadAppendScope`, the fused draw is scheduled with `async_eval` and returned as lazy device tokens; the collect half finishes each row once the tokens are on the host. `unwind_appends(SequenceId, n)` undoes up to `DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS` positions through the pool trim or `rewind_decode_appends`. |
| close | `close(SequenceId) -> ClosedSequence { id, was_open }` | Releases the model's per-sequence state and the pool entry. |

Errors are per row. `EngineError` (`MissingSequence`, `Batch`, `Open`, `Trim`, `Rewind`, `Eval`) names the sequence it concerns; `RowOutcome.error` (`RowError::Structured`, `RowError::Eval`) marks one row's failure inside a batch, and the scheduler finishes only that row with `FinishReason::Error` (#822). A failed `unwind_appends` is the row-only error that overrides a `Length` or `Stop` recorded on the same tick so the desynchronized state is not donated (#2182).

### A single row equals a batch of one, at the engine

`Engine::step` is the only decode entry for every `B`; the scheduler's synchronous single and batched paths both call it (`run_engine_step`). Inside, `B = 1` runs the family's single-row forward (`forward_with_sequence_id`), which is what the provided `forward_batched` does at `b == 1` and what the measured B=1 throughput rests on; `B > 1` runs `forward_batched_with_context_and_ids`.

The trait shape is unchanged: `LanguageModel::forward` stays the required method and the batched entries stay provided methods with per-family overrides. Inverting the required method (a provided `forward` delegating to a required batched entry) is a mechanical change across 224 implementations, about 40 of which override a batched entry, and buys nothing the engine-level single entry does not already give: no caller other than the engine chooses between the two forwards. Instead, the equivalence the issue's criterion wants is pinned where it can be violated: `src/models/single_row_batch_parity_tests.rs` runs, on a real checkpoint under `MLXCEL_SDPA_DETERMINISTIC=1`, the single-row forward and the one-row batched forward of every locally available family that overrides the batched entry (Qwen3, Llama 3) and requires byte-identical logits at each of four decode steps. A family whose one-row batched path ever differs makes its `b == 1` delegate to the single-row forward.

## Consequences

- The scheduler keeps admission, tick policy (ADR 0005), preemption, prompt-cache lookup, donate and adopt, disaggregated handoff and the speculative burst generators; `rg '\.forward[a-z_]*\(|sample_token_[a-z_]*\(' src/server/batch/scheduler --glob '!*_tests.rs'` returns only `PrefillPlan::forwarded_len()` accessors, and no `RowSampler::draw` or `fused_sample` call. The `prefill_span_coverage_tests` source guard classifies the engine entries instead of raw forwards.
- `mlxcel-engine-parity` gains arm (d), a direct `Engine` B=1 run through `engine_probe::engine_direct::DirectEngine`, compared against the server's dense B=1 arm; a divergence there is scheduler policy, not engine execution.
- Phase 5 (#2173) gives the CLI a `StepRowHooks` of its own and drives `Engine` the way `DirectEngine` does.
- Deferred from this phase and still open: Gemma 3, Llama 4 and the `model_owned` dispatch helpers keep their dense-pointer paged kernels and read `DecodeBatchContext`; moving them onto `attend` / `attend_batched` needs a `RotatingKVCache::attend` that keeps `RotatingKVCache::update` as the write path for the #2182 undo log, and the `DecodeBatchContext` argument of `forward_batched_with_context_and_ids` (ignored by qwen3_5, qwen3_moe and muse_glimmer) goes with it.

## References

- Epic #2166, issue #2172, PR #2217.
- [ADR 0007](0007-unified-batch-native-engine.md), [ADR 0008](0008-kv-attention-dispatch-in-the-cache.md).
- `src/lib/mlxcel-core/src/engine/mod.rs`, `src/lib/mlxcel-core/src/engine/rows.rs`, `src/server/batch/scheduler/step_rows.rs`, `src/server/engine_probe/engine_direct.rs`.
