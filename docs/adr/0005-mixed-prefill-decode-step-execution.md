# ADR 0005: Mixed prefill/decode step execution

**Status:** Accepted (2026-08-03), issue #908, final sub-issue of epic #909.

**Decision in one line:** reject model-level ragged mixing and kernel-level mixing on the MLX path; keep tick alternation; fix the chunked-prefill starvation this spike found, which is the real defect behind the symptom issue #908 set out to solve.

## Context

Issue #908 opens with a premise: mlxcel's scheduler alternates prefill and decode by tick, so "while a prefill chunk runs, every decoding stream waits a full model forward, which shows up as inter-token latency (ITL) spikes on admission". The proposed remedy is to stop alternating and start mixing, at one of two maturity levels: a ragged forward that concatenates the decode batch with one prefill chunk, or a single-launch kernel that serves both from one work queue.

The premise is wrong, and it is wrong in the direction that removes the motivation. `decide_action` does not alternate. Reading the branch that fires while a long prompt is mid-prefill (`src/server/batch/scheduler.rs`, now `src/server/batch/tick_policy.rs::decide_tick`):

```rust
// A chunked prefill is parked mid-prompt.
if state.chunked_prefill_in_progress {
    if !state.active_is_empty {
        return TickChoice::Decode;
    }
    return TickChoice::Prefill;
}
```

The policy is a pure function of scheduler state, and running a decode changes none of the state it reads. So when a chunked prefill is parked and any sequence is decoding, the tick resolves to `Decode`, and the next tick resolves to `Decode` again, and so on until the last decoding sequence finishes. The parked prompt runs chunk 0 at admission and then makes no further progress. Decoding streams are never blocked by a chunked prefill. The chunked prefill is blocked by them.

That behaviour has been in place, asserted, and mislabelled for some time. `scheduler_tests.rs` carried a test named `chunked_prefill_interleaving_pattern` whose third assertion expects `Decode` where its own comment says "after decode, back to prefill continuation", annotated "(still interleaving because batch is non-empty)". The assertions were correct and the name was not, because the tests re-implemented the policy in a local helper instead of calling it: a mirrored policy is only as truthful as the copy, and this copy was kept in sync with the code while its naming stayed with the intent. Issue #908 landed exactly on that gap.

Two further facts bound the problem before any execution model is considered.

**The issue's scenario is unreachable under the shipped defaults.** The admission branch fires only when `!active_batch.is_full() && !prefill_queue.is_empty()`. `--parallel` defaults to 4 and sets the batch ceiling, so with 4 decode streams the batch is full, the admitted request stays in the queue, and no prefill runs during decode at all. Reproducing the scenario at all requires `--parallel` strictly greater than the stream count.

**Chunked prefill bounds the admission spike to one forward.** When a slot is free, admission runs exactly one prefill forward before decode reclaims every subsequent tick: one chunk of `--prefill-chunk-size` (512 by default), or one unchunked forward for a prompt below that threshold. The sustained window of alternating spikes the issue describes does not occur, because there is no sustained window.

So the question this ADR has to answer is not the one it was asked. It is: given that the interleave does not exist today, is a fused ragged step the right thing to build once it does?

## Candidate execution models

**(a) Tick alternation, as designed.** One workload per tick, chunk size trading ITL against the admitted request's TTFT. Requires the starvation fix to actually alternate. No change to any family forward.

**(b) Ragged mixed forward at the model level.** Pack the decode batch's `q_len=1` rows and one prefill chunk's `q_len=N` rows into one tensor, run the linear layers over all rows at once, split at the attention boundary, recombine.

**(c) Kernel-level mixing.** One launch carries both workloads, decode executed as tiny-q prefill, CTAs claiming tiles from a shared atomic work counter.

### What (b) can recover, exactly

Within one transformer layer, the two workloads fuse very differently.

The projections (QKV, O) and the MLP (gate, up, down) are row-wise. Appending `B` decode rows to a `C`-row prefill chunk turns a `C`-row GEMM into a `(C+B)`-row GEMM, which at `C=512, B=4` is 0.8% more rows for the same weight read. Under alternation those same `B` rows pay a second full read of every weight in the model. This is the entire win.

Attention does not fuse at all. The prefill rows attend causally within the chunk plus their own past; each decode row attends to its own separate KV history. There is no single SDPA that serves both, so a mixed step runs the same attention calls the two separate steps ran, just inside one forward. Nothing is saved.

The recoverable fraction of an admission window is therefore

```
saving  =  D_lin / (C/P + D)
```

where `C` is the chunk size, `P` the prefill throughput at that chunk, `D` a decode step's wall time, and `D_lin` the part of `D` that is not attention. The denominator is one co-scheduled tick under (a) with the starvation fixed; the numerator is what (b) removes from it.

Grounding the terms in already-published in-tree measurements, not in a run made for this ADR: `docs/benchmark_results/longprompt-prefill-gb10.md` puts llama-3.1-8b-4bit at 3286 tok/s prefill at 512 tokens and 1666 tok/s averaged over 8192, so a 512-token chunk costs roughly 160 ms early in a prompt and 310 ms or more late in one, since the average hides an attention term that grows with position. A batched decode step on the same class of model is tens of milliseconds. Across those ranges `saving` spans roughly 3% at the pessimistic end to about 20% at the optimistic one, and the optimistic end pairs the fastest chunk with the slowest decode step, which is not a combination one machine produces. At the shipped 512-token chunk it sits toward the low end, and it falls further as the chunk grows, because `C/P` grows with `C` while `D_lin` does not.

That upper bound is worth taking seriously rather than rounding away, and it still does not carry the decision. A 20% saving would apply only to admission windows, which are a small fraction of serving wall time; `--prefill-chunk-size` reaches the same frontier from the other direction for free; and this epic has now watched op-level wins shrink on contact with their call sites more than once (#899 measured 2.78x fused and 1.15x end to end). A bound of that size does not justify a per-family ragged execution path. It does justify measuring both terms rather than asserting them, which is what the harness below does.

The same formula explains where (b) would be worth something: small chunks. At `C=128` the denominator shrinks about fourfold while `D_lin` is unchanged, so the saving rises correspondingly. But that regime is reachable today by setting `--prefill-chunk-size 128`, which costs one flag and no code, and which moves the ITL/TTFT frontier over a far wider range than (b) can. `--prefill-chunk-size` is the first-order lever. A ragged forward is a second-order correction on top of it.

### What (b) and (c) would cost

The tree has no mechanism for a genuinely ragged batch, and the places that look like one are not. `plan_prefill_cohorts` pads every cohort member to the window's longest prompt. Gemma 4's batched MTP path, whose helper is named for ragged shapes, left-pads every row to `max_prompt_len`. `create_padded_prefill_mask` and `create_causal_mask_with_window_and_left_padding` both take a single `size`/`padded_len` for the whole batch axis. Every production caller of `BatchedAttentionMetadata` builds it through `uniform_kv_caches`, which forces one shared query length even though the struct can carry per-row lengths. The one true variable-length pattern in the tree, `cu_seqlens` segment loops, lives only in the vision towers, where attention is non-causal, has no KV cache, and needs no RoPE offset, so it solves a structurally easier problem.

Building (b) therefore means: a packed `[1, N]` layout, per-token rather than per-row RoPE positions, a new segment-loop attention entry point, and logits extraction that picks different rows for the decode segments and the prefill segment. Per family. Llama 3 alone is a prototype; feature parity is every family.

**Graph-cache churn is a real second cost, not a footnote.** MLX keys its compiled-graph and kernel-selection caches on shapes. Today the decode path sees one shape per active batch size (`[B, 1]`, `B` in 1 to `--parallel`) and the chunked prefill path sees the chunk size plus one terminal remainder, so the working set is a handful of shapes that stay resident. A packed mixed step has row count `B + C`, which varies with the decode batch *and* the terminal chunk's length, so the product of both axes churns through the cache. The effect is worst exactly where mixing is supposed to help, because the decode batch changes as sequences complete during the very window a long prefill occupies. Padding the packed length to a tile boundary would bound the shape set, at the cost of wasted rows in the GEMM the whole design exists to make efficient. That trade is unresolved and would have to be measured, which is more work before (b) could even be evaluated.

(c) inherits all of that and adds a fused kernel. Two adjacent measurements say what to expect from that. ADR 0001's Phase 6 built the fused paged-attention kernel and found it lost to gather-then-SDPA at long context on Apple Silicon, because MLX folds the gather into a tuned SDPA and unified memory removes the host copy a discrete-GPU design is avoiding. Issue #899's fused decode measured 2.78x at op level and 1.15x end to end. Nothing about (c) suggests it would escape that pattern, and its ceiling is still the `saving` formula above, since a kernel cannot recover work that is not there.

## Decision

**Reject (b) and (c). Adopt (a).**

The symptom issue #908 names is real but misattributed: admission does perturb serving latency, and the perturbation worth fixing is the admitted request's unbounded time to first token, not the decoding streams' ITL. That is a scheduler-policy defect measured in a handful of lines, not an execution-model limitation. Fusing the forward addresses a cost that is bounded by the formula above at single-digit to low-teens percent of an admission window, in a regime that `--prefill-chunk-size` already reaches more cheaply, at the price of a per-family ragged execution path the tree has no precedent for.

A "defer" would be the wrong shape here. Defer implies a trigger that later evidence could fire, as ADR 0001 did for its fused kernel. There is no such trigger for (b): its ceiling is a structural property of which layers are row-wise, and no workload makes attention fusible across a prefill chunk and unrelated decode histories. What could reopen this is listed at the end, and it is a different design, not more of this one.

### What this issue ships

**The starvation is pinned, not silently fixed.** The policy moved to `src/server/batch/tick_policy.rs` as one pure `decide_tick` over a flattened `TickState`, called by both `BatchScheduler::decide_action` and the tests, so there is no copy left to drift. `chunked_prefill_starves_until_active_batch_drains` asserts the fixed point directly. The default policy is unchanged: `mixed_step_off_is_identical_to_the_pre_908_policy` compares `decide_tick` against a transcription of the pre-#908 policy over the complete 128-state space.

Fixing the starvation itself is deliberately not part of this issue. Changing which workload wins a contended tick changes the latency profile of every concurrent workload, it needs its own fairness policy (a grant counter, a starvation deadline, or a prefill-priority mode), and it needs its own measurement. This spike's job was to decide about mixed execution, and shipping a scheduler fairness change under that heading would bury it. It is split out; see "Consequences".

**The prototype is the scheduling half, on purpose.** `MLXCEL_MIXED_STEP=1` adds `BatchSchedulerAction::MixedStep`, a tick that decodes the active batch and then advances the parked chunk, so both workloads progress. It is not a fused ragged forward: the two forwards still run back to back.

That split is what makes the experiment decisive rather than expensive. Total benefit of a full mixed step decomposes as

```
benefit(ragged fused step)  =  benefit(tick co-schedule)  +  D_lin per chunk
```

The prototype measures the first term end to end on a real server. The second term is bounded by the formula above and is measurable from the same run. If the first term is where the value is, the fused forward was never the interesting part; if the first term is small, the second is smaller still, and (b) is dead without writing it. Either way the decision does not require a per-family ragged execution path to exist first, which is the trap this epic hit when op-level wins failed to survive contact with their call sites.

**Default off, structurally.** With `MLXCEL_MIXED_STEP` unset, `MixedStep` is unreachable. The flag is read once at scheduler construction into a `bool` field, so the tick loop never touches the environment. It changes behaviour only in the `chunked_prefill_in_progress && !active_is_empty` branch, so the pure-decode control (no chunked prefill parked) and the pure-prefill control (empty batch) take a byte-identical path with the flag on. The "under 1% overhead when idle" bar in the issue is met structurally, not statistically; the harness confirms it rather than establishing it.

**Greedy token parity is structural, not measured.** The issue asks for greedy parity between the arms for both the decoding streams and the admitted request. The prototype changes only which tick a unit of work runs on. It calls the same `execute_decode_step` and the same `continue_chunked_prefill`, with the same arguments, against the same per-sequence KV caches; no forward signature, mask, RoPE offset, or sampling path differs between the arms. Decode and the chunk continuation also touch disjoint sequences, since a sequence being chunk-prefilled is owned by `chunked_prefill_seq` and is not in the active batch until `finish_prefill` admits it. The paged block pool may hand out different physical block ids under the interleaved allocation order, but block contents are per-sequence, so the values read back are identical. The lookahead pipeline is inert in both arms, because `lookahead_safe()` already returns false whenever a chunked prefill is parked, so decode is synchronous either way.

The honest caveat: this is an argument from the code, and this epic has repeatedly found that arguments from the code were wrong about what the code did. The harness run is what confirms it, and greedy parity across the two arms at `temperature 0` is worth checking on the first real run. What would break parity is a future change that lets the two workloads share a cache or a graph, which is precisely what (b) would have introduced.

**Speculative rounds and mixed steps exclude each other,** as the issue anticipated. `MixedStep` sits where `Decode` sat in the #734 alternation, so a pending speculative slice still outranks it and a yielded slice still falls through to it. `speculative_round_outranks_mixed_step` and `yielded_speculative_slice_falls_through_to_mixed_step` pin both directions. A speculative round never carries prefill work in this prototype.

## Measurement

Two harness changes exist because the scenario could not previously be expressed.

`scripts/bench_serving_concurrency.py` fires N identical requests simultaneously and reports per-request aggregates. It has no staggered admission and no per-token timestamps, so it can measure neither "a long prompt arrives while others decode" nor "ITL during the prefill window". `scripts/bench_mixed_step_admission.py` adds both: it starts `--streams` decode streams, waits `--settle-s`, admits one `--admit-prompt-tokens` request, and reports per-stream ITL p50/p95 split into a quiet window and an admission window, plus the admitted request's TTFT.

`mlxcel_batch_mixed_steps_total` on `/metrics` is the dispatch proof. It can only move when `MLXCEL_MIXED_STEP` is set and a tick advanced both workloads. The harness refuses to print a latency table when attribution fails, following `examples/sparse_paged_decode_bench.rs`:

- `/metrics` unavailable: no counter can attribute the run.
- Prefill-chunk counter flat: the admitted prompt never entered the chunked path, so there was no prefill-during-decode window at all. Usually `--parallel` was not above `--streams`, or the prompt fit in one unchunked forward.
- `--expect mixed` with a flat mixed-step counter: the run measured the default policy against itself. This is the failure mode #899 shipped.
- `--expect baseline` with a moving mixed-step counter: `MLXCEL_MIXED_STEP` leaked into the server's environment.

### Reproduce

Serve with a batch ceiling above the stream count, or the admitted request never leaves the queue:

```bash
# Baseline arm.
./target/release/mlxcel-server -m models/llama-3.1-8b-4bit \
    --parallel 8 --prefill-chunk-size 512 --metrics --port 8080
python3 scripts/bench_mixed_step_admission.py --expect baseline

# Prototype arm, same server flags.
MLXCEL_MIXED_STEP=1 ./target/release/mlxcel-server -m models/llama-3.1-8b-4bit \
    --parallel 8 --prefill-chunk-size 512 --metrics --port 8080
python3 scripts/bench_mixed_step_admission.py --expect mixed
```

Controls, both arms, confirming the flag is inert when no chunked prefill is parked:

```bash
# Pure decode: no admission, so MixedStep is never selected.
python3 scripts/bench_serving_concurrency.py --concurrency 4 --prompt-tokens 128 --max-tokens 256
# Pure prefill: empty batch, so the chunked branch continues as before.
python3 scripts/bench_serving_concurrency.py --concurrency 1 --prompt-tokens 8192 --max-tokens 8
```

Run under `caffeinate -i` and let the machine cool between arms; Apple Silicon down-clocks under sustained load and a hot machine inflates the later arm. Repeat each arm and report dispersion, not a single number.

### Results

Not yet filled. Record them in `docs/benchmark_results/mixed-step-prototype-<hw>-<date>.md` and summarize here. The two cells that decide whether anything above needs revisiting:

1. `admitted_first_token_after_last_stream` in the baseline arm. If true, the starvation is confirmed end to end and the follow-up issue is justified on measurement rather than on code reading alone.
2. The admission-window ITL p95 inflation in both arms. The baseline arm bounds what alternation costs the decoding streams today (expected: near zero, since decode never yields). The prototype arm shows what a real interleave costs them, which is the price of fixing the starvation and the number a fairness policy has to be tuned against.

The `saving` formula's terms come from the same runs: `C/P` is the admission window divided by the chunk count, `D` is the quiet-window ITL. If `D / (C/P + D)` comes out materially above the low teens on some hardware, the rejection of (b) is worth revisiting on that hardware, and only there.

## Consequences

- **`decide_action` semantics are unchanged** with `MLXCEL_MIXED_STEP` unset, proven exhaustively rather than by inspection. Every existing scheduler unit test passes untouched.
- **The mirrored-policy pattern is gone** from the scheduler tests. This is the durable part of the change: the starvation was invisible for as long as it was because the tests asserted against a copy. `scheduler_tests.rs` helpers now delegate to `decide_tick`.
- **A follow-up issue covers the starvation fix**: grant a parked chunked prefill a tick under a fairness policy so a long prompt's TTFT is bounded while the batch decodes. `MixedStep` is one candidate policy (every tick, both workloads) and the harness measures its cost; a grant counter every N decode ticks is the cheaper alternative and lands on the same ITL/TTFT frontier. That issue owns the choice, the default, and the tuning.
- **No CUDA path was written.** The prototype is scheduler-level and backend-agnostic, so it needs none, and no GB10 hardware was available to validate one.
- **The prototype stays experimental.** It is not wired to a CLI flag and is not documented as an operator knob beyond the environment-variable reference, because the fairness question belongs to the follow-up issue.

## What reopens this

- **A serving workload where `D` is a large fraction of `C/P`.** Small models with heavy per-step overhead, or a backend whose prefill is fast relative to its decode step, push the `saving` formula up. It is a per-hardware, per-model question, and the harness above measures both terms directly, so the check is cheap.
- **Genuine varlen attention in MLX.** The rejection of (b) rests partly on the tree having no ragged mechanism and on attention not fusing across the two workloads. A fused variable-length attention primitive that serves a packed `[1, N]` batch with per-segment KV would remove the per-family cost, though not the ceiling.
- **A scheduler that needs to run prefill and decode concurrently for a different reason,** such as disaggregated serving collapsing back into one process. The packing work would then be paid for by that requirement, and mixing would be a free rider rather than the justification.

## References

- Issue #908, this spike. Epic #909.
- `src/server/batch/tick_policy.rs`, the extracted policy and the `MLXCEL_MIXED_STEP` gate.
- `src/server/batch/tick_policy_tests.rs`, the starvation pin and the exhaustive default-off parity proof.
- `scripts/bench_mixed_step_admission.py`, the admission-during-decode harness.
- `docs/CONTINUOUS_BATCHING.md`, scheduler flags and the chunked-prefill contract.
- `docs/benchmark_results/longprompt-prefill-gb10.md`, the compute-bound prefill throughput the `saving` formula is grounded in.
- ADR 0001, the precedent for a measured adopt/defer decision on an execution-path change, and the Apple-Silicon unified-memory reasoning that its Phase 6 outcome confirmed.
- `examples/sparse_paged_decode_bench.rs`, the `NOT COMPARABLE` guard pattern the harness follows.
