# Technical Report: PR #1301 - A total order for preemption victim selection

## Executive Summary

`BatchScheduler::select_eviction_victim` chose which in-flight sequence to preempt with `max_by_key` on `generated_tokens.len()` under `LongestFirst`, and `min_by` on priority-then-length under `LowestPriority`. Both ran over `ActiveBatch::iter_sequences`, which is `HashMap::values`, so ties fell through to hash order.

This is the eighth instance of the class recorded in `docs/code-guidelines.md`, and the only one whose tie is reachable through ordinary state rather than through a coarsening of the key. Sequences admitted together decode in lockstep and therefore share a token count, and both `PreemptionPolicy::LongestFirst` and `RequestPriority::Normal` are the shipped defaults. Two identically configured workers under identical load preempted different requests, and an operator could not reproduce the choice from the batch state.

## 1. Problem Statement

Nothing was corrupted and the policy's stated intent was never violated: any tied-longest sequence does satisfy "evict the longest". What was lost is reproducibility of which user's request was sacrificed, which is exactly what someone investigating a preemption needs.

The reachability is what separates this from #1291's four sites, whose keys are `Instant` values stamped once per call and therefore effectively never collide. Here the key is a small integer that collides routinely.

## 2. Technical Decisions

### 2.1 Smallest `seq_id`, and why `created_at` was rejected

`seq_id` is monotonic, but `try_evict_for_preemption` reallocates its victim under a fresh, higher id, so a large id marks a recently preempted sequence rather than a late arrival. Smallest-id-wins therefore rotates preemption onto sequences that have not been hit yet.

`created_at` was considered as the more operator-meaningful key and rejected for the mirror-image reason: it survives preemption, so smallest-`created_at`-wins would keep selecting the same oldest request forever, since nothing about being preempted moves the field. It is also an `Instant`, so `seq_id` would have had to sit behind it regardless. Both arms resolve to the same direction, so the choice does not vary by policy.

### 2.2 Opposite-facing tie components for the same direction

`max_by_key` returns the last maximum and `min_by` the first minimum, so the two arms need the tie component facing opposite ways to produce the same outcome.

- `LongestFirst` keys on `(generated_tokens.len(), std::cmp::Reverse(seq_id.as_u64()))`. The maximum of that key is the longest sequence and, among equals, the largest `Reverse(id)`, which is the smallest id.
- `LowestPriority` appends `.then_with(|| a.seq_id.as_u64().cmp(&b.seq_id.as_u64()))` ascending. The minimum is the lowest priority, then the longest, then the smallest id.

Because both keys are now total, neither the last-maximum nor the first-minimum rule can fire at all. `SequenceId` does not derive `Ord`, so both go through `.as_u64()`, the same trap `PromptCacheKeyDigest` posed in #1291.

### 2.3 One copy of the policy, with the filter inside it

The two existing tests reimplemented the selection expression inline rather than calling the private method, and had already drifted: neither carried the `.filter(|seq| seq.structured.is_none())` guard that production has. A regression test written in their shape would have pinned a copy.

The policy is now a free function, `select_eviction_victim_from(sequences, policy)`, with the method reduced to a one-line call. A free function rather than an `ActiveBatch` method because `active.rs` currently knows nothing about `PreemptionPolicy`. The `structured.is_none()` filter moved **inside** the function, so no caller can lose it again, and its doc comment says to extend the policy there and never at a call site.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/server/batch/scheduler.rs` | `select_eviction_victim_from` extracted; both arms given a total order; filter moved inside; comments naming `seq_id` as the total-order component |
| `src/server/batch/scheduler_tests.rs` | The two drifted tests rewritten to call the extracted function; 3 new tests |
| `docs/code-guidelines.md` | The instance list bumped to eight, plus the arithmetic that depended on "seven" |

## 4. Review Findings

The tests were built before the fix rather than written after and validated by reverting, so no `git stash` and no file surgery was needed. Against the unfixed expression:

```
LongestFirst resolved a fully tied batch to something other than seq_id 2 on 58 of 64
freshly built batches; the tie is falling through to HashMap order

LowestPriority resolved a batch tied on priority and length to something other than seq_id 2
on 39 of 64 freshly built batches; the tie is falling through to HashMap order
```

The tests accumulate mismatches over all 64 iterations rather than panicking on the first, which is what produces the "N of 64" figure and shows the pre-fix failure rate directly.

Three things worth recording:

Updating the guideline's instance count to eight made three other sentences arithmetically wrong, which the issue did not anticipate. The dependent figures were corrected, and the measurement sentence was scoped to "the first five fixes above (#1293 was found after the measurement ran)" rather than silently restating #1287's measurement as covering something it never measured.

One nuance was added to the enforcement record rather than left to be misread: #1293 **does** use `max_by_key`, so the candidate check's "0 of 8" is not because the rule ignores that method. It was missed because the receiver is a cross-module accessor (`ActiveBatch::iter_sequences`) rather than a literal `.values()`, which is the limitation the same section already describes for #1277.

The `structured.is_none()` guard is now in the executed path but no test constructs a `Some(..)`, because `StructuredOutputConstraint` has no test constructor and building one needs a real tokenizer plus a compiled grammar. Stated in the PR body rather than left implicit.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64), branch current with `main` at gate time, run with no other cargo process on the box.

- `make verify-test-cuda`: recorded in the PR thread.
- `cargo test --profile test-fast --features cuda --lib server::batch::scheduler`: 111 passed, 7 ignored, exit 0. `server::batch`: 373 passed, exit 0. The eviction filter alone: 5 passed.
- `cargo fmt --all -- --check`, `cargo check --lib --tests --features cuda`, `cargo clippy --lib --tests --features cuda -- -D warnings`: all exit 0.

Neither pre-existing eviction test changed its expected victim, because neither constructs a tie.

## 6. Related Work

- #1293: the issue this closes.
- #1287 and PR #1290, plus the correction in PR #1294: the guideline this instance is added to, and the `max_by_key` direction fact that the fix depends on.
- #1291 and PR #1299: the four latent siblings, where the opposite call was made on testing because the tie is unreachable.
- #1265 and PR #1266, #1267 and PR #1269, #1276 and PR #1281, #1277 and PR #1284, #1286 and PR #1288: the rest of the class.
