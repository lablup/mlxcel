# Technical Report: PR #1299 - A total order for the four LRU eviction victim selections

## Executive Summary

Four LRU eviction sites picked their victim with `min_by_key` directly over a `HashMap`. `Iterator::min_by_key` returns the first minimum, so entries sharing a timestamp were separated by `HashMap` iteration order, which `RandomState` randomizes per map instance. Each now selects through a total order by appending the map key.

**This was not a live defect and the change does not fix an observed bug.** All four keys are `std::time::Instant`, stamped once per call, so the tie was not reachable. What the change removes is a latent fragility that would become real the moment the key got coarser than a per-call `Instant`.

## 1. Problem Statement

The sites, all over an unordered map:

| Site | Function | Map |
| --- | --- | --- |
| `src/server/prompt_cache/store.rs:315` | `evict_oldest` | `entries: HashMap<PromptCacheKeyDigest, EntrySlot>` |
| `src/server/prompt_cache/store.rs:336` | `evict_oldest_snapshot` | `snapshots` |
| `src/server/responses_store.rs:243` | LRU eviction loop | `HashMap<String, Entry>` |
| `src/server/conversation_store.rs:151` | `evict_to_capacity` | `HashMap<String, Entry>` |

Why the tie is unreachable today, stated so the severity is not misread later: each writer computes `Instant::now()` once per call and stamps a single entry with it, and `Instant` resolves to nanoseconds on the platforms this project targets, so two entries never take the same value. The reachable-tie counterpart of this pattern is #1293, where the key is `generated_tokens.len()`, a small integer that ties routinely.

These four are also the reason the `min_by_key` static-check candidate was declined in #1287: it flags exactly these, catches none of the seven live instances, and gating on it would have meant suppressing four of four findings on day one. Fixing them by hand removes that tension rather than carrying it.

## 2. Technical Decisions

### 2.1 Two forms, chosen per key type

`PromptCacheKeyDigest` derives only `Clone, Copy, PartialEq, Eq, Hash` (`src/server/prompt_cache/key.rs:54`), so the obvious tuple key does not compile. Rather than widen the type's derive, both prompt-cache sites key on the raw bytes: `min_by_key(|(digest, slot)| (slot.entry.last_used(), *digest.as_bytes()))`. `[u8; 32]` is `Ord`, and the digest type keeps its original surface.

The two `String`-keyed stores use the comparator form, `min_by(|(key_a, a), (key_b, b)| a.last_accessed.cmp(&b.last_accessed).then_with(|| key_a.cmp(key_b)))`, because `min_by_key` would have to clone the `String` into a tuple on every element.

Both forms are recorded as correct in `docs/code-guidelines.md` under "HashMap Iteration Order", and the choice between them there is explicitly stylistic. Each site carries a comment naming the tie-break component as the thing that makes the order total, per the same guideline, since otherwise it reads as redundant and gets simplified away.

### 2.2 No regression test, deliberately

The issue framed this as a judgment call rather than a requirement, and the judgment here is not to add one. Constructing the tie means writing an identical `Instant` directly into private state, bypassing every real write path (`touch`, `insert`, `get`, `append`). A test built that way pins the behavior of a comparator rather than any reachable production state, and it would need private-field access or a test-only setter to exist at all. The reasoning is recorded in the PR body so the absence is a decision rather than an oversight.

This is the opposite call from #1288, where the tie was reachable and eight tests were both possible and necessary. The distinction is reachability, not effort.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/server/prompt_cache/store.rs` | Both sites key on `(last_used(), *digest.as_bytes())`, with comments |
| `src/server/responses_store.rs` | Comparator form tie-breaking on the map key, with a comment |
| `src/server/conversation_store.rs` | Same |

33 lines added, 5 removed, across three files. No public API changed, no type derive widened.

## 4. Review Findings

One implementation detail worth recording, checked rather than assumed: `min_by_key` invokes its closure once per element, so `slot.entry.last_used()` still takes its mutex exactly as many times as before. Folding the digest into the key adds no additional locking.

The issue's coordination note about #1248 was re-checked at implementation time: still `status:ready`, no PR, not in flight, so all four sites stayed in this PR rather than two of them moving there.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64). The branch was current with `main` at gate time.

- `make verify-test-cuda`: recorded in the PR thread.
- `cargo test --profile test-fast --features cuda --lib server::prompt_cache`: 168 passed, exit 0. `server::responses_store`: 8 passed. `server::conversation_store`: 5 passed. Both of the latter through the prebuilt binary, since `cargo test` takes one filter.
- `cargo fmt --all -- --check`, `cargo check --lib --tests --features cuda`, `cargo clippy --lib --tests --features cuda -- -D warnings`: all exit 0.

## 6. Related Work

- #1291: the issue this closes.
- #1287 and PR #1290: the guideline that records the class and the accepted remedy forms, and whose declined static check flagged exactly these four sites.
- #1293: the reachable-tie sibling in `BatchScheduler::select_eviction_victim`, handled separately because its tie occurs in ordinary batch state and therefore does warrant tests.
- #1265 and PR #1266, #1267 and PR #1269, #1276 and PR #1281, #1277 and PR #1284, #1286 and PR #1288: the seven live instances of the same class.
