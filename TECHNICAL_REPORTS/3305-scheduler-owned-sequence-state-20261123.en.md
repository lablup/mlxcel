# PR #3305: Move per-layer sequence state from models to the scheduler

**Date**: 2026-11-23
**Status**: Open PR, not merged (closes #3254, part of epic #3259); verified with clippy, changed-area lib tests, core tests, integration contract tests, engine parity and a decode benchmark on GB10
**Languages**: Rust, Python (benchmark scripts), Markdown
**Risk Level**: High

## Executive Summary

Gemma 3 and Qwen 3.5 text no longer keep per-sequence caches inside their model wrappers. Each sequence's state is a `Vec<LayerState>` held by its `CachePool` entry, the engine forwards it through a new `LanguageModel::forward_layer_state`, and snapshot, restore, truncation, turn-boundary checkpoints, ring checkpoints, pad trim and decode rewind are implemented once (`server::prompt_cache::state_snapshot` and `mlxcel_core::cache::layer_state`) instead of once per family. ADR 0010 records the design and an audit of every model-owned family. Stored snapshots keep their tensor names, shapes and scalars, replies and `/v1/cache/stats` match `main` on real checkpoints, and decode throughput shows no regression in any measured cell. The other model-owned families stay on their existing paths and are split into six follow-up issues under #3259.

## 1. Problem Statement

### 1.1 Background

Prompt-cache reuse was built once for scheduler-owned dense and paged KV (radix-trie store, `adopt` / `adopt_paged`, `DetachedCacheSet::truncate_to`). Families whose layers are not all the same storage (sliding rings beside full layers, recurrent state beside attention) kept their caches inside the model wrapper, keyed by sequence id, because the trait hands a forward `&mut [KVCache]` and one homogeneous slice cannot carry a `RotatingKVCache` or a GatedDeltaNet state. Those families answer `SequenceStateLayout::model_owned`; the scheduler skips plain KV-entry reuse for them and `CachePool::detach` refuses them.

### 1.2 Existing Issues

- **Every capability written once per family.** Exact-prefix restore, truncation (#1145), turn-boundary checkpoints (#3047), ring checkpoints (#3255), pad trim (#1755) and decode rewind (#2159) each had a per-family hook. Families that never got snapshot hooks had no prefix reuse at all.
- **The next epic items would multiply this again.** Paged sliding windows (#3256), `--swa-full` (#3257) and quantized sidecars (#3250) each touch sequence state, so each would have needed one implementation per model-owned family.
- **Two owners of state.** The pool entry existed for these sequences but only as a length mirror; the real state lived in a per-sequence map in the model, and the scheduler had to reach it through model hooks.
- **The per-conversation checkpoint cap was branch-blind** (carried from the #3283 review). `--ctx-checkpoints N` counted all branches below a re-inserted common ancestor together, so regenerating an earlier turn after a fork could evict a sibling branch's checkpoints.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A restore installs state that does not match the model (wrong layer count, mismatched K/V, offset past the rows) and output drifts silently | High | Low (every restore validates before install; tests cover the mismatch cases) |
| A caller that bypasses the pool runs several sequences on one shared fallback slot | High | Low (by-id entries on migrated wrappers panic with a message naming `forward_layer_state`) |
| A snapshot stored before the change no longer restores | Medium | Low (generic path uses the existing encoders; tests compare tensor for tensor) |
| Decode slows down from the extra indirection | Medium | Low (no regression measured; see Appendix B) |

## 2. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core/src/cache/layer_state.rs` (new) | `LayerStateKind { Full, Sliding { window }, Chunked { chunk }, Recurrent, None }` replaces `models::model_owned::LayerKv`. `LayerState { Kv, Rotating, Chunked, Recurrent, Empty }` (the Gemma 3 `Cache` enum promoted into core) implements `KvAttention` by matching to the inner cache, plus `rewind_padded_prefill`, `rewind_decode_appends` and `visible_len`. `RecurrentState` is an ordered map of named tensors plus an offset. |
| `mlxcel-core/src/cache/layer_state_request.rs` (new), `generate.rs`, `language_model_ref.rs` | `LayerStateForward` request type; trait methods `layer_state_kinds()`, `make_layer_state()`, `layer_state_snapshot_spec()`, `forward_layer_state(request, rows)` with defaults that keep unmigrated families on their path. |
| `mlxcel-core/src/cache.rs` | `SequenceStateBackend::SchedulerOwned`; `SequenceCacheSet` gains `layer_state: Option<Vec<LayerState>>`, kept also when the paged decode override allocates the entry as `PagedKvCache`. `layer_state_nbytes()` reported separately from `nbytes()`. |
| `mlxcel-core/src/engine/{mod, prefill, direct, direct_decode, lookahead, speculative}.rs` | `step`, `prefill`, `score`, `prefill_cohort` and `verify` dispatch to `forward_layer_state` when the sequence holds layer state, and run pad trim, decode rewind and the paged length mirror over the layers directly. |
| `src/server/prompt_cache/state_snapshot.rs`, `state_snapshot_dispatch.rs` (new) | Single implementation of whole snapshot, restore, truncated restore, truncatability, state checkpoint and checkpoint restore over `[LayerState]`. Dispatch takes the generic path for scheduler-owned families and the model hook for the rest. |
| `src/models/hybrid_snapshot.rs` (new) | Hybrid (attention + GatedDeltaNet) encoder extracted from `Qwen3NextCache` so Qwen3-Next and the generic Qwen 3.5 path share it. |
| `src/models/gemma3.rs` | Forward made generic over a small trait implemented by both the model-owned `Cache` (AFMoE, Kolibri-1, pipeline stage, backbone) and `LayerState`. Per-sequence map and ring checkpoint hooks deleted; one id-less fallback slot kept for bare `forward` callers. |
| `src/models/qwen3_5.rs`, `qwen3_5_layer_state.rs` (new), `qwen3_5_mtp_target.rs` | Qwen 3.5 runs on pool state by lending it as `Qwen3NextCache` for one forward (`LentCaches`). The MTP target adapter carries the borrowed state for a tick instead of a sequence id. |
| `src/models/pooled_model.rs`, `pooled_model_language_model.rs` (new) | `Pooled<M>`: a model over a real `CachePool`, so model tests written against by-id calls drive the same entry points the engine uses. |
| `src/server/prompt_cache/state_checkpoints.rs` | Per-branch checkpoint cap. |
| Scheduler (`scheduler.rs`, `scheduler/prompt_cache.rs`, `speculative_burst.rs`, `speculative_finalize.rs`) | Calls `state_snapshot::*` instead of model hooks; MTP slice guard for Qwen 3.5 sequences without layer state. |
| About 35 other model files | Mechanical `LayerKv` to `LayerStateKind` rename. |
| Docs | ADR 0010, `docs/architecture.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/llama-server-compat.md`, `CHANGELOG.md`, and the decode benchmark record with raw JSON and scripts. |

111 files, +18,571 / -1,163 lines. About 12,600 of the added lines are raw benchmark JSON under `docs/benchmark_results/scheduler-owned-state-decode-gb10-2026-11-23/`. 33 new `#[test]` functions.

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `1105b1db` | fix | count the checkpoint cap per conversation branch |
| `0dbb8502` | refactor | move per-layer sequence state from models to the scheduler |
| `365ee0b2` | test | pin scheduler-owned round trips for Gemma 3 and Qwen 3.5 |
| `c98af10f` | fix | keep by-id drivers and the MTP adapter off the shared fallback slot |
| `902e6b6b` | docs | record the scheduler-owned state decode benchmark |
| `9151cc8d` | fix | share one pooled driver in the image trace and decline MTP without layer state |
| `03b630ca` | docs | add the Gemma 3 256-token rerun and state what the benchmark leaves open |
| `f3caea80` | fix | refuse ids on Gemma 3 batched entry and keep pooled MRoPE state |
| `9a274d0e` | fix | validate attention tensors on every scheduler-owned restore |
| `00940c54` | update | keep Qwen 3.5 recurrent slot tables in place while lent |
| `4a879e9c` | fix | refuse a restored full layer with no rows behind its offset |
| `3ddbeb0d` | test | pin the Qwen 3.5 MTP slice guard and record the quiet-host Gemma 3 rerun |

## 3. Technical Decisions

### 3.1 A separate forward instead of a polymorphic `KVCache`

**Context:** The state of a migrated family cannot ride in `&mut [KVCache]`.

| Option | Pros | Cons |
|--------|------|------|
| Make `KVCache` polymorphic over rotating, chunked and recurrent storage | Every existing forward signature stays | A second storage dispatch inside `KVCache` duplicates the `KvAttention` dispatch from ADR 0008/0009 and puts recurrent tensors behind an attention-cache type |
| Keep by-id entries on migrated wrappers with a per-sequence table in the wrapper | No engine changes | The model-owned layout under another name, with two owners of one state |
| **Chosen: `forward_layer_state(request, rows)` where `rows[i]` is row `i`'s `&mut [LayerState]`** | Pool is the single owner; engine handles pad trim, rewind and length mirror once | The engine now has two forward routes, and each migrated family must implement the new entry |

**Trade-off:** The engine checks for layer state before every forward. The old by-id entries remain on the trait for unmigrated families, and on migrated wrappers they panic with ids rather than silently running on the fallback slot. Review found the Gemma 3 batched entry was still reachable through the trait default, which dropped the ids (`f3caea80`); it now refuses ids like the single-row entries.

### 3.2 Reuse the existing encoders, do not define a new snapshot format

`state_snapshot` encodes no tensor itself. A family's `LayerStateSnapshotSpec` selects either `Windowed { standard, rotating }` (the existing `kv_snapshot` and `kv_ring_checkpoint` under the family's own names; Gemma 3) or `Hybrid` (the extracted `hybrid_snapshot`; Qwen 3.5). A snapshot written by the generic path is byte-compatible with what the model hooks wrote, so a running server's stored snapshots and any persisted state survive the upgrade. Tests compare both paths tensor for tensor and pin the names. The cost is that the generic path inherits the per-family naming vocabulary; it cannot be dropped until no stored snapshot uses it.

### 3.3 Lend Qwen 3.5's state instead of rewriting its decoder

Qwen 3.5's decoder layers, batched decode, speculative verify and rollback are written over `Qwen3NextCache`, which Qwen3-Next and Qwen4-Exp share and still own. Rewriting those paths over `LayerState` would have touched two unmigrated families. Instead `LentCaches` moves each layer out of the pool's vector into the typed view for one forward and back on drop: no tensor copy, and the state lives only in the pool between forwards. Every layer is checked before any is moved, so a refusal leaves the state untouched. Review found the first version rebuilt a `RecurrentState` per linear layer on drop, one heap allocation per linear layer, row and decode step; `00940c54` keeps the recurrent slot table in place and moves only the tensors. When Qwen3-Next and Qwen4-Exp migrate, the typed view can become the storage and lending goes away.

### 3.4 Validate every restore before install

A restore fills a fresh vector and installs it only when every layer validated: family tag, layer count, storage kind per layer against the table, and the key and value tensors of each attention layer (`check_restored_layer` calling `check_kv_tensors`). The initial implementation ran the tensor check on whole and truncated restores but not on checkpoint restore, and the hybrid arm relied on `hybrid_snapshot::restore_attention`, which assigns keys, values and offset unchecked (`9a274d0e`). A second gap: Qwen 3.5's attention decoder sets the offset to the snapshot's token length even when the snapshot has no attention rows, and an absent pair was accepted (`4a879e9c`). The check now reads:

```rust
(None, None) => {
    return match full_offset {
        Some(offset) if offset > 0 => {
            Err(format!("no keys or values behind an offset of {offset}"))
        }
        _ => Ok(()),
    };
}
```

A failure is a named error, the sequence is untouched, and the scheduler records `LayoutConstraints` and prefills cold, as before. Windowed and KV-shared layers restore absent tensors at offset 0 and are unaffected.

### 3.5 Migrate two reference families, not all of them

Gemma 3 (full + sliding) and Qwen 3.5 text (full + recurrent) cover the two shapes of heterogeneous state. Gemma 4 and AFMoE keep their #3255 hooks, which call the same `kv_ring_checkpoint` functions as the generic path, so ring checkpoint logic already has one implementation with two callers. The audit in ADR 0010 found one real reason for model ownership across all families (a heterogeneous per-layer vector versus a homogeneous slice); none requires the state to live in the model. The rest are grouped into six follow-ups (Gemma 4/AFMoE; other full + sliding families; Llama 4 chunked; GatedDeltaNet and linear hybrids; SSM and convolution hybrids; families with more than one storage per layer). `exaone_moe` and `baichuan_m1` were found not to be model-owned and leave the table.

### 3.6 Checkpoint cap per root-to-leaf line

A checkpoint is dropped when some line through the inserted checkpoint holds `N` longer checkpoints above it. A fork keeps up to `N` per branch, the result does not depend on insertion order, and the existing checkpoint byte budget remains the bound across all branches. The cap no longer bounds total checkpoints per conversation by itself; the byte budget does.

## 4. Implementation Details

```
[Before]
Scheduler ─ CachePool entry (length mirror only)
    │
    └─ model hooks ─> Wrapper { HashMap<SequenceId, Vec<Cache>> }   (one per family)
                         snapshot / restore / checkpoint / trim / rewind per family

[After]
Scheduler ─ CachePool entry { layer_state: Vec<LayerState> }   (single owner)
    │            │
    │            └─ Engine: forward_layer_state(request, rows)
    │                       pad trim / decode rewind / visible_len over LayerState
    └─ state_snapshot::* ─> kv_snapshot / kv_ring_checkpoint / hybrid_snapshot encoders
                             (model hooks only for unmigrated families)
```

Pooled MRoPE state: `Pooled::prepare_sequence_state` first called the model's release hook, which for Qwen 3.5 deletes the sequence's MRoPE positions. The mixed MRoPE and text decode parity test installs those positions before preparing, so its multimodal row silently became a text row. Prepare now drops only the pool entry (`f3caea80`).

MTP slice guard: a Qwen 3.5 MTP slice step for a sequence whose pool entry holds no layer state now fails that request with one client-visible error instead of panicking in the target adapter, matching slice 0 (`f3caea80`, test in `3ddbeb0d`).

## 5. Learning Points

### 5.1 Lend-and-return over an owned vector

`LentCaches::take` holds `&mut [LayerState]` and builds a `Vec<Qwen3NextCache>` by moving each attention layer out with `std::mem::replace(layer, LayerState::Empty)` and taking each recurrent layer's two tensors out of its slot table; `Drop` moves everything back. This gives a legacy typed API temporary ownership without copying device tensors and without two long-lived owners. The pattern requires validating all layers before moving any, otherwise a mid-way refusal leaves a half-emptied vector.

### 5.2 A storage enum that implements the attention trait

`LayerState` implements `KvAttention` by matching to its inner cache, so attention code that takes `impl KvAttention` runs unchanged over pool state, and the place that maps `Sliding` to a ring buffer is a single constructor (`LayerState::new`). Replacing rings with window-bounded paged blocks (#3256) changes that constructor and the `Rotating` arms only, not model files.

## 6. Related PRs and Issues

- Issue #3254: this PR's issue. Epic #3259: sliding-window and hybrid state work.
- #3283 / #3255: `--ctx-checkpoints` and ring checkpoints; this PR routes Gemma 3's ring checkpoints through the generic path and fixes the branch-blind cap from that review.
- #3047 (state checkpoints), #1145 (truncation), #1755 (pad trim), #2159 (decode rewind), #1346 (paged decode override): capabilities now implemented once for migrated families.
- ADR 0007, 0008, 0009: constraints this ADR builds on.

## 7. Follow-up Actions

### Required
- [ ] File and implement the six migration issues under #3259 listed in ADR 0010.
- [ ] Decide the fate of the ignored test `rollback_speculative_cache_panics_on_ragged_batched_accepts` in `src/models/qwen3_5_tests.rs`. It fails on this branch: the panic message it expects is absent from `qwen3_5.rs` on `main` too, and it was not run on `main`, so whether it was already broken is not established.

### Not verified
- No decode benchmark for Gemma 4 or AFMoE. Their path is unchanged apart from the `LayerKv` rename, so none was run.
- The MTP slice guard test (`scheduler_qwen35_slice_guard_tests`) passes, but it was not checked to fail with the guard removed.
- `make verify-test-cuda` was not part of the recorded verification.

### Monitoring
- `LayoutConstraints` rejections in `/v1/cache/stats` on Gemma 3 and Qwen 3.5 after upgrade; a rise would mean stored snapshots failing the new restore validation.
- `layer_state_nbytes()` is not included in `SequenceCacheSet::nbytes()`, matching what was reported before. Memory accounting that relies on `nbytes()` still does not see this state.

## Appendix

### A. Verification

Orchestrator pass on the final branch:

- `cargo clippy -D warnings` on core and lib, including tests.
- 2,134 changed-area lib tests and 919 `mlxcel-core` tests, run serialized; integration contract tests.
- Engine parity: identical on `qwen3.5-0.8b`. `gemma-3-1b` and `gemma-4-e2b` each show one dense versus prompt-cache-miss divergence; both reproduce identically on `main` at `bd931f76`, so they predate this PR.
- Real checkpoints, `main` against branch with the prompt cache on and `--ctx-checkpoints 4` (three-turn chat, an edit, a replay, four concurrent fresh requests) on `gemma-3-1b-it-4bit` and `qwen3.5-0.8b-4bit`: identical replies at every step and identical `/v1/cache/stats` values apart from a wall-clock field. A one-snapshot difference after the concurrent step on Qwen 3.5 repeated on both binaries and depends on request arrival.

### B. Decode Benchmark

`docs/benchmark_results/scheduler-owned-state-decode-gb10-2026-11-23.md`: GB10, CUDA release builds, arms `main-a`, `branch`, `main-b` (null arm, same binary as `main-a`), rotating order, prompt cache off, `P` in {256, 8192}, `B` in {1, 4}. Reply hashes identical across arms in every cell.

Runs 1 to 4 ran on a shared, loaded host. Gemma 3 at `B = 4` read -2.2% to -4.1% there. Run 5 repeated those cells on a quiet host with the branch rebuilt at `4a879e9c`:

| model | P | B | main-a | main-b | branch | null | branch vs main |
|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 4 | 366.21 | 363.91 | 388.52 | 0.6% | +6.4% |
| gemma-3-1b-it-4bit | 8192 | 4 | 350.60 | 344.62 | 357.85 | 1.7% | +3.1% |

The earlier negative readings came from host load. With twelve samples per arm, `B = 1` cells fall within the null (Gemma 3 -0.1% at 256; Qwen 3.5 +3.9% and -1.4%). Qwen 3.5 at `B = 4` read -0.1% and -0.3% in its quiet run. No cell shows a regression. The +6.4% should not be read as a speedup: the branch does the same device work per step, and no code change explains a gain.
