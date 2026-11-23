# ADR 0010: The scheduler owns per-layer sequence state

**Status:** Accepted (2026-11-23). Issue #3254, epic #3259. Builds on [ADR 0007](0007-unified-batch-native-engine.md) (one engine, dense single-sequence storage default), [ADR 0008](0008-kv-attention-dispatch-in-the-cache.md) (the storage behind a cache picks the attention kernel) and [ADR 0009](0009-engine-step-api.md) (the engine owns the model and the `CachePool`). It closes ADR 0009's open item on the model-owned families for the two families migrated here and sets the rule for the rest.

## Context

Prompt-cache reuse was implemented once for scheduler-owned dense and paged KV (the radix-trie store, `adopt` / `adopt_paged`, `DetachedCacheSet::truncate_to`) and then again, family by family, for every model that keeps its caches inside the model wrapper. Those families answer `SequenceStateLayout::model_owned`, the scheduler skips plain KV-entry reuse for them, `CachePool::detach` refuses them, and the only reuse left is a whole-state `ModelStateSnapshot` written per family, and only for the families that implemented the snapshot hooks.

Every capability therefore had to be written once per family: exact-prefix restore, truncation (#1145), turn-boundary checkpoints (#3047), ring checkpoints (#3255), pad trim (#1755) and decode rewind (#2159). Families that never got snapshot hooks get no prefix reuse at all. The next items on the epic (paged sliding windows in #3256, `--swa-full` in #3257, quantized sidecars in #3250) would each have multiplied by the number of families again.

### Audit: why each family keeps its state in the model

Read from the code on `main` at `bd931f76`. "Recorded reason" is the reason for being model-owned, not the reason for a padded-prefill or storage-reporting opt-out.

| Layer kinds | Families | Recorded reason | Snapshot reuse |
|---|---|---|---|
| Full + sliding (`RotatingKVCache`) | gemma3 | `docs/CONTINUOUS_BATCHING.md`: model-owned state is never pool-backed. ADR 0009: the `Cache` enum implements `KvAttention` by matching to the inner cache. | yes, with ring checkpoints. **Migrated by this ADR.** |
| Full + sliding + KV-shared (empty slot) | gemma4, and the `gemma4_vl` / `gemma4_unified` wrappers | `gemma4.rs` wrapper doc: the sliding-window-aware `Cache` enum "cannot be wired to scheduler-managed `&mut [KVCache]` slices", mirroring Gemma 3. | yes, with ring checkpoints |
| Full + sliding | afmoe, kolibri1 | `afmoe.rs` and `kolibri1.rs` module docs: sliding layers need a `RotatingKVCache` and global layers a `KVCache`, "which does not fit the trait's homogeneous `&mut [KVCache]`". | yes (afmoe also checkpoints) |
| Full + sliding | mimo_v2_flash | Module doc, same wording as AFMoE. Also `supports_batching() == false`. | no |
| Full + sliding | muse_glimmer | Module doc: "The decoder owns a mixed cache layout", and scheduler calls must carry `SequenceId`s so each row resolves to an isolated cache vector. | yes |
| Full + sliding | ministral3, exaone4, mellum, laguna, limite | One-line wrapper comments ("uses internal cache management for sliding window attention", "owns the mixed full/sliding caches"), and `docs/CONTINUOUS_BATCHING.md` "Mixed full/sliding cache families". | no |
| Full + sliding | gpt_oss | Only "manages internal mixed caches" on the wrapper. Inferred: the same heterogeneous-cache reason. | no |
| Full + sliding | step3p5 | No recorded reason found. Inferred: the same heterogeneous-cache reason. | no |
| Sliding + pooled state in every layer | deepseek_v4 | Module doc: "Heterogeneous per-layer state (rotating local KV plus one or two pooling caches) does not fit the trait's homogeneous `&mut [KVCache]`". | no |
| Full + sliding, two caches per layer | iquestloopcoder | `iquestloopcoder_model.rs`: the external surface is one dense cache per layer, and "this family needs two caches per layer and one of them is a RotatingKVCache". Prefix reuse is excluded on purpose. | no (deliberate) |
| Full + chunked (`ChunkedKVCache`) | llama4 | Wrapper doc: internal `Llama4Cache` management for the interleaved-GQA attention pattern. The post-pad trim needs a chunk-boundary rewind (#1335, #1755). | yes |
| Full + recurrent (GatedDeltaNet) | qwen3_5 | None in the file. It inherits Qwen3-Next's: the model "interleaves attention (KVCache) and linear (GatedDeltaCache) layers, so it cannot use the trait's homogeneous `&mut [KVCache]`". | yes, with state checkpoints. **Migrated by this ADR.** |
| Full + recurrent (GatedDeltaNet) | qwen3_next | `qwen3_next.rs`, as quoted above. Listed in `HYBRID_SSM_MODEL_TYPES`. | yes, with state checkpoints |
| Full + index cache, recurrent + PLE window and token contexts | qwen4_exp | Module doc: each attention layer carries an index cache beside its KV cache, and each PLE layer a convolution window and token context beside its GatedDeltaNet state. | yes, with state checkpoints |
| Full + recurrent (GLA state) | bailing_moe_linear | Wrapper doc: linear layers hold a recurrent state and global ones a KV cache, "which does not fit the trait's homogeneous `&mut [KVCache]`". | yes |
| Full KV + four convolution states in every layer | inkling | No recorded reason found. Inferred: KV plus convolution state per layer does not fit one `KVCache` per layer. | yes |
| Recurrent / SSM + attention | lfm2, plamo2, falcon_h1, granitemoehybrid | Module docs: the convolution and SSM state is recurrent, not per-token positional, so the model owns its mixed cache. | yes |
| Recurrent / SSM + attention | jamba, mamba, mamba2, nemotron_h | One-line comments ("recurrent state is model-owned and isolated by sequence id"). Listed in `HYBRID_SSM_MODEL_TYPES`. | yes |
| Recurrent / linear + attention | kimi_linear, kimi_k3 | None in the source. `docs/turbo-kv-cache.md` lists them with the model-owned families. Listed in `HYBRID_SSM_MODEL_TYPES`. | no |
| Recurrent only, or recurrent + sliding | rwkv7, z1t, recurrent_gemma | One-line comments; `z1t.rs` says it follows RWKV-7. Listed in `HYBRID_SSM_MODEL_TYPES` (rwkv7, recurrent_gemma). | no |
| Remote, per pipeline stage | `distributed/pipeline/server_runtime.rs` | None for the layout. The padded-prefill comment states the fact: each sequence's K/V lives in the pipeline stages behind the runtime. | no |

Two entries of the issue's table are not model-owned and leave it. `exaone_moe` uses the default dense layout: its sliding layers keep a full `KVCache` history and are windowed by the mask. `baichuan_m1` also uses the dense layout and keeps only a per-layer convolution side state in the model (#2567, `kv_cache_captures_sequence_state() == false`).

The recorded reasons reduce to one: a sequence's state is a heterogeneous vector (a different storage per layer, or more than one tensor per layer), and the trait hands a forward `&mut [KVCache]`. Nothing in them requires the state to live inside the model.

### Constraints carried in

- ADR 0007 measured paged B=1 decode 6 to 23 percent slower than dense on Qwen3 and Llama, so the default single-sequence storage stays dense.
- Int8 and Turbo sequences are never pool-backed (`cache/paged_batch_decode.rs` variant table).
- A snapshot written before this change must restore after it.
- A family that is not migrated keeps every code path it has.

## Decision

### One table per family

`mlxcel_core::cache::LayerStateKind { Full, Sliding { window }, Chunked { chunk }, Recurrent, None }` replaces `models::model_owned::LayerKv`. `LanguageModel::layer_state_kinds()` returns one kind per layer, or `None` (the default) for a family that stays on its current path. `KvCacheLayerModes::storage_modes` and `table_less_storage_modes` take the same kinds, so storage reporting, allocation and the snapshot policy read one table and cannot drift.

`LayerKv::Fp16Only` had two meanings. Llama 4's chunked layers become `Chunked { chunk }`, which stores FP16 whatever the injected mode is. Every other user was a family with no mode table, where `Full` under the fixed FP16 mode reports the same thing.

### One storage type, owned by the pool entry

`mlxcel_core::cache::LayerState { Kv(KVCache), Rotating(RotatingKVCache), Chunked(ChunkedKVCache), Recurrent(RecurrentState), Empty }` is the Gemma 3 `Cache` enum promoted into the core. It implements `KvAttention` by matching to the inner cache, as ADR 0008 and ADR 0009 describe. `RecurrentState` is an ordered map of named tensors plus an offset.

`SequenceStateBackend::SchedulerOwned` is the natural backend of a family that reports a table, and `SequenceCacheSet` holds `layer_state: Option<Vec<LayerState>>` for it. `CachePool::allocate_with_layout` builds the vector from `LanguageModel::make_layer_state()`, whose default reads the table, the injected KV modes and the lookahead-rewind capability.

The vector is keyed on the model's natural layout, not on the allocated backend. Under the paged decode override every family is allocated as `PagedKvCache` for block accounting (#1346), and a scheduler-owned sequence still carries its layer state there, with the paged table a length mirror exactly as it was for the model-owned layout.

`LayerState::new` is the only place that maps `Sliding` to a `RotatingKVCache`. Replacing that storage with window-bounded paged blocks (#3256) changes that constructor, the `Rotating` arms in `layer_state.rs`, and the `Rotating` arms of the snapshot code, and nothing in a model file.

### How a forward reaches the state

The trait's forwards take `&mut [KVCache]`, so the state cannot ride in that slice. `LanguageModel::forward_layer_state(request, rows)` is the forward of a scheduler-owned family: `rows[i]` is the layer state of row `i`, and the request carries the input, optional embeddings and mask, the optional last position, and the sequence ids for side state a model still keys by id (the Qwen MRoPE position tables). One row is a prefill piece or a decode step, several rows are a batched decode step or a prefill cohort.

The engine dispatches. `Engine::step`, `prefill`, `score`, `prefill_cohort` and `verify` run `forward_layer_state` over the pool's rows when the sequence holds layer state, and the `KVCache`-slice forward otherwise. The engine also does what the model hooks did for a model-owned family, over the layers directly: the pad trim (`LayerState::rewind_padded_prefill`), the decode rewind (`LayerState::rewind_decode_appends`) and the paged length mirror (`LayerState::visible_len`). `trim_state(Some)`, `rewind_decode_appends` and `sync_sequence_storage` are not called for a scheduler-owned sequence.

A migrated wrapper keeps one slot: the id-less fallback state a bare `forward` runs on (the CLI drafter targets, benchmarks, `mlxcel generate`'s speculative targets). It has no per-sequence map. A `KVCache`-slice entry called with a sequence id on a migrated wrapper panics with a message naming `forward_layer_state`, because that caller bypassed the pool and would otherwise run on the wrong state.

Rejected: making `KVCache` polymorphic over rotating, chunked and recurrent storage, so the existing slice could carry the state. ADR 0008 and ADR 0009 already route attention per storage through `KvAttention`; a second dispatch inside `KVCache` would duplicate it and put recurrent tensors behind an attention-cache type.

Rejected: keeping the by-id trait entries alive on a migrated wrapper through a per-sequence table in the wrapper. That is the model-owned layout under another name, with two owners of one state.

### Snapshot, restore, truncation and checkpoints, once

`server::prompt_cache::state_snapshot` implements all of them over `[LayerState]`:

| Layer kind | Whole snapshot | Truncation | Checkpoint |
|---|---|---|---|
| `Full` | every row | always | rebuilt from the donor, cut back |
| `Sliding` | the ring and its scalars | while the ring is unwrapped | the ring itself (#3255) |
| `Recurrent` | every slot | never | the slots themselves (#3047) |
| `None` | nothing | always | nothing, restores fresh |

No tensor is encoded there. The encoders are the ones the model-owned families use, selected by the family's `LayerStateSnapshotSpec`:

- `Windowed { standard, rotating }` is `models::kv_snapshot` and `models::kv_ring_checkpoint` under the family's names. Gemma 3 uses it.
- `Hybrid` is `models::hybrid_snapshot`, extracted from `Qwen3NextCache` so both callers share it. Qwen 3.5 uses it.

A snapshot written by the generic path therefore has the tensor names, shapes and scalars the family stored before, and a stored snapshot restores unchanged. Tests compare the generic path with the model-owned encoders tensor for tensor and pin the names.

A restore fills a fresh vector and installs it only when every layer validated: family tag, layer count, each layer's storage against the table, and the key and value tensors of each attention layer. A mismatch is a named error, the sequence is untouched, and the scheduler records `LayoutConstraints` and prefills cold, as before.

The scheduler calls `state_snapshot::{snapshot_sequence_state, restore_sequence_state, restore_sequence_state_truncated, snapshot_truncatable_to, state_checkpoint_from_snapshot, restore_sequence_state_from_checkpoint}` in place of the model hooks. Each takes the generic path for a scheduler-owned family and the model's hook for every other family.

### Ring checkpoints: one implementation, two callers

`models::kv_ring_checkpoint` (#3255) is the single implementation of the ring subset and restore. The generic path calls it for Gemma 3, whose per-family hooks are deleted. Gemma 4 and AFMoE stay model-owned in this change and keep their #3255 hooks, which call the same functions. Their migration removes the hooks.

### Reference migrations

Gemma 3 (full + sliding) and Qwen 3.5 text (full + recurrent) run on `SchedulerOwned`. Their VLM wrappers forward the table, the state builder, the snapshot vocabulary and `forward_layer_state`.

Gemma 3's forward is generic over a small trait that both the model-owned `Cache` (AFMoE, Kolibri-1, the pipeline stage, the backbone) and `LayerState` implement, so it is written once.

Qwen 3.5's decoder layers, batched decode, speculative verify and rollback are written over `Qwen3NextCache`, which Qwen3-Next and Qwen4-Exp share and still own. A Qwen 3.5 forward borrows the sequence's layer state as that typed view for its duration (`qwen3_5_layer_state::LentCaches` moves each cache out of the pool's vector and back when dropped). No tensor is copied and the state lives only in the pool between forwards. The MTP target adapter carries the borrowed layer state for a tick instead of a sequence id. When Qwen3-Next and Qwen4-Exp migrate, the typed view can become the storage and the lending step goes away.

Capabilities are unchanged for both families: `supports_paged_decode_backend()`, padded-prefill eligibility, the Gemma 3 decode lookahead rewind, batched decode and Qwen 3.5's batched prefill.

### Per-branch checkpoint cap

The per-conversation checkpoint cap (`--ctx-checkpoints N`) counted every branch below a re-inserted common ancestor together, so regenerating an earlier turn after a fork could evict a sibling branch's checkpoints. It now counts one root-to-leaf line at a time: a checkpoint is dropped when some line through the inserted checkpoint holds `N` longer checkpoints above it. A fork keeps up to `N` per branch, the result does not depend on insertion order, and the checkpoint byte budget remains the bound on the memory all branches hold together.

## Consequences

- A capability over sequence state is written once in `state_snapshot.rs` or `layer_state.rs` and every scheduler-owned family has it.
- A family is migrated by reporting its table, implementing `forward_layer_state`, naming its snapshot vocabulary and deleting its per-sequence map and snapshot hooks.
- `SequenceCacheSet::nbytes()` reports what it reported before: it does not count layer state, which was not counted while it lived in the model. `layer_state_nbytes()` reports it separately.
- Model-level tests can no longer forward a migrated family by sequence id on the bare wrapper. `models::pooled_model::Pooled` drives the model over a real `CachePool` and the same entry points the engine and the prompt cache use, behind the by-id calls those tests were written against.
- A scheduler-owned family with a recurrent or chunked layer cannot be pad-trimmed or decode-rewound by the engine. Qwen 3.5 declines both, as it did before.

## Follow-ups

One issue per remaining group, each migrating the group onto this layer:

1. Gemma 4 (text, VL, Unified) and AFMoE: remove the #3255 hooks. Gemma 4 needs the KV-shared `None` layers, the MTP hooks and the VLM per-layer-input side map.
2. The other full + sliding families: Kolibri-1, Muse Glimmer, MiMo V2 Flash, GPT-OSS, Ministral 3, ExaOne 4, Mellum, Laguna, Step 3.5, LIMITE. Most gain prefix reuse for the first time.
3. Llama 4: `Chunked`, with the chunk-boundary pad rewind.
4. The GatedDeltaNet and linear-attention hybrids: Qwen3-Next, Qwen4-Exp, Bailing MoE Linear, Kimi Linear, Kimi K3. Qwen4-Exp needs family-specific recurrent slots and its rollback log.
5. The SSM and convolution hybrids: LFM2, PLaMo 2, Falcon-H1, Granite MoE Hybrid, Jamba, Mamba, Mamba 2, Nemotron-H, RWKV-7, RecurrentGemma, Z1T, Inkling.
6. The families whose per-layer state is not one storage per layer: DeepSeek V4 (pooled state beside the ring) and IQuest-Coder Loop (two caches per layer).

The remote pipeline runtime keeps its layout: its state lives in other processes.
