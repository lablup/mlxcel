# Technical Report: PR #1752 - feat(core): share KV snapshot serialization across attention cache families

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Pre-merge (44 new tests, clippy and fmt green; three real checkpoints run against `mlxcel-server` on an M5 Max; the capability lands but is capped by a 512 MiB snapshot bucket a two-turn Llama 4 conversation already fills, filed as #1761)
**Languages**: Rust, Markdown
**Risk Level**: Medium (three families gain a cross-request state path they did not have, and two defects inside code Gemma 4 and Muse Glimmer already depend on are fixed as part of it; the stored tensor layout does not change)

---

## Executive Summary

Prompt-cache reuse in `mlxcel-server` has two producers. A family whose KV lives in the scheduler's `CachePool` gets radix prefix matching and partial-block adoption; a family that owns its caches per `SequenceId` gets only the exact-prefix `ModelStateSnapshot` copy, and only if it opts in through `LanguageModel::supports_snapshot_reuse()`. Opting in meant hand-writing the per-cache-type tensor and scalar layout, so only Gemma 4 and Muse Glimmer had done it. Gemma 3, AFMoE and Llama 4 held ordinary `KVCache`, `RotatingKVCache` and `ChunkedKVCache` state in `ModelOwnedSequenceState` and were excluded from every prompt-cache path, which means every multi-turn request on those families re-prefilled the whole conversation from token zero. This PR moves the layout into one module, `src/models/kv_snapshot.rs`, opts the three families in, and has Gemma 4 and Muse Glimmer delegate to the same code under the tensor names they have always written.

The module owns the truncation contract, not just the byte layout, because the truncation rule is the part a per-family copy gets wrong. `KVCache` is always truncatable, `RotatingKVCache` only while its ring is unwrapped, `ChunkedKVCache` only while its front is untrimmed. Two defects surfaced while getting Gemma 3 through a real conversation, and both were fixed as causes rather than worked around: `RotatingKVCache`'s `CacheInterface::live_len` sized the prefill mask from the physical ring buffer instead of the visible window, and `restore_fp16_snapshot_state` rejected the legitimate `idx > max_size` shape that an over-window first turn produces. The first is an MLX shape throw on the append after any restore; the second silently sent every over-window first turn down a cold prefill, which on a 4B Gemma 3 with a 1024-token window is most real conversations.

What ships is bounded by a follow-up that was deliberately left outside the diff. The snapshot bucket is still sized by the fixed 512 MiB default meant for recurrent state, which is O(1) in context, while an attention cache is O(context). Llama 4 Scout stores about 192 KiB per token, so the measured entries on the validation run were 384 MiB at 1817 tokens and 480 MiB at 2352. A third turn of that conversation cannot be stored at all, and the rejection is silent. That is #1761, and until it lands the feature works for short conversations and stops working for exactly the long ones it exists to accelerate.

---

## 1. What the opt-in cost before this branch

The gate is a single trait method consulted on both sides. Donation for a model-owned family returns early unless `supports_snapshot_reuse()` answers true, and lookup yields `SnapshotLookupOutcome::NoCandidate` before the store is consulted for the same reason. Neither path logs a decline that names the family, so a Gemma 3 operator watching `/v1/cache/stats` saw a working prompt cache with zero hits and nothing explaining why.

Answering true required four more methods and, behind them, a serializer per cache type. Gemma 4 carried one for `Standard` and `Rotating`, Muse Glimmer carried a near-identical copy under different tensor names, and no serializer for `ChunkedKVCache` existed anywhere in the tree. The two copies had already drifted apart in their error strings, and the `KVCacheMode` tag helpers existed twice, once privately in `gemma4.rs` and once privately in `recurrent_snapshot.rs`. Adding a third family meant a third copy of the same 150 lines, including a third copy of the truncation predicate, which is the one part where a wrong answer corrupts generation instead of failing.

---

## 2. The shared module

`src/models/kv_snapshot.rs` exposes four functions per cache type: `snapshot_*` writes the tensors and scalars under a `layer{idx}` prefix, `restore_*` reads them back, `*_truncatable_to` answers from the snapshot's own scalars whether a shorter restore is sound, and `truncate_*` performs it. The `kv_cache_mode_to_i32` / `kv_cache_mode_from_i32` pair moved out of `gemma4.rs` and was promoted to `pub(crate)` in `recurrent_snapshot.rs`, which already held a private duplicate. That is a small divergence from #1335's implementation plan, which put the pair in the new module; promoting the existing one removes a duplicate instead of creating a third home, and the error message lost its Gemma 4 label on the way.

### 2.1 Three cache types, three truncation rules

Truncation matters because a stored snapshot of N tokens can serve a request whose prompt is a proper prefix of those N, but only if the shortened cache is indistinguishable from a cold prefill of the shorter prompt. The three types answer differently, and the reason is geometric rather than a policy choice.

| Cache type | Truncatable when | Why, and how the truncation is performed |
|---|---|---|
| `KVCache` | always, for `0 <= target_len <= offset` | Every token keeps its own slot, so dropping the tail is a rewind. `KVCache::trim` rewinds `offset` logically and the physical buffer keeps its capacity for the next update to overwrite. |
| `RotatingKVCache` | `buffer_size == 0 && idx == offset && offset <= max_size` | Once the ring wraps, logical token `t` no longer sits at slot `t`. `RotatingKVCache::trim` rewinds only `offset` and `idx`, which is sound while slot and position agree and silently wrong once they do not, so `truncate_rotating` refuses a wrapped ring outright through `is_trimmable()`. |
| `ChunkedKVCache` | `start_position == 0` and `target_len <= offset` | After a front trim the buffer holds `[start_position, offset)`, while a cold prefill of `target_len` tokens would hold `[target_len - chunk_size, target_len)`. Whenever `start_position > target_len - chunk_size` the truncated restore attends over strictly fewer tokens than the cold run. The truncation itself is a physical slice, not a rewind: `ChunkedKVCache` has no `trim`, and `update_and_fetch` reads the buffer length back through `get_buffer_size`, so a logical rewind would leave the abandoned tail visible to the next growth decision. |

Exact-prefix restore is exempt from all three rules. A full-length restore reproduces the cache as captured, wrapped ring and trimmed front included, because nothing is being dropped. Only the shorter-prefix variant declines, which is why Gemma 3 keeps restoring across turn 3 with the ring long since wrapped (section 6.2) while refusing to truncate into that same wrapped ring (section 6.3).

One implementation detail carries the mixed-layer families. Each `*_truncatable_to` returns true when its own tensors are absent, so `Cache::snapshot_truncatable_to` can conjoin the rotating and standard predicates and have the conjunction reduce to whichever arm that layer actually stored. Llama 4 does the same with the chunked and standard pair. The alternative, reading a per-layer type tag back out of the snapshot, would have added a scalar to a format two families already ship.

### 2.2 The names parameter, and what it protects

Tensor names are part of a family's on-the-wire snapshot contract, and the two families that shipped before this module spell the same two cache types differently: Gemma 4 writes `standard` / `rotating`, Muse Glimmer writes `full` / `sliding`. A `KvSnapshotNames` value carries the two segment names plus a human-readable family label for error messages, so both keep producing byte-identical snapshots and both keep restoring snapshots written by earlier builds. `chunked` is a bare constant rather than a parameter, because Llama 4 is its only holder and has no prior spelling to preserve.

AFMoE reuses Gemma 3's `Cache` enum but passes its own `KvSnapshotNames`. The segment names match Gemma 3's, so the only difference is the label in the error string, and the snapshot's family tag already keeps the two apart at restore. Threading the label through costs one constant per family and stops an AFMoE failure from claiming to come from Gemma 3.

Backward compatibility runs one layer deeper than the names. Every scalar in `restore_rotating` falls back to the live cache's own value when the field is absent, and `snapshot_mode` defaults to `Fp16` when the tag is absent, because snapshots written before the tag existed were always FP16. A snapshot from an older build therefore restores rather than being rejected for a missing field.

### 2.3 Only FP16, at both ends

Quantized modes carry sidecar buffers the snapshot container does not model, so both snapshot and restore refuse anything that is not `KVCacheMode::Fp16`, and restore additionally requires the stored mode and the live cache's configured mode to agree. A refusal is an error string, and the caller falls back to a cold prefill rather than reinterpreting a quantized buffer as FP16. `ChunkedKVCache` has no quantized variant at all, so Llama 4's chunked layers are FP16 by construction and the quantized-mode request is warned about once at cache construction instead.

The window width is treated the same way in `restore_rotating`: `max_size` is model configuration, not sequence state, so a snapshot whose window differs from the live cache's is refused instead of silently re-opening the window. `restore_chunked` requires `chunk_size` to agree for the same reason.

---

## 3. Two defects found as causes

Neither was reachable from a live prefill-only sequence, which is why both had survived. The first sits in Gemma 3's own mask path, which AFMoE shares; the second sits in `mlxcel-core`, on the restore path every rotating snapshot family runs, Gemma 4 and Muse Glimmer included.

### 3.1 `live_len` sized the prefill mask from the physical ring

`RotatingKVCache`'s `CacheInterface::live_len` returned `seq_len()`, the physical buffer length, where the sliding prefill mask has to be sized from `visible_len()`. `update_in_place` grows the ring by `step` (256) blocks during decode, so a cache that has generated even one token holds `physical > offset`, while `update_concat` concatenates only `visible_len()` prior keys onto the new ones. Sizing the mask from the physical length made it wider than the returned K/V and tripped `broadcast_shapes` on the first multi-token append after a decode.

That shape is exactly a snapshot restore followed by an appended-token prefill, and nothing else in the server produces it: a live prefill-only sequence leaves `physical == offset` because concat does not over-allocate. The accessor is now `visible_len()`, and `live_len_matches_the_keys_a_multi_token_append_returns` pins the decode-grown case the pre-existing mask test could not see, since on a freshly concatenated cache the two values coincide.

### 3.2 The restore bound refused every over-window first turn

`restore_fp16_snapshot_state` rejected `idx > max_size` whenever `buffer_size == 0`. The intent was to catch a corrupt write position, and the bound looks right if a rotating cache never holds more than one window. It does: `update_concat` stores more than `max_size` after a prefill longer than the window and pins `idx` to the stored length, so a snapshot captured between that prefill and the next single-token step legitimately carries `idx == physical_len > max_size`.

The old check therefore sent every over-window first turn down a cold prefill, with the restore reported as a rejection rather than a bug. On `models/gemma-3-4b-it-4bit` the sliding window is 1024 and the validation conversation's first turn is 1323 prompt tokens, so the shared serializer would have been dead on the family it was written for. The bound is now `idx <= physical_len`, which still catches a genuinely corrupt `idx` and is the tighter of the two checks while the buffer is still growing toward `max_size`. Guarded by `rotating_round_trip_survives_an_over_window_prefill`.

---

## 4. Two scoped declines

### 4.1 Padded prefill

Gemma 3, AFMoE and Llama 4 answer `supports_padded_prefill()` with `false`, and Gemma 4's existing opt-out gains the same note. The reason is a missing hook rather than a property of these models. The contract requires the caches to be trimmed back to the real prompt length after an NA tile-aligned padded chunk; the scheduler honours it by trimming the `CachePool`'s `Vec<KVCache>`; and a `model_owned` family's pool entry is `SequenceCacheSet::model_owned`, whose `caches` vector is empty. The trim reaches nothing, the pad positions stay in the model's own caches, and `offset` runs ahead of the real token count by the pad width. The path is gated by `should_align_prefill()`, so the defect is M5-only.

The consequence is worse for a snapshot family than for anyone else, and that is why the opt-out arrives with this branch rather than separately. A snapshot is keyed on a token vector, so the cached state has to hold exactly those tokens and no others. A padded prefill leaves state that does not correspond to its own key, which is not a quality regression but a wrong-answer path.

The general repair is a sequence-aware trim hook the scheduler can call for model-owned families, filed as #1755. `trim_internal_caches` cannot serve: it takes no `SequenceId` and is wired only into the CLI generate paths, so it cannot address one sequence's state inside a batched server. #1755 also names the three model-owned families that still leave the default at `true` and carry the same latent defect without a snapshot to corrupt: `deepseek_v4`, `bailing_moe_linear` and `qwen3_next`.

### 4.2 The whole-entry gate on the snapshot branch

`try_adopt_cached_prefix` enforced `require_whole_entry` only on its KV branch. The snapshot branch runs first and returns before the KV branch is reached, so a truncating snapshot restore was reachable for a multimodal request, and `admission.rs` drops the prepared VLM embeddings once `prefill_start_offset > 0`. Placeholder tokens left in the suffix are then forwarded as ordinary token ids.

The gap predates this branch. `vision::gemma4_vl` and `vision::gemma4_unified` have forwarded `snapshot_truncatable_to` since Gemma 4 first answered it, so a multimodal request could already reach a truncating restore with the whole-entry rule never consulted. What this PR changes is the reach: forwarding the five snapshot hooks through the shared `vision::VisionLanguageModel` brings the Gemma 3 and Llama 4 VLM checkpoints onto the same path. The gate now sits on both branches, declines before allocating a sequence slot so the entry stays available for a later exact match, and records the same `PromptCacheRejectReason::ModeMismatch` the KV branch records. An exact-prefix match is unaffected, since it leaves `partial` false and no placeholder tokens in the suffix. This closes #1756.

The wrapper forwards rather than implements, and the doc comment records why that is sound: `VisionModule` is an encoder, a connector and a processor, all stateless across requests, and image embeddings are recomputed into the prompt before every forward, so a restored text-model snapshot reconstitutes the whole of what the wrapper carries between requests. A media payload cannot leak across a restore either, because the prompt-cache key folds in the request's multimodal digest, so a text-only turn and the same tokens with an image land in different buckets.

---

## 5. Restore is untrusted input

A `ModelStateSnapshot` reaching `restore_*` came from this process, but it came from a different request and a possibly different model configuration, and the branch treats it as input to validate rather than state to install. The rotating path gets most of this for free, since `restore_fp16_snapshot_state` checks the ring geometry against the buffers it is handed. The full-attention and chunked paths assign their scalars directly, so the same class of check lives in `check_restored_buffers`.

| Refusal | What it catches |
|---|---|
| keys present without values, or the reverse | a half-written or half-read entry, at snapshot and at restore |
| either buffer not rank 4 | anything that is not the `[B, H_kv, T, D]` an attention cache is indexed as |
| keys and values disagreeing on batch, heads or length | two buffers from different captures |
| declared length negative, or greater than the buffer's `T` | a state whose `offset` runs past its own buffer |
| `start_position < 0` or `offset < start_position` | an inverted chunked window |
| `max_size` or `chunk_size` differing from the live cache | a snapshot captured under a different model configuration |

The declared-length check is the one with teeth. Without it, a state whose `offset` runs past its own buffer installs cleanly, and the next append slices past the end of the sequence axis, because `KVCache::update_fp16` normalizes to `buffer_idx()` before it grows. That surfaces as an MLX throw at the FFI boundary rather than as a declined restore. The chunked variant checks `offset - start_position` rather than `offset`, since a chunked buffer holds only the untrimmed part of the window.

Two ordering properties matter as much as the checks. Validation happens before assignment, so a snapshot that fails leaves the freshly built cache untouched and the caller falls back to a cold prefill. And `truncate_chunked` builds both slices before installing either, so a failure on the value slice cannot leave the cache holding a shortened key buffer beside a full-length value buffer. That one is pinned by `chunked_truncate_does_not_shorten_keys_when_the_value_slice_fails`, which is the kind of guarantee that is easy to state in a commit message and easy to lose in a later edit.

`restore_sequence_state_truncated` re-checks `snapshot_truncatable_to` rather than trusting the caller on all three families. Installing a partially truncated state corrupts generation silently; an error here costs a cold prefill.

---

## 6. Validation

Real checkpoints on an Apple M5 Max, `mlxcel-server` on port 19335, prompt cache on, greedy (`temperature 0`, `seed 0`), with `preserve_thinking` pinned in every arm.

### 6.1 The comparison that does not work

The obvious test, prompt cache on against `--no-prompt-cache`, does not isolate the snapshot and is not what was used here. With the cache on, `capture_history_boundary_snapshot` (#1143) splits even a cold first turn at the conversation boundary. Gemma 3's 1323-token turn 1 is forwarded as 1320 then 3, where the cache-off arm forwards 512/512/299. Per `docs/benchmarks.md`, the forward width selects which quantized-matmul kernel MLX dispatches, and that width change reproducibly flips one near-tied token in an 89-token reply, a straight apostrophe against a curly one.

Two controls pin the cause to segmentation rather than to the snapshot. Reruns of the same arm in separate processes are bit-identical, so the flip is not run-to-run noise. And `models/qwen2.5-7b-instruct-4bit`, which has no snapshot support and therefore takes no boundary split, is byte-identical across the flag. A comparison that varies the cache flag is varying prefill segmentation at the same time, and a difference it reports cannot be attributed.

The substitution is to hold the cache on in both arms and vary only whether the turn is served from a restored snapshot or computed cold in a fresh process. That keeps prefill segmentation matched and leaves the restore as the only difference.

### 6.2 Warm restore against cold

**Gemma 3** (`models/gemma-3-4b-it-4bit`, rotating 1024 plus standard). Turn 1 is 1323 prompt tokens over a 1024-token window, the over-window shape section 3.2's fix admits, and generates 89. The completion donate stores `token_len=1412`, exactly turn-1 prompt plus generated, which the warm-up extends to the turn-2 boundary at 1417. Turn 2 adopts `restored 1417/1435, stored=1417, partial=false`; turn 3 adopts `restored 1547/1567, partial=false` with the ring long since wrapped, which is the exempt case from section 2.1 exercised on real weights.

| Turn | Restored | Against cold |
|---|---|---|
| 2 | 1417 of 1435 | byte-identical over 104 of 104 tokens, 103 with bitwise-equal logprobs, minimum top-2 gap 0.25 |
| 3 | 1547 of 1567 | byte-identical over 18 of 18 tokens, every logprob bitwise equal |

**Llama 4 Scout** (`models/llama-4-scout-17b-16e-instruct-4bit`, chunked plus standard). Turn 1 is 1305 prompt tokens and 512 generated; the completion entry is `1817 = 1305 + 512`, extended to 1822. Turn 2 adopts `restored 1822/1840, partial=false` and is byte-identical to cold over all 512 tokens, 338 of them with bitwise-equal logprobs. The minimum top-2 gap over the run is 0.0, so the restored arm agreed with cold even at exactly tied positions, which is the strictest form this comparison takes.

**AFMoE** (`models/trinity-nano-preview-4bit`). Turn 1 is 1281 prompt tokens and 512 generated, and the completion entry is `1793 = 1281 + 512`. Turn 2 adopts `restored 1280/1755, stored=1793, partial=true`: the reply does not re-tokenize as a prefix of the stored tail, so the store truncates to the end of turn 1's prompt. That is the truncation predicate answering on real weights rather than on a synthetic cache, which is the one thing the unit tests cannot supply. Output identity is not a usable signal for this checkpoint: `docs/supported-models.md` records it emitting degenerate repetitive text under the mlx-lm reference as well as here, and both arms do exactly that. Its snapshot machinery is covered by unit tests instead, including a non-ignored `snapshot_restore_matches_cold_decode`.

### 6.3 Declining correctly

The contract also has to refuse. Replaying Gemma 3's 1323-token prompt finds the 1417-token entry, refuses to truncate into a wrapped ring, and rejects with `snapshot_diverged (context_len=1323, entry_len=1417)` instead of restoring a wrong window. A predicate that only ever answers true is not evidence that it is checking anything, and this is the run that separates the two.

### 6.4 Gates

| Gate | Result |
|---|---|
| `--lib models::kv_snapshot` | 23 passed |
| `--lib models::gemma3` | 25 passed, 6 more under `--ignored` |
| `--lib models::llama4` | 5 passed, 3 more under `--ignored` |
| `--lib models::afmoe` | 28 passed |
| `--lib server::batch::scheduler::scheduler_model_owned_cache_tests` | 3 passed |
| `--lib models::gemma4_tests`, `--lib models::muse_glimmer` | 10 and 13 passed, unchanged by the delegation |
| `cargo clippy --lib --tests --features metal,accelerate`, `cargo clippy -p mlxcel-core --lib` | clean at `-D warnings` |
| `cargo fmt --all -- --check` | clean |

The model-level snapshot tests carry `#[ignore = "requires serial MLX execution"]` because they build a synthetic wrapper and run a forward; each was run alone by exact name and passes. The PR body's test table was written against an earlier commit and reports 16 for `models::kv_snapshot`; the final commit added seven more, and 23 is the count in the tree.

---

## 7. What limits the feature, and what is not established

### 7.1 The snapshot bucket is sized for recurrent state (#1761)

`snapshot_family_is_model_aware` lists only recurrent and hybrid families, and none of `gemma3`, `afmoe` or `llama4` classifies as `Hybrid` or `PureSsm`, so `recommend_model_snapshot_capacity_from_config` returns `None` and all three fall back to `DEFAULT_SNAPSHOT_CAPACITY_BYTES`, a fixed 512 MiB. A fixed bucket is the right policy for a recurrent family, whose state is O(1) in context. It is the wrong policy for an attention-cache family, whose snapshot is O(context). Past the bucket an entry is rejected `Oversized` and the prompt cache silently does nothing.

Llama 4 Scout stores roughly 192 KiB per token (48 layers, 8 KV heads, head dim 128, fp16, K and V), which puts the ceiling at about 2731 tokens. The entries measured on the run above were 402,653,712 bytes for the 1817-token turn and 503,317,008 for the 2352-token one, which is 384 MiB and 480 MiB of step-aligned buffer. The two-turn validation sits just under the cap by 6 percent and a third turn would not fit. Gemma 3's entries grew 188,253,128 bytes at 1320 tokens to 341,346,192 at 1412, and one turn stores two entries (a `Boundary` one and a `Completion` one), so a three-turn conversation already pushes the bucket into continuous eviction.

The same issue records a second defect in that function: `model_type()` prefers `text_config.model_type`, and the checkpoints under `models/` report `gemma4_text`, `gemma4_unified_text` and `muse_glimmer_text`, none of which match the `gemma4` or `muse_glimmer` entries in the list, so the model-aware path is dead code for its existing entries too. It was left outside this diff because it changes a default memory policy and needs its own measurement rather than an assertion.

This is the honest statement of what merges: three families gain exact-prefix reuse, and on Llama 4 it is usable below roughly 2700 tokens of conversation and silently absent above it.

### 7.2 A whole-prompt hit re-runs the last token (#1760)

Replaying an identical prompt inside one process diverges from the first reply on every family tested. The adopt covers the whole prompt, and `admission.rs` backs the prefill cursor off one token so the sampler sees fresh logits, without rewinding the restored cache. The last prompt token is therefore forwarded onto a cache that already contains it.

This is pre-existing rather than introduced here, and the control is `models/gemma-4-12b-it-4bit`, which has shipped this path since before the branch, is untouched by it, and reproduces the divergence identically. #1760 keeps two explanations open (a genuinely duplicated token, or the forward-width effect of section 6.1) and names the measurement that separates them: assert the per-layer cache offset against `prompt_tokens.len() - 1` right after the clamp. What this branch changes is the size of the affected set, since three more families can now reach the path.

### 7.3 The off-by-one in #1754 does not reproduce

#1754 claims completion-origin snapshots donate `token_len = tokens.len()` while the caches hold one fewer, because the decode step forwards the previously sampled token and the final one is never fed back. The issue scopes the gap to non-EOS finishes, `Length` among them.

Llama 4's turn 1 above finished on `length`, which is exactly that class, and the restored turn 2 was still byte-identical to cold over 512 of 512 tokens with a minimum top-2 gap of 0.0. If the stored state held one fewer token than the entry claims, every suffix token would sit one RoPE position low and the restored run would have to diverge. Gemma 3 is consistent with the same reading. The counter-evidence is posted on the issue, along with the most likely reconciliation: the adopted entry was the 1822-token warm-up entry rather than the 1817-token completion entry, so the warm-up extend at `prompt_cache.rs` may be re-synchronising before the adopt. The recommendation on the issue is to assert `snapshot.token_len()` against the per-layer cache offset at capture before changing the donate arithmetic, because a fix aimed at an off-by-one that is not there would introduce one.

---

## 8. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 18 |
| Lines added | 3285 |
| Lines removed | 463 |
| Tests added | 44 (23 module, 8 Gemma 3, 7 Llama 4, 5 AFMoE, 1 scheduler) |
| Net new families with prompt-cache reuse | 3 text, plus their VLM wrappers |

### Changes by area

- `src/models/kv_snapshot.rs` (new, 784 lines): the twelve serializer entry points, the `KvSnapshotNames` vocabulary, and the shared validators (`check_restored_buffers`, `restored_rank4_shape`, `check_truncate_target`, `slice_leading_tokens`).
- `src/models/gemma3.rs`, `src/models/afmoe.rs`, `src/models/llama4.rs`: four cache-level methods delegating to the module, the five `LanguageModel` hooks, and the `supports_padded_prefill() -> false` opt-out. Gemma 3 also carries the `live_len` fix.
- `src/models/gemma4.rs` (41+/252-), `src/models/muse_glimmer_cache.rs` (20+/143-): the hand-written serializers deleted in favour of calls into the module, tensor names preserved through their `KvSnapshotNames` constants.
- `src/lib/mlxcel-core/src/cache.rs`: the `restore_fp16_snapshot_state` idx bound.
- `src/vision/mod.rs`: the five hooks forwarded to the text model, with the statelessness and multimodal-digest reasoning recorded.
- `src/server/batch/scheduler/prompt_cache.rs`: the whole-entry gate moved ahead of the sequence allocation and applied to both branches.
- `src/models/recurrent_snapshot.rs`: the `KVCacheMode` tag pair promoted to `pub(crate)` with a family-neutral error message.
- `docs/turbo-kv-cache.md`, `docs/supported-models.md`: the snapshot-capable roster, the per-type truncation rules, the padded-prefill exclusion, and a `grep` line so the roster can be regenerated rather than trusted.

### Commits

| Hash | Type | Subject |
|---|---|---|
| `2dde058` | fix | bound the rotating snapshot idx by the physical buffer |
| `a40d421` | feat | share KV snapshot serialization, opt in three families |
| `ec96fa3` | fix | apply the whole-entry policy to the snapshot adopt branch |
| `9da46f4` | fix | bound KV snapshot restores by their own buffers |
| `3a5e506` | test | the model-owned gate test now expects a snapshot donation |
| `7647c23` | test | close snapshot validation and family-hook gaps for #1335 |
| `a0f9692` | docs | note snapshot prompt-cache reuse for #1335 families |

### Related issues

Closes #1335 and #1756. Follow-ups opened from this branch's review: #1754 (donate arithmetic, with counter-evidence posted), #1755 (sequence-aware trim hook, which would return padded prefill to these families), #1760 (whole-prompt hit), #1761 (snapshot bucket sizing). Depends on #1143 for the history-boundary snapshot that makes a hit possible at all, and on #1145 for the truncating-restore path the truncation predicates answer for.

---

## 9. Follow-up

**#1761 gates the value of this PR.** Everything else here is correctness work on a path that already existed; #1761 decides whether the path is reachable for a conversation worth caching. It should be measured on Llama 4 Scout and Gemma 3 rather than asserted, since it changes a default memory policy, and the `_text` suffix defect it names means the existing entries need a test that uses the spellings checkpoints actually carry.

**The padded-prefill opt-out is a workaround with an expiry.** #1755 asks for at least one of the three families opted out here to be restored to padded prefill through the new hook, which is what would prove the hook works end to end rather than only existing.

**Three model-owned families still default to `true`.** `deepseek_v4`, `bailing_moe_linear` and `qwen3_next` carry the same latent trim defect without a snapshot to corrupt. They are named in #1755 and should be either fixed or opted out explicitly.

**The same `seq_len()` claim survives one file over.** `src/models/exaone_moe.rs`'s `AnyKVCache::live_len` returns `seq_len()` for a rotating cache, under a comment asserting that `seq_len()` already reports the live window, which is the assertion section 3.1 disproved. That family owns no snapshot, so nothing there produces a multi-token append after a decode today, and the guard added here covers only Gemma 3's copy. Worth reading rather than asserting either way.

**The scheduler gate test lost a counter assertion.** `prompt_cache_reject_model_owned_state` counts the donate-side decline, and Gemma 3 now returns on the snapshot branch before reaching it. The guarantee it guarded is still asserted directly as an empty K/V bucket, but the counter itself now has no model in that test file exercising it, which is worth remembering the next time it is trusted as coverage.

### Transferable lesson

Two of the defects here were found by running the feature to completion on a real conversation, not by reading the code that was being changed. The `live_len` mask width and the `idx > max_size` bound had both been correct for every shape the server could previously produce, and both became wrong the moment a restored cache was appended to. A shared abstraction extracted from two working copies inherits their blind spots along with their behavior, and the only thing that surfaces those is a caller shaped differently from the two the copies were written for.

The validation design is the other half of it. The first comparison anyone reaches for, feature flag on against feature flag off, was invalid here because the flag also changes prefill segmentation, and the way that showed up was a single apostrophe flipping in an 89-token reply. The fix was not a tolerance; it was a different control, holding the flag constant and varying only the restore. A test that varies two things and reports a difference has told you nothing about either.
