# Technical Report: PR #1580 - fix(phi3): select the LongRoPE table by whole-prompt position

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Implemented and unit-verified; real-checkpoint token parity above and below 4096 tokens is still outstanding
**Languages**: Rust, Markdown
**Risk Level**: Medium (changes the rotary frequencies every `longrope` Phi checkpoint uses below its trained context, and adds a cross-crate prefill hint consumed inside the attention hot path)

---

## Executive Summary

Phi-3.5, Phi-4-mini and Phi-4-multimodal declare `rope_scaling.type` as `longrope` (older conversions spell it `su`) and ship two per-dimension factor lists. The trained model rotates with `short_factor` while the sequence fits in `original_max_position_embeddings` and with `long_factor` above it. mlxcel built one table from `long_factor` and used it at every position, so on `mlx-community/Phi-3.5-mini-instruct-4bit`, where `long_factor` reaches 64.8 on the low-frequency pairs while `short_factor` stays under 2.9, every prompt inside the trained 4096-token context was rotated with the wrong table.

The fix keeps both tables, pairs each with its own attention-magnitude scale (`short_mscale` / `long_mscale` when present, the default `sqrt(1 + ln(M / L) / ln(L))` otherwise), and resolves the top-level `original_max_position_embeddings` the shipped configs actually carry. The part that is not in the issue: the selection cannot be made from one forward pass's `offset + seq_len`, because mlxcel prefills a long prompt in chunks and that rule splits a single prompt across both tables. A new `mlxcel_core::prefill_span` carries the total prompt length from the driver that splits the prefill to the model that needs it.

---

## 1. Problem Statement

### 1.1 Background

`configure_su_rope` in `src/models/phi3.rs` read `scaling.long_factor`, built `freqs[i] = long_factor[i] * base^(2i/d)` from it, and stored one `su_rope_freqs` array on every `Phi3Attention`. `short_factor` was declared on the `RopeScaling` struct and had no read site anywhere in the tree. That single table fed both the fused quantized path (`forward_fused_qkv_split_su_scaled_rope`) and the graph fallback (`prepare_qkv_with_rope`), so both were consistently wrong rather than inconsistently wrong.

The attention-magnitude scale was always the default formula, computed from `scaling.original_max_position_embeddings` with a hardcoded fallback of 4096. `models/mlx/phi-3.5-mini-4bit/config.json` puts `original_max_position_embeddings` at the top level and not in the block, so the value in use was the fallback and was correct only because 4096 is what that checkpoint declares. A checkpoint declaring anything else at the top level would have been silently read as 4096. `short_mscale` and `long_mscale` were not fields on the struct at all, so a Phi-4 config carrying them had them dropped by serde.

`src/models/phi4mm.rs` aliases `phi3::ModelArgs` and `phi3::Phi3Model`, so Phi-4-multimodal's text decoder inherited all of this, as did Phi-3 Vision and the Phi-4 SigLIP VLM through the same decoder.

### 1.2 Existing Issues

- **Every short prompt was rotated with the long table.** On Phi-3.5-mini the two lists diverge by more than an order of magnitude on the low-frequency pairs, so this is not a rounding-level difference; it is the wrong positional encoding for the regime the model was trained in.
- **The defect is invisible to the usual smoke test.** A wrong-but-finite rotary table still produces fluent text, and a six-token greedy prompt cannot separate the two tables. Only a token-level comparison against an implementation that selects by position exposes it.
- **mlx-lm cannot be used as the oracle for the fixed behavior.** Its `SuScaledRotaryEmbedding` takes `short_factor` and `short_mscale` as constructor arguments and then builds `self._freqs` from `long_factor` and `self.scale` from `long_mscale` alone. mlxcel's pre-#1358 behavior therefore matched mlx-lm exactly, which is presumably how it survived review; matching mlx-lm here means not matching the checkpoint.
- **A verbatim reading of the issue's rule would have shipped a worse bug than the one it fixes.** See 2.1.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Chunked prefill splits one prompt across both tables, mixing rotations in one KV cache | Critical (output degenerates within a few tokens) | Certain, if the selection is made per forward pass |
| A later contributor "corrects" mlxcel back toward mlx-lm's long-table-only behavior | High (silently reintroduces the original defect) | Moderate, since mlx-lm is the reference for most of this tree |
| A driver outside the two patched ones chunk-prefills a `longrope` model without announcing the total | High | Low today (only two such drivers exist), and the reader's `max` fallback bounds it |
| Decode crossing `L` mid-generation diverges from transformers | Low (documented, and outside the trained 4096-token context) | Certain by construction |

---

## 2. Technical Review

### 2.1 Root cause, and the correction to the issue's specification

The issue specified selecting per forward pass:

```
position_end = o + s
use_long     = position_end > L
```

That is exactly what HuggingFace transformers computes (`seq_len = torch.max(position_ids) + 1`, then `seq_len > self.original_max_position_embeddings`), and it is correct there because transformers prefills a whole prompt in one pass. mlxcel does not. `src/lib/mlxcel-core/src/generate.rs` sets `DEFAULT_PREFILL_CHUNK = 2048` for the single-sequence CLI and bench path, and the server's batch scheduler splits a prefill across ticks with its own `--prefill-chunk-size`.

Under the per-pass rule, a 5136-token prompt at the default chunk size resolves as:

| Chunk | offset | len | `offset + len` | Table |
|-------|--------|-----|----------------|-------|
| 1 | 0 | 2048 | 2048 | short |
| 2 | 2048 | 2048 | 4096 | short |
| 3 | 4096 | 1040 | 5136 | long |

The KV cache then holds keys rotated with two different tables, and the measured greedy output degenerates into repetition within a few tokens. The correct rule is that the whole prompt decides: every chunk of a prompt longer than `L` uses the long table, every chunk of a shorter prompt uses the short table.

### 2.2 Where the total prompt length can come from

The attention layer sees only `(x, cache, mask)` and the cache's `offset`; nothing in that signature carries the future. Four mechanisms were considered:

1. **A field on `KVCache`.** Per-sequence and exactly the right lifetime, but `KVCache` is the shared cache type for every model in the tree, with many constructors, a detach/adopt serde path and a paged backing. A new field there is a large blast radius for one family.
2. **A `LanguageModel` trait hook.** Additive and idiomatic for this tree (`after_prefill`, `trim_internal_caches`, `prepare_sequence_state` all exist), but it needs a matching set/clear discipline, and the natural clear point (`after_prefill`) fires only at the end of a whole prefill. In the server that leaves the hint set across the decode ticks the scheduler interleaves between two chunks, which is precisely the leak that has to be avoided.
3. **Opting `Phi3Model` out of chunked prefill.** Correct for the CLI and a real memory regression: a 100k-token Phi prompt would then be prefilled in one pass. The server's chunking decision does not consult `supports_chunked_prefill` either, so it would need its own change.
4. **A thread-local announcement with an RAII guard, in `mlxcel-core`.** Chosen.

### 2.3 Compatibility and dependencies

`Phi3Attention`'s `su_rope_freqs`, `su_rope_scale` and `su_rope_scale_arr` fields were public but have no reader outside `src/models/phi3.rs`, so replacing them with one `su_rope: Option<SuRope>` breaks nothing. `src/loading/vlm_special.rs` and `src/model_metadata.rs` reference `phi3::ModelArgs` only, which gains a defaulted `Option` field and is unaffected. `src/lib/mlxcel-xla/src/emitter/config.rs` reads `rope_scaling.long_factor` out of raw JSON on its own and is untouched.

### 2.4 Code quality

`SuRope` owns both tables and the threshold, so "both tables exist or neither does" is structural rather than a pair of `Option`s that could drift apart. `Phi3Attention::forward` resolves the table once and passes the same `&SuRopeTable` to the fused and graph paths, so the two cannot disagree about which table a pass used; the previous shape passed the frequencies and the scale as two separate arguments to each path.

---

## 3. Technical Decisions

### 3.1 A thread-local RAII announcement rather than a trait hook

`mlxcel_core::prefill_span` holds `Cell<Option<i32>>` in a `thread_local!` and hands out a `PrefillSpan` guard that restores the previous value on drop. Three properties made this the right shape:

- **The lifetime is the forward call, not the prefill.** The server scheduler runs other sequences' decode batches between two chunks of one prompt, so a hint that outlives a single chunk's forward would hand another sequence this prompt's length. A guard scoped to the `let logits = { ... }` block expresses that directly; a trait hook would need the same discipline without the compiler enforcing it.
- **A thread-local is the right width.** `KVCache` is deliberately neither `Send` nor `Sync`, so a model forward always runs on the thread that drives it.
- **It needs no per-model plumbing.** The reader is in the root `mlxcel` crate and one setter is in `mlxcel-core`, so the state has to live in `mlxcel-core`; from there both the CLI driver and the server driver reach it without a new trait method on `LanguageModel`, `LoadedModel`'s delegation macro, and the VLM wrapper.

The reader takes `max(announcement, offset + seq_len)` rather than trusting the announcement outright. The two agree on every correct announcement, and the maximum means a driver that under-announces cannot make a long sequence look short, only the reverse.

### 3.2 Flip the table in place on decode, rather than re-encoding

When generation crosses `L`, transformers drops the KV cache and re-encodes the whole sequence with the long table (`prepare_inputs_for_generation`, the branch on `input_ids.shape[1] >= original_max_position_embeddings + 1` and `past_length <= original_max_position_embeddings`). This change flips to the long table at `offset + 1 > L` and leaves the short-table keys already in the cache untouched. That is what the issue asked for and is far cheaper, but it is an approximation, not parity. It is recorded in `docs/supported-models.md` and in the `SuRope::table_for` doc comment, with the explicit instruction not to claim transformers parity for a generation that crosses `L` mid-decode.

### 3.3 A `long_factor`-only block keeps its old behavior

Nothing in the tree ships such a config, but the construction path falls back to `long_factor` for both tables when `short_factor` is absent or too short. That keeps any such checkpoint bit-identical to its pre-change behavior instead of changing it as a side effect.

---

## 4. Implementation Details

### 4.1 The two tables

```rust
pub struct SuRopeTable {
    pub freqs: UniquePtr<MlxArray>,   // factor[i] * rope_theta^(2i / rope_dims)
    pub scale: f32,                   // short_mscale / long_mscale, or the default
    scale_arr: Option<UniquePtr<MlxArray>>,  // None when scale == 1.0
}

pub struct SuRope {
    short: SuRopeTable,
    long: SuRopeTable,
    original_max: i32,
}
```

`SuRope::from_args` returns `None` unless the block names `longrope` or `su` and carries a `long_factor` at least `rope_dims / 2` long, which is the same guard the old code applied; a truncated list still builds nothing rather than a half-filled table.

### 4.2 Selection

```rust
fn table_for(&self, offset: i32, seq_len: i32) -> &SuRopeTable {
    let pass_end = offset.saturating_add(seq_len);
    let span =
        mlxcel_core::prefill_span::current().map_or(pass_end, |total| total.max(pass_end));
    if span > self.original_max { &self.long } else { &self.short }
}
```

`Phi3Attention::forward` calls this once and hands the result to both paths.

### 4.3 Announcement sites

`mlxcel-core`'s `chunked_prefill_last_logits` announces `prompt_tokens.len()` once around its chunk loop; nothing else runs inside that function.

On the server the announcement is made once per scheduler entry point that runs prefill work for one sequence, not once per forward, through the `announce_prefill_span(&seq)` helper on `BatchScheduler`. The guard is function-scoped, which is the correct lifetime: the scheduler runs other sequences' decode batches between two chunks of one prompt, but everything inside one of these calls belongs to one sequence. The sites are `execute_full_prefill`, `start_chunked_prefill`, `continue_chunked_prefill`, `capture_history_boundary_snapshot` and `run_next_prompt_cache_warmup`. `run_padded_batched_prefill` is prefill and deliberately does not announce, because its pass starts at offset 0 and spans the longest row of the cohort, so the pass geometry already answers the question for that row and one scalar cannot say anything different for the shorter ones.

The CLI's unchunked branch announces nothing either, because its own `offset + seq_len` already equals the sequence length.

### 4.4 The first attempt announced in the wrong places, and only the server caught it

The first version of this change announced inside the two chunked-prefill functions only, at the `let logits = { ... }` block that runs one chunk. Every CLI gate passed: the 4-bit checkpoint matched a position-selecting oracle 30 out of 30 tokens on both a 1078-token and a 5136-token prompt, the bf16 checkpoint matched an f16-cast oracle, phi-3-mini was unchanged, and `MLXCEL_PREFILL_CHUNK=0` equalled 2048. The server returned repetition for the same 5136-token prompt through `POST /v1/completions`, because the prompt cache is on by default and reaches the model through two more forwards the fix never touched. `capture_history_boundary_snapshot` forwards `prompt_tokens[start..boundary]`, a strict prefix that can end below the threshold while the prompt does not. `run_next_prompt_cache_warmup` forwards the delta after restoring a snapshot, where the restored prefix is not reflected in the scheduler's `KVCache::offset` for a non-batching family.

The response is the function-scoped helper above, which covers every forward an entry point can reach rather than the ones the author happened to look at, plus `src/server/batch/scheduler/prefill_span_coverage_tests.rs`: a source-level guard that classifies every model forward under `src/server/batch/scheduler` as prefill or decode and fails when a forward turns up in a function the table does not name.

### 4.5 Cross-request KV reuse is a third place the table has to agree

Announcing the prefill span covers every path that encodes one prompt. It does not touch the path that restores KV encoded by a different one. The server's radix prefix cache found that gap on the next round: a 1078-token request encoded its prompt under the short table and donated it, and a 5136-token request whose prompt starts with the same tokens hit `cached=1056/5136` and prefilled only the remaining 4080 under the long table. A third identical request hit `cached=5136/5136` and decoded from the poisoned entry.

A position-selected table makes a cached prefix valid only for requests on the same side of `original_max_position_embeddings`, in both directions. `LanguageModel::rope_table_regime(total_prompt_len) -> Option<u8>` is the hook, defaulting to `None` for every model whose rotation does not depend on the sequence length. `Phi3Model` answers it by reading layer 0's tables rather than a copy of the config, so the cache's tag cannot drift from what the layers rotate with.

The tag joins bucket identity in all three places a lookup can reach an entry: the `PromptCacheKey` digest (domain separator bumped to v3), `BucketKey` for the per-session prefix scan, and `SessionlessBucketKey`, the radix trie's index key and the one the regression actually went through. The span the scheduler asks about is the announced prefill span, not the length of the vector being stored, so `insert_model_state_snapshot` takes an explicit `encoded_span`: a 3000-token boundary snapshot cut out of a 5136-token prompt was rotated with that prompt's table. An entry whose generation crossed the boundary mid-decode spans both tables and is declined outright, counted as `PromptCacheRejectReason::RopeRegimeMismatch`. A miss falls back to a full prefill under the request's own table, which is what transformers does by resetting its cache.

The cross-node disaggregated handoff needs no tag: it carries no prompt-cache state and moves one request's KV from its prefill node to its decode node rather than across requests.

### 4.6 Scale application

Both paths multiply the rotary prefix of Q and K by the table's scale before rotation, which is where the old code applied `su_rope_scale`. Upstream scales `cos` and `sin` instead; RoPE is linear in its input, so the two are the same map, and scaling the input costs one multiply on a shorter tensor.

---

## 5. Validation

| Command | Result |
|---------|--------|
| `cargo test --profile test-fast --features metal,accelerate --lib models::phi3::tests` | 13 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib server::batch::scheduler::prefill_span_coverage` | 3 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib server::batch::scheduler_prompt_cache_tests` | 30 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib server::prompt_cache` | 173 passed |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib prefill_span` | 5 passed |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib generate::tests` | 36 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib models::rope_utils` | 28 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` (both crates) | clean |
| `cargo fmt --all -- --check` | clean |
| `python3 scripts/ci/check_cross_repo_refs.py` | OK |

The phi3 tests cover: both tables' frequency construction against the closed form; the threshold switch at `L` for `(0, L)`, `(0, L + 1)`, `(L - 1, 1)` and `(L, 1)`; the chunked-prefill case over 2048-token chunks for a 5136-token and a 3000-token prompt, including an assertion that the un-announced geometry does split the 5136-token prompt across both tables (so the test fails if the announcement is ever removed); `short_mscale` overriding while an absent `long_mscale` keeps the default; top-level and block resolution of `original_max_position_embeddings` and the 4096 fallback; a config with no `rope_scaling` and a `linear` block building no tables; a `long_factor`-only block using it for both; and a fused-versus-graph relative-RMS parity check on a synthetic 4-bit quantized layer for each table, with a guard that the two tables produce measurably different Q so the parity check cannot pass vacuously.

### Not verified here

The real-checkpoint gates are outstanding and are run outside the implementation unit: `models/mlx/phi-3.5-mini-4bit` greedy token parity against a position-selecting transformers oracle on a prompt under 4096 tokens; the same checkpoint above 4096 tokens, token-exact against the same oracle and unchanged from current mlxcel output; and `models/mlx/phi-3-mini-4bit` (no `rope_scaling`) as the regression control. The full `--workspace` test and `--all-targets` clippy runs are likewise outside this unit.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|--------|-------|
| Files changed | 20, plus the two report files |
| Additions | about 2000 including the reports |
| Deletions | 82 |
| New modules | 1 (`mlxcel_core::prefill_span`) |
| New unit tests | 25 (11 in `phi3_tests.rs`, 5 in `prefill_span_tests.rs`, 3 in `prefill_span_coverage_tests.rs`, 1 in `generate.rs`, 5 in `scheduler_prompt_cache_tests.rs`) |

### Changes by category

| Category | Files |
|----------|-------|
| Model fix | `src/models/phi3.rs` |
| Core mechanism | `src/lib/mlxcel-core/src/prefill_span.rs`, `src/lib/mlxcel-core/src/lib.rs`, `src/lib/mlxcel-core/src/generate.rs` |
| Server wiring | `src/server/batch/scheduler/prefill.rs`, `src/server/batch/scheduler/prompt_cache.rs`, `src/server/batch/scheduler/mod.rs` |
| Prompt-cache regime | `src/server/prompt_cache/{key,types,store,metrics}.rs`, `src/loaded_model.rs`, `src/vision/mod.rs` |
| Tests | `src/models/phi3_tests.rs`, `src/lib/mlxcel-core/src/prefill_span_tests.rs`, `src/server/batch/scheduler/prefill_span_coverage_tests.rs`, `src/server/batch/scheduler_prompt_cache_tests.rs` |
| Docs | `docs/supported-models.md` |

### Related commits

- `a770022` fix(phi3): select the LongRoPE table by whole-prompt position
- `docs: add technical report for PR #1580`
- `d24c0a4` fix(server): announce the prefill span on every prefill forward
- `fa066b4` fix(server): make prompt-cache reuse RoPE-table-regime aware

### Related PRs and issues

- Closes #1358

---

## 7. Follow-up Actions

1. Run the three real-checkpoint gates in section 5 before merge.
2. `src/models/minicpm3.rs` has its own `SuScaledRoPE` that reads a scalar `long_factor` from a JSON map. It is a different shape (one scalar scaling the base, not a per-dimension list) and was out of scope here, but it is worth auditing for the same class of question: whether the checkpoint it serves declares a short list it ignores.
3. `Phi-3-small` (`phi3small`) and Phi-3.5-MoE have their own attention modules and were explicitly out of scope. If either carries `longrope`, it needs the same treatment.
4. `mlxcel_core::prefill_span` is generic: any future family whose RoPE table, mask shape, or scale depends on the whole sequence length can read it. The two announcement sites are the complete set of multi-call prefill drivers today; a third one added later must announce, and the doc comment says so.

### The broader lesson

This change took three rounds, and each round's gap was the same shape one level out. Round one selected the table correctly per forward pass, which is right for an implementation that prefills a whole prompt at once and wrong for one that chunks. Round two announced the whole prompt's length at the chunked-prefill sites, which covers every path that encodes one prompt and misses the paths that restore KV encoded by another. Round three made cross-request reuse regime-aware. The invariant that was missing each time is the same: **everything that puts keys into one KV cache must agree on the table**, whether "everything" means the passes of one prefill, the several places one request reaches the model, or two different requests sharing a prefix. A fix that names only the paths the author happened to look at will keep being incomplete at the next boundary out.

A defect of this class survives every test that asks "does the output read as text". The pre-change code matched its usual upstream reference exactly, which is normally the strongest evidence available in this tree, and was still wrong because that reference is itself incomplete for this feature. The property that separates the two implementations is not fluency and not RMS against mlx-lm; it is token identity against the implementation the checkpoint was trained and released with. Picking the oracle is part of the fix, not a step before it.
