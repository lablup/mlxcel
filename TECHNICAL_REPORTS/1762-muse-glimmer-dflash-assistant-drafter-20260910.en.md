# Technical Report: PR #1762 - feat(speculative): Muse Glimmer DFlash assistant drafter on the round loop

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation plus a pre-merge review round
**Status**: Completed (validated on Apple M5 Max / Metal against `models/muse-glimmer-30b-4bit` paired with `models/muse-glimmer-30b-assistant-bf16`; the `--workspace` gate was not run locally because the machine is shared, and the narrow scopes plus CI cover it)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a third family joins the DFlash round loop, the loop's verify width stops being fixed for the first time, and a family that previously rejected every `--draft-*` flag now accepts one drafter)

---

## Executive Summary

Meta publishes an "assistant" drafter for Muse Glimmer 30B: five sliding-attention decoder layers fed five of the target's residual streams through an `encoder.fc` projection, predicting a whole 16-token block in one non-causal forward, and borrowing the target's embedding table and head instead of shipping its own. mlxcel already served Muse Glimmer and already had the DFlash round loop, so the shape of the work is the same as the LFM2 DSpark change that preceded it (#1751): a drafter model, a target adapter, and the dispatch wiring.

Three things make it not a copy of that work.

The drafter's attention is not the Qwen-shaped one. It carries per-head q/k norms, and its mask is bidirectional over absolute positions rather than causal or absent. That mask is what rules out `RotatingKVCache` for the drafter's own context: once a ring has wrapped, a single-row append hands back the buffer in physical slot order, and a mask built from absolute positions cannot be laid over rows whose order it does not know. The drafter therefore gets a small purpose-built temporal window instead.

The verify width stops being a constant. The published `block_size` is 16, and on an M5 Max a fixed 16 rows decodes below classic decode on natural text while a 2660-token repetitive log accepts 14 of 15 proposals at that width. Neither number is the answer, so `DFlashGenerator` now reads the drafter's declared depth and, when the drafter does not insist on the requested width, runs the MTP loop's throughput comparator between the depth and the ceiling. Qwen 3.5 DFlash declares no depth and DSpark prefers the requested width, so both keep the behaviour they had.

That second change is also what the pre-merge review caught. An adaptive width invalidates a gate that probes one width: the burst asked `exactness_allows(16)` and the loop then spent its warm-up at 4. Both fixes in the review round come from taking the new width policy seriously, and one of them turned out to matter on the measured host.

---

## 1. The checkpoint

### 1.1 The config is flat and missing two keys

The issue's specification listed `num_target_layers 52` and `vocab_size 202048`. The published `meta-models/Muse-Glimmer-30B-assistant` `config.json` carries neither, and carries no `dflash_config` block either: `block_size`, `mask_token_id` and `target_layer_ids` sit at the top level.

Defaulting the two missing keys to zero and then validating `target_layer_ids[i] < num_target_layers` would reject the real checkpoint. Making them required would too. They are `Option`s, and the facts they would have stated are measured off the bound target instead, where the pairing has to be verified anyway:

- `target_layer_ids.last() < target.num_layers()` replaces the `num_target_layers` bound, and is the stronger check, since it compares against the target in hand rather than against what the drafter's author believed the target to be.
- `mask_token_id < vocab` is measured by forwarding a zero hidden through `target.lm_head_module()` and reading the width of the result. The mask id indexes the target's embedding table, so the target's head width is the number that has to hold.
- When either key IS present (a re-export through a tool that writes them), it is compared as well.

`MuseAssistantConfig::from_json` still lifts a nested `dflash_config`, so a checkpoint re-exported through the Qwen 3.5 or DSpark tooling loads too.

### 1.2 No `embed_tokens`, no `lm_head`, and both matter

The 58 tensors are `encoder.fc`, `encoder.output_norm_enc`, `norm`, and five decoder layers with `q_norm` / `k_norm` beside the four projections. There is no token table and no head.

Which table it borrows is load-bearing. Muse Glimmer's `LanguageModel::embed_tokens` returns the lookup with `embed_norm` already applied, which is what the target's own forward wants. The drafter runs its own `input_layernorm` and wants the raw table. `embed_tokens_module()` and `lm_head_module()` hand out shared handles to the raw table and the untied head with no `output_multiplier` and no `final_logit_softcapping` behind it. Feeding the drafter the `embed_norm`-wrapped lookup was measured, not assumed: acceptance per round fell from 1.87 / 3.40 / 1.80 to 1.28 / 1.35 / 0.90 on three natural prompts.

---

## 2. The drafter forward

### 2.1 The mask is bidirectional over absolute positions

A draft block is predicted in one forward, so every proposal slot may see every other slot. The only restriction is the window:

```
mask[i, j] = |(query_start + i) - (key_start + j)| <= sliding_window
```

`bidirectional_sliding_mask_bool` builds it as a boolean array rather than an additive bias on purpose. An additive bias has to match the dtype the fused SDPA promotes q/k/v to, and that is not the query dtype when the target's residual streams and the drafter's dense weights are stored differently; a mismatch is an MLX throw across the cxx bridge, which aborts rather than fails. A boolean mask is dtype-free.

Because every query row is also a key row at distance zero, no row of the mask is entirely closed, which is the property the additive form would have needed a `-inf` audit for.

### 2.2 The context cache is not a `RotatingKVCache`

`RotatingKVCache` is what the Muse TARGET uses for its sliding layers, and it is the right structure there: the target's mask is built by the layer from the cache's own accounting. The drafter's mask is built by the drafter from absolute positions, and it needs the cache to hand back its rows in temporal order together with the absolute position of the first one. A wrapped ring cannot do that for a single-row append, and the drafter appends one row after every zero-accept round.

`MuseAssistantContextCache` is therefore append-only, never trimmed, keeps at most `window` rows, and counts `offset` over every row the layer has ever consumed including rows skipped before they reached it. `key_start()` is `offset - len()`, which is exactly what the mask needs. It is never rolled back: the round loop already hands the next round only the committed rows, so the drafter's window is the committed prefix by construction.

### 2.3 One projection, one RoPE

Context rows and proposal rows go through `k_proj` / `v_proj` as one concatenated input, and one `fast_rope` at `cache.offset` places the context at `offset..offset + S` and the proposals immediately after. The queries get the same treatment at `offset + S`. Splitting them would need two RoPE calls at two offsets and would make the position arithmetic a thing to keep in sync in two places.

Only the context rows enter the cache. The proposal K/V is concatenated onto the fetched window and discarded, which is what keeps `offset` equal to the committed row count rather than the verified row count.

---

## 3. The target side

### 3.1 Capture before the final norm

`forward_speculative` is the generation forward with `mask = None`, so every layer builds its own causal plus sliding mask, plus a push of `h` after each layer named in `capture_layer_ids`. Capturing before the final norm is the specification, and it is also the measured choice: capturing one layer earlier (the HF `hidden_states[i]` indexing) changed acceptance by nothing worth having (1.83 / 3.47 / 1.50 against 1.87 / 3.40 / 1.80).

`prefill_forward_with_capture_layers` keeps the trait default, which is the opposite of what LFM2 did with the same hook. LFM2 overrides it because its verify output carries prompt-sized short-conv snapshots that a prefill has no use for. A Muse verify output carries no rollback state at all, because the rollback reads cache offsets, so there is nothing prompt-sized to skip.

### 3.2 Rollback is a trim, and the buffer is what makes it one

Every Muse cache is a KV cache: `KVCache` on the full-attention layers, `RotatingKVCache` on the sliding ones. A partial accept therefore rewinds by arithmetic, `offset -= n` and, for the ring, `idx -= n`. No data moves and the next append overwrites the rejected rows.

That holds only if the rejected rows did not overwrite something still visible on the way in. A `bs`-row verify block appended to a full ring displaces `bs` window entries, and rewinding the offsets does not bring them back. The sliding caches are therefore armed before the first round with `enable_speculative_buffer`, which is extra capacity past `max_size`. As long as the buffer is at least as wide as the block, an append-then-rewind is lossless. Section 7.2 is about the case where it was not.

---

## 4. The verify width is measured, not fixed

The DFlash round loop had one `block_size` for the life of a run. That is right for a drafter whose trained width is the width to use, and wrong here.

Measured on the M5 Max pairing, all under the inexact override so the arms are comparable, three natural prompts:

| Verify rows | tok/s | Accepted per round |
|---|---|---|
| 16 (published) | 13.0 / 18.1 / 12.4 | 1.87 / 3.40 / 1.80 |
| 8 | 24.9 / 30.5 / 21.7 | |
| 4 | 33.6 / 41.3 / 31.6 | 1.34 measured separately |
| 2 | 30.0 / 33.1 / 31.7 | first proposal accepted 74 to 82 percent of rounds |

Classic decode on the same prompts is 18.1 tok/s, so the published width is a slowdown on natural text. On a 2660-token repetitive log the same 16 rows accept 14 of 15 proposals. A fixed width has to give one of those two up.

`Drafter::configured_block_size` and `prefer_requested_block_size` already existed for the MTP loop. `DFlashGenerator` now reads them: a drafter that declares a depth below the requested width and does not insist on the requested width gets `BlockThroughputController` between the two, which warms up at the depth, then alternates measurement windows and holds whichever emits more tokens per millisecond. Qwen 3.5 DFlash returns `None` for the depth and DSpark returns `true` for the preference, so `dflash_round_loop_starts_at_the_configured_depth` pins that both still draft at the requested width from round one.

---

## 5. The server contract

PR #1751 left `DFlashTargetModel` with one piece of per-family policy outside the trait: the batched variant gate carried a hardcoded LFM2 arm, with a note that a `supports_batched()` hook was the fix and was deferred to the next family that needed it. Muse Glimmer is that family, so the hook is here and the gate is now a match that recovers the concrete type and reads the policy off the trait.

`requires_dspark_drafter() -> bool` becomes `required_drafter_family() -> Option<DFlashDrafterFamily>` for the same reason: two families now restrict their pairing, and a boolean cannot say which one. The message names the family that was passed and the family that is required, in both run arms.

This gate cannot be moved into the drafter. A drafter sees its target as a `LanguageModel` with no architecture string, so a plain DFlash drafter's `validate_target_compat` returns `Ok` for a Muse target it cannot run. The target is the side that knows.

---

## 6. The startup guard

`validate_muse_glimmer_unsupported_startup` used to reject `--draft-model`, `--draft-kind` and `--draft-block-size` together, unconditionally. It now splits on whether a drafter was named:

- No drafter: `--draft-kind` and `--draft-block-size` are still rejected, which is the operator mistake the blanket rule also caught.
- A drafter: the directory has to declare `muse_glimmer_assistant` (or the `MuseGlimmerAssistantModel` architecture), and `--draft-kind` has to be absent or `dflash`.

Both halves refuse by name before the server loads anything, and adapters, KV modes, TP and PP stay rejected.

---

## 7. Pre-merge review round

Three findings were fixed on the branch. The rest are recorded in the PR body.

### 7.1 The exactness gate probed a width the loop does not run at

The burst calls `exactness_allows(bs)` with the requested block size, memoized per (model, width) for the process. That was sound while the width was fixed. Section 4 makes it unsound: the loop warms up at 4 and the gate measured 16.

This is not a formality. The forward width selects which quantized-matmul kernel MLX dispatches, which is why `docs/benchmarks.md` records the same comparison reading 20.6 percent disagreement at width 8 and 0.0 percent at width 32. A pass at the ceiling says nothing about the depth.

`probed_verify_widths(block_size)` names the widths a run can settle on, narrowest first, and `dflash_exactness_allows_every_width` requires all of them. Rounds the emission budget forces narrower at the end of a run are deliberately not listed: that clamp predates the adaptive width and applies to every DFlash family, so it belongs to a change that addresses it for all of them.

On the measured host the probe declines at width 4 as well (162404 of 404096 logit bytes differ, against 187123 at width 16), so the gate had been publishing a verdict for a width the loop would have left after the warm-up. The served behaviour on this host does not change, because both widths decline; on a host where 16 passes and 4 does not, it is the difference between a correct decline and a silent divergence.

### 7.2 The speculative buffer could be narrower than the block it buffers

`speculative_buffer_size` was `clamp(block * 8, 32, 128)`, the Gemma 4 MTP rule verbatim. The ratio is generous up to 16 rows and then the cap takes over, so at 129 rows and above the buffer is narrower than the block.

`--draft-block-size` is not bounded above anywhere on the way in: `resolve_draft_block_size` returns an override verbatim and documents that concrete generators enforce their own minimums, which is a statement about minimums. So a `--draft-block-size 200` run appends 200 rows into 128 rows of slack, overwrites 72 still-visible window entries, and then `RotatingKVCache::trim` clamps against the live length and rewinds the offsets over rows it cannot restore. The sequence continues from a window with holes in it, with no error anywhere and no test that would notice.

The rule now floors at the block size. At every width a real run reaches this is the same number it was, which is why the existing Gemma-rule test is unchanged and a second test carries the invariant.

Two smaller items came from the same reading. A short rewind now emits `tracing::error!` naming the layer and the shortfall, rather than only a `debug_assert` that the release profile compiles out. And a verify that captured no residual stream for a drafter slot says so instead of quietly substituting a zero slab, which the drafter would read as a residual stream.

### 7.3 Config dimensions, and an encoder projecting rows it discards

Every dimension in `MuseAssistantConfig` is handed to MLX as an `i32`. A `sliding_window` past the wrap point becomes a negative extent, and a negative extent reaches MLX as a slice it throws on, which crosses the cxx bridge as an abort rather than as a load error. `validate` now bounds each of them, plus the `len(target_layer_ids) * hidden_size` product that `encoder.fc`'s shape check computes.

Separately, the drafter's `forward` trimmed the context to the window AFTER running `encoder.fc` and `output_norm_enc` over it. Both are per-row, so the trim now happens first. The rows that arrive on the first round are the whole prompt: at `-c 16384` the incoming `[1, S, 5 * 6656]` slab is 1.1 GB in f16, and seven eighths of it was being cast to the drafter's dtype and projected only to be sliced away.

---

## 8. Real-checkpoint measurement

`models/muse-glimmer-30b-4bit` with `models/muse-glimmer-30b-assistant-bf16`, M5 Max, GPU generation 17, port 19343, server rebuilt after the rebase and after the review fixes with no source file newer than the binary. Three text prompts at `temperature: 0, max_tokens: 256, logprobs: true`, token strings compared position by position against the same server without `--model-draft`, plus one image request.

| Prompt | Prompt tokens | Shipped default | Under `MLXCEL_MTP_ALLOW_INEXACT=1` |
|---|---|---|---|
| tides | 66 | IDENTICAL, served classically | DIVERGES at token 59; 109 rounds, 146 of 375 accepted, mean 1.34, 38.4 tok/s decode |
| cross-window decode | 1968 | IDENTICAL, served classically | IDENTICAL; 47 rounds, 209 of 213 accepted, mean 4.45, 60.5 tok/s decode |
| wrapped in prefill | 2660 | IDENTICAL, served classically | IDENTICAL; 47 rounds, 209 of 214 accepted, mean 4.45, 60.6 tok/s decode |

End to end against the classic arm of the same prompt: 36.7 against 31.6, 33.9 against 22.5, 31.3 against 20.1 tokens per second.

The shipped default declines because the probe declines, which is the documented Apple GPU generation 15 and newer condition and the same verdict LFM2 DSpark gets on this host. The tides divergence under the override is the byte-identity the probe declined, observed after overriding it, which is the argument for the override not being the default rather than an argument against the pairing.

The image request declines with one `multimodal VLM request detected` line and is served classically at HTTP 200.

---

## 9. Change summary

| Item | Value |
|------|-------|
| Files changed | 28 |
| Lines added | +3848 |
| Lines deleted | -165 |
| New `#[test]` functions | 20 |

| Area | Summary |
|------|---------|
| Drafter core | `drafter/dflash/muse/` (new): config with the two optional keys and the dimension ceiling, sliding attention with per-head q/k norms and the bidirectional mask, the temporal context cache, the model with its target binding, the `Drafter` adapter and its pairing gate; `muse_glimmer_assistant` in `drafter_kind_by_model_type` and in `load_drafter`; `Drafter::is_muse_assistant` |
| Round loop | `DFlashGenerator` reads `configured_block_size` / `prefer_requested_block_size` and runs `BlockThroughputController` between a declared depth and the requested ceiling |
| Muse target | `muse_glimmer_speculative.rs` (new): verify forward with capture, trim rollback, speculative buffers, the exactness probe and the multi-width gate, `SpeculativeTarget`; `MuseCache::trim` and `MuseCache::enable_speculative_buffer`; `embed_tokens_module` / `lm_head_module` on both wrappers |
| Server | `DFlashTargetModel::supports_batched` replaces the hardcoded LFM2 batched arm; `required_drafter_family` replaces the DSpark boolean; `MuseGlimmerVLM` joins the three `run_dflash_burst` arms and `model_variant_label`; the startup guard admits the assistant drafter and nothing else |
| CLI | `peek_muse_assistant_configured_block_size` in `resolve_draft_block_size`; the offline rejection names the third drafter shape |
| Docs | `supported-models.md` DFlash row and family paragraph, `speculative-acceptance.md` multi-width gate, README |

Verified: `cargo fmt --all -- --check`; `cargo clippy --lib --tests` and `cargo clippy -p mlxcel-core --lib --tests` with `-D warnings`, both under `--features metal,accelerate`. Tests under `--profile test-fast --features metal,accelerate`: `-p mlxcel-core drafter::dflash` 86 passed, and the root crate on `muse_glimmer` 109, `server::batch::speculative_burst` 67, `server::startup` 80, `models::detection` 63, `cli::speculative_args` 22, `server::batch::dflash_target` 4.

---

## 10. Follow-up

- `--draft-block-size` is still unbounded above. The buffer floor removes the silent-corruption consequence for Muse, but an absurd width still allocates an absurd verify block and, on the MTP arms, still reaches the per-row exactness probe. A shared clamp in `resolve_draft_block_size` or in the server startup validation would cover every DFlash and MTP family at once. It is the same item #1751's report left open.
- The burst's decline line names the requested ceiling while the probe verdict above it names the width that actually failed. The burst message is shared by all DFlash families and it points the reader at the verdict, so making it family-specific was not worth the divergence.
- `MuseAssistantContextCache::append` rebuilds the whole window every round. A ring with a rotate-on-read, or `RotatingKVCache` taught to report absolute key positions, would remove the copy. It is a fraction of one target verify forward and does not move the measurements above.
- Batched (B > 1) Muse windows and multimodal requests under the drafter remain out of scope, and `draft_window_size` is accepted and unused.
- Issue #1289 (order-preserving streamed qmv) is the route to passing the probe on the kernels that decline today. Until then the Muse burst is a measured decline on generation 15 and newer, not a measured speedup, exactly as for DSpark.
- `ProbeKey` still has no family discriminator. Pre-existing, shared with the MTP and DSpark arms, and now shared with a third family, so the case for fixing it at the memo is stronger than it was.
