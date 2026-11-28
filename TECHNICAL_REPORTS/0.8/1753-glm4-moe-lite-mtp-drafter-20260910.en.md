# Technical Report: PR #1753 - feat(speculative): GLM-4.7-Flash (glm4_moe_lite) MTP drafter

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation plus review follow-up cycle
**Status**: Completed (validated on Apple M5 Max / Metal against `models/glm-4.7-flash-4bit` paired with the 4-bit drafter `mlxcel split-mtp` produced from `models/glm-4.7-flash-bf16`; the `--workspace` gate was not run locally because the machine is shared, and the narrow scopes plus CI cover it)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a fourth family joins the MTP round loop, `glm4_moe_lite` becomes the first pool-cached family to hold a model-owned cache slot, and a new subcommand writes checkpoint directories from user-supplied paths)

---

## Executive Summary

`zai-org/GLM-4.7-Flash` stores one next-token-prediction layer as `model.layers.47.*`, one past its 47 decoder layers, and every community 4-bit conversion drops it. There is nothing to download and point `--draft-model` at, so this change manufactures the drafter: `mlxcel split-mtp` extracts the nextn block into a standalone `glm4_moe_lite_mtp` directory, `Glm4MoeLiteMtpDraftModel` runs it as a stateful drafter, and an `MtpTarget` adapter puts `Glm4MoeLiteModel` on the round loop that already served Gemma 4, Qwen 3.5 and Inkling. The round loop, the drafter trait, the target trait, the deferred-greedy verify rule and the block-versus-chain exactness gate were all in place; what was missing is a tool, a model and an adapter.

Two of the decisions are worth carrying forward on their own. The verify forward has two width-sensitive pieces and they are decided in opposite directions: the attention runs one materialized query row at a time so every SDPA call has the shape a decode step issues, while the MoE is deliberately left batched and handed to the probe to measure. And the review found that the model-owned cache slot this branch introduced was built from a bare `KVCache::new()` with no path from the operator's `--kv-cache-mode` to it, because `glm4_moe_lite` is the first pool-cached family to acquire such a slot and had never needed the mode hook that model-owned families implement.

---

## 1. The drafter is not in any checkpoint you can download

GLM-4.7-Flash declares `num_nextn_predict_layers: 1` and stores that layer as `model.layers.47.*`: a DeepSeek-V3-style nextn block with its own `embed_tokens`, `enorm`, `hnorm`, `eh_proj`, one MLA plus MoE decoder block, `shared_head.norm` and `shared_head.head`. The mlx-community 4-bit conversion names none of those keys in its index. So the raw checkpoint is the only source, and the drafter has to become a separate directory that the target loader never reads.

Reading the nextn layer at target load was rejected in the issue and stays rejected. A separate directory is independently quantizable, which matters here: the target is 16 GB at 4 bits and the extracted drafter is 705 MB at 4 bits, and an operator who wants the drafter at a different depth than the target should not have to re-convert the target. It also keeps the target loader's layer range at `0..num_hidden_layers`, so a raw checkpoint served without a drafter behaves exactly as before.

`ModelArgs` gains `num_nextn_predict_layers: usize` with `#[serde(default)]` and the loader still builds `0..num_hidden_layers`. The field exists so the drafter config, which is a copy of the target's, can derive its default block size from it.

---

## 2. `split-mtp`

`src/lib/mlxcel-surgery/src/ops/split_mtp.rs` holds the transform (`split_mtp` over an in-memory `WeightMap`) and the directory driver (`split_mtp_dir`); `src/commands/split_mtp.rs` is the `mlxcel split-mtp` subcommand, both behind `#[cfg(feature = "surgery")]`.

The driver never loads the checkpoint whole. `index_names_nextn_layer` reads `model.safetensors.index.json` alone and answers whether any key starts with `model.layers.{num_hidden_layers}.`, opening no shard at all; `load_weights_from_dir_index_filtered` then opens only the shards the index names for that prefix, three of the raw checkpoint's 48. A 62 GB source costs a few GB of address space.

### 2.1 The rename, and the two rewrites that make the block loadable

Six prefix renames lift the block's root tensors out of the layer namespace, `*.rotary_emb.inv_freq` is dropped, and everything else lands under `model.mtp_block.`:

```text
model.layers.N.embed_tokens.weight      -> model.embed_tokens.weight
model.layers.N.enorm.weight             -> model.enorm.weight
model.layers.N.hnorm.weight             -> model.hnorm.weight
model.layers.N.eh_proj.weight           -> model.eh_proj.weight
model.layers.N.shared_head.norm.weight  -> model.shared_head_norm.weight
model.layers.N.shared_head.head.weight  -> lm_head.weight
model.layers.N.<rest>                   -> model.mtp_block.<rest>
```

Renaming alone does not produce a loadable block, because the `glm4_moe_lite` loader expects two tensor layouts the raw checkpoint does not ship. Both rewrites are the ones the target's own sanitizer already performs per decoder layer.

The first is the MLA decomposition. `kv_b_proj.weight` at `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` is reshaped to `[H, qk_nope + v, rank]` and split into `embed_q.weight` at `[H, kv_lora_rank, qk_nope_head_dim]` (the nope half, transposed) and `unembed_out.weight` at `[H, v_head_dim, kv_lora_rank]`, each copied so `MultiLinear`'s matmul sees contiguous strides rather than a view. On GLM-4.7-Flash that is `[8960, 512]` becoming `[20, 512, 192]` and `[20, 256, 512]`.

That code used to live inline in `src/models/glm4_moe_lite_sanitize.rs`, which is why the sanitizer is `+29 / -112` in this diff and went from 167 lines to 84. It moved to `mlxcel_core::mla::decompose_kv_b_proj` because `mlxcel-surgery` sits below the binary crate in the dependency graph and could not call it where it was. Three callers share it now: the target sanitizer (once per decoder layer, labelled `layer {i}`), the surgery op (once, labelled `MTP block`), and the drafter's own sanitizer for a hand-assembled directory that kept `kv_b_proj`. A `split-mtp` output already carries the pair, so the third caller returns `Ok(false)`. `kv_b_proj_geometry(&ModelArgs)` is public alongside it so the two sanitizers cannot disagree about which head dimension is which.

The second rewrite is expert stacking: `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight` for `e in 0..n_routed_experts` becomes `mlp.switch_mlp.{proj}.weight` at `[64, out, in]`. A projection already present in stacked form is skipped rather than rebuilt.

### 2.2 Two exemptions, and they are different exemptions

The dtype pass casts everything to bf16 except `mlp.gate.e_score_correction_bias`, which is forced to float32 because the router's selection compares scores against it and a bf16 round trip changes which experts are chosen.

The optional affine quantization pass exempts a different tensor for a different reason. `quantizable` requires a `.weight` suffix, rank two or more, and a last axis that divides the group size, and it excludes `mlp.gate.weight` by name so the router itself stays dense. Everything matching is written as packed `.weight` plus `.scales` and `.biases`, and the config records `{"group_size": 64, "bits": 4, "mode": "affine"}` at both `quantization` and `quantization_config`. The loop iterates a key snapshot taken before quantization, so the `.scales` it writes are never revisited.

The produced 4-bit drafter is 54 tensors. `mlp.gate.weight` is there at `[64, 2048]` in bf16 and `e_score_correction_bias` at `[64]` in float32, which is the pair of exemptions visible in the artifact.

The written `config.json` has four keys plus the two quantization blocks: `model_type: "glm4_moe_lite_mtp"`, `block_size`, `tie_word_embeddings: false` (unconditional, since the block ships its own head), and `text_config`, which is the source text config with `quantization` and `quantization_config` stripped so a 4-bit source's stale block cannot describe the drafter's own state. Five companion files are copied when present, and a missing `tokenizer.json` or `tokenizer_config.json` is reported as a warning rather than an error.

### 2.3 What the tool refuses

`--block-size` is bounded at `MAX_BLOCK_SIZE = 16`, checked before the geometry reads and before any tensor work. The recorded value becomes the server's default verify width through `peek_glm4_moe_lite_mtp_configured_block_size`, and this family materializes one query row per verify position, so the width multiplies both the per-round graph size and the exactness probe's chain arm. An unchecked `--block-size 20000` produced a directory that loaded fine and then built roughly a million graph nodes per round with the scheduler tick held. Drafting past the trained depth already loses acceptance, so a ceiling well above any useful width costs nothing.

The `--force` guard now screens on whatever makes a directory a checkpoint, not on `model.safetensors` alone: `existing_checkpoint_marker` returns the first of `model.safetensors`, `model.safetensors.index.json`, `config.json`, or a `model-*.safetensors` shard. A sharded checkpoint carries no `model.safetensors`, so before this an `--output` pointed at one passed the guard and the run replaced its `config.json` with the drafter's and overwrote up to five tokenizer files while orphaning the shards. `--force` additionally removes a stale `model.safetensors.index.json`, because `collect_shard_paths` prefers an index over a bare `model.safetensors` and a leftover one would send the loader to the victim's shards.

An `--output` that canonicalizes to the source is refused before anything is written. The failure mode is quiet: `std::fs::copy` on a path to itself returns `Ok(0)` after opening the destination with `O_TRUNC`, verified on the validation host, so every companion file would be zeroed with no error surfaced. A nested path under the source is a different directory and is still allowed.

A nextn layer that arrives already quantized is refused by `reject_packed_tensors`, which runs after the decomposition and the stacking so it only sees packed tensors neither of those handles. `stack_experts` rejects unstacked packed experts per expert by name, but a conversion whose experts were already stacked and packed slipped past that check and reached the bf16 pass, which rewrote the packed uint32 payload as floats with no error at all.

Finally, a checkpoint whose index names no `model.layers.{N}.*` tensor is refused with the message that names the cause: converted without its MTP layer, use the raw zai-org checkpoint. Nothing is created in the output directory before that fires.

### 2.4 `decompose_kv_b_proj` and `i32::try_from`

Moving the decomposition into a shared helper put it on the surgery path, where the config is a user-supplied JSON file rather than a checkpoint the loader already validated. The four geometry fields were converted with `as i32`. The `test-fast` profile inherits release, so overflow checks are off, and `num_heads * head_dim` under `as i32` semantics can wrap to a value that accidentally equals `w_shape[0]`, passing the very shape check that exists to keep a malformed config away from `reshape`. MLX reports a bad reshape by throwing, and a C++ throw crossing the `cxx::bridge` (which returns `UniquePtr<MlxArray>`, not `Result`) aborts the process rather than failing the call. In the server that happens during weight sanitization and takes down every tenant.

The conversions are now `i32::try_from` with a named field in the error, `qk_nope_head_dim + v_head_dim` is a `checked_add`, and the head-dim product is range-checked with `checked_mul` before the shape comparison uses it. The guard product is computed and discarded; the comparison recomputes it.

---

## 3. The drafter

### 3.1 One step

`Glm4MoeLiteMtpDraftModel` holds its own token table, `enorm`, `hnorm`, `eh_proj`, one `TransformerBlock`, `shared_head_norm` and an untied `lm_head`. Nothing is borrowed from the target. With `e = embed_tokens(token_{t+1})` and `h_t` the target's hidden at position `t`:

```text
x           = eh_proj(concat(enorm(e), hnorm(h_t), axis=-1))
x           = block(x, cache)          # RoPE at cache.offset, own KVCache
logits      = lm_head(shared_head_norm(x))
hidden_next = x                        # fed back for the next draft step
```

The block is built through `TransformerBlock::from_weights_with_prefix(weights, args, "model.mtp_block", is_moe)`, a new entry point that `TransformerBlock::from_weights` now delegates to with `model.layers.{i}` and `args.is_moe_layer(i)`. That is the point of the whole arrangement: the drafter's absorbed MLA, its sigmoid routing with the selection-only correction bias, its grouping and `routed_scaling_factor` and shared expert are the decoder's code, not a second copy that can drift.

Weight inventory fails closed. Every key must start with one of seven allowed prefixes, and a directory carrying `model.layers.0.*` is rejected by name, so a full target checkpoint cannot be mistaken for a drafter directory.

### 3.2 The cache arithmetic

The drafter is stateful: one `KVCache` holding one entry per target position consumed, with `next_position` tracking the absolute target-sequence position of the next append and `round_appended` counting this round's speculative appends.

`prefill_from_target_hidden` shifts the prompt left by one, appends the first bonus, and runs the whole thing paired with the target's prompt hidden in a single forward, leaving the cache at `P`. `accept_verified_tokens` is where the arithmetic lives:

```rust
let keep = accepted.min(self.round_appended.max(0) as usize);
let trim = self.round_appended - keep as i32;
if trim > 0 { self.cache.trim(trim); self.next_position -= trim; }
let mut tokens: Vec<i32> = draft_tokens[keep..accepted].to_vec();
if let Some(&last) = new_tokens.last() { tokens.push(last); }
```

`keep` is capped by `round_appended` rather than by `accepted` alone because the seeded first proposal's cache entry was appended by the previous hook, not by this round, so it is not this round's to trim. The re-forward pairs `draft_tokens[keep..accepted]` plus the bonus with verify-hidden rows `[keep, keep + tokens.len())`, which is the target's true hidden for those positions rather than the drafter's own guess. Starting from `P`, every accepted count lands the cache at `P + accepted + 1`, which is what lets the next `set_shared_kv` keep the history instead of re-anchoring.

A `draft_block` that finds a seed spends zero forwards on its first proposal, so a seeded round runs `block_size - 2` forwards and a seedless one runs `block_size - 1`. At the default `block_size = 2` that means a seeded round drafts without running the block at all.

`accept_verified_tokens` validates before it mutates and clears runtime state on the error path, so a poisoned drafter re-anchors cleanly at the next `set_shared_kv` rather than continuing from a half-applied round.

### 3.3 Which hidden state feeds it

The issue left this open deliberately: no key name settles whether the nextn block wants the target's post-final-norm output or the pre-norm residual, and it asked for the answer to be measured and recorded in the drafter's module docs.

**Post-final-norm won and is the default.** On `models/glm-4.7-flash-4bit` paired with the 4-bit drafter, M5 Max, 128 greedy tokens at `block_size = 2`, the post-final-norm tap accepted 55 of 73 proposals (mean accepted length 1.753) against 53 of 74 (1.716) for the pre-norm residual. The emitted 128-token id streams were identical under both taps, as the target's verify pass guarantees, so what the tap moves is acceptance and nothing else. `MLXCEL_GLM_MTP_HIDDEN_TAP=pre` selects the loser for re-measurement.

The measurement agrees with the DeepSeek-V3 nextn convention and with the Qwen 3.5 adapter's existing choice, which is the reason to trust a 0.037-token gap on a single prompt: the number is a check on a prior, not the whole basis for the choice. Both forwards capture the pre-norm residual and `apply_final_norm` is re-applied on the capture path, so the default tap recomputes the final RMSNorm the forward already computed for its logits.

### 3.4 Where the drafter is built

`mlxcel-core` cannot construct this drafter. It reuses `glm4_moe_lite`'s `TransformerBlock`, which lives in the binary crate above it, so core has no name for the type. `src/models/drafter_loader.rs` is the one entry point the offline CLI, the server's drafter slot and the speculative bench all go through: it resolves the kind, builds the GLM drafter itself when the peeked `model_type` matches, and delegates everything else to core unchanged.

Core still registers `("glm4_moe_lite_mtp", DrafterKind::Mtp)` so `--draft-model` auto-detects the kind without `--draft-kind`, and `GLM4_MOE_LITE_MTP_MODEL_TYPE` is public (the Qwen constant next to it stays private) because dispatch happens one crate up. Core's own `load_drafter` refuses the model type by name with `DrafterError::BinaryCrateDrafter`, placed ahead of the Gemma 4 assistant catch-all so the directory does not fall into a loader whose weight-inventory error would blame the wrong family. The error text names `mlxcel::models::drafter_loader::load_drafter` as the remedy, and a test asserts the message keeps pointing there.

`configured_block_size()` reports `runtime_block_size()`, which is `min(block_size, num_nextn_predict_layers + 1)` floored at 2, so it is the trained depth rather than the parsed field. `prefer_requested_block_size()` is `true`, so a larger `--draft-block-size` still wins.

---

## 4. The verify forward: two width-sensitive pieces, decided opposite ways

The temperature-0 contract is that an `M = bs` verify block emits the same logits as `bs` single-token decode steps. Three things can break it on a quantized checkpoint: which quantized-matmul kernel MLX dispatches at `M = bs` versus `M = 1`, the attention, whose SDPA kernel and score reduction differ between a one-row and a multi-row query, and the MoE, whose reduction depends on the row count. The gate's `qmv_wide` retry handles the first. The other two are decided against each other.

**The attention runs one materialized query row at a time.** The projections (`q_a`/`q_b`, `kv_a`, `embed_q`, `unembed_out`, `o_proj`) run once over the whole block, because that is where the verify saves time. The per-row loop then slices row `i` out and attends it against the `offset + i + 1` cached entries it may see, so causality comes from the slice extent and no additive mask is built, and every SDPA and every `q_pe @ k_pe^T` call has exactly the shape the single-token decode step issues. That is the same split Qwen 3.5's `target_verify` makes.

Shape alone was not enough. A one-row slice of the batched `[1, H, bs, d]` query keeps the parent's strides, and MLX's SDPA and matmul pick their kernel on contiguity, so the row views are passed through `mlxcel_core::contiguous(..., false)` before use. Left as views, the verify rows drifted from the decode chain data-dependently, first at position 4 on one real prompt and at 14 on another, and the drift compounded through the cache. Only the query side is materialized; the key and latent slices start at index zero and are already leading prefixes of a contiguous parent.

That defect also sized the probe. `PROBE_BLOCKS_PER_DRAW = 4` exists because one block per draw missed the divergence entirely: it surfaced only after several `M = bs` blocks had fed the cache. The probe is 3 draws of 8-token prompts, each walking 4 consecutive blocks and comparing every row bitwise, memoized on `ProbeKey` through the shared `mtp_exactness_gate` that owns the decline log line, the `qmv_wide` retry and `MLXCEL_MTP_ALLOW_INEXACT`.

**The MoE is left batched on purpose.** `SwitchGLU::forward` switches to the gather-sort path once `n_tokens * top_k >= 64`. At the default `block_size = 2` with `num_experts_per_tok = 4` the product is 8, so the verify block takes the same reduction as the decode chain and nothing needs doing. A wide `--draft-block-size` crosses the threshold: 16 rows at top-4 is exactly 64. Undoing that per row would give back the verify's entire saving, so the probe is left to measure it, and a width that crosses the threshold and diverges declines the pairing rather than emitting a divergent stream. The probe is the only thing standing between a wide `--draft-block-size` and a divergent stream on this family, which is a deliberate position rather than an oversight.

---

## 5. The MTP slot and the KV cache mode table

This is the review finding rated HIGH, and it is a structural gap rather than a live regression. Both halves of that sentence matter.

`Glm4MoeLiteModel` is a `DenseKvCache` family: classic serving keeps its caches in the scheduler's `CachePool`, and the scheduler upgrades them to the resolved per-layer modes inside the pool right after allocation. That is why the model never implemented `LanguageModel::set_kv_cache_layer_modes` and took the no-op default. Gemma 4 and Qwen 3.5 are model-owned throughout and both implement it.

The MTP adapter cannot use the pool. Its trait methods are `&self` with no cache parameter, and the server's default MTP path is the tick-cooperative slice that rebuilds the adapter every tick, so the caches have to live on the model. This branch therefore added `ModelOwnedSequenceState<KVCache>` to `Glm4MoeLiteModel`, making it the first pool-cached family to hold a model-owned slot, and every install built that slot from `make_caches()`, which is unconditionally FP16. Nothing carried the operator's resolved mode table to the caches an engaged MTP session actually runs on, while the server still logged the requested mode as applied.

The fix is a `KvCacheLayerModes` table on the model, both trait methods implemented, and a new `make_configured_caches()` that every slot install goes through. `make_caches()` stays unconditionally FP16 on purpose, because it feeds the pool, which applies modes itself. Seven call sites moved: the two `reset_mtp_sequence_state` arms, the two lazy-creation closures in the prefill and verify hooks, `reset_runtime_state`, and **both arms of the block-versus-chain probe**. The last of those is the one that would be easy to skip and wrong to skip: a quantized KV mode is part of the arithmetic the probe is deciding about, so probing on FP16 would clear a path the engaged session does not run.

The offline CLI needed a third injection point. `run_offline_mtp` builds no `GenerationConfig`, so nothing on that path would have carried the announced mode to the slot; it now computes the same `resolve_layer_modes(kv_cache_mode, num_layers, boundary_v_layers_from_env())` the server's `resolved_kv_cache_layer_modes` computes and injects it before dispatch. That injection is scoped to `glm4_moe_lite` deliberately: the other MTP families have the same gap on the offline path, but correcting theirs changes what their offline runs measure and belongs with a real-checkpoint validation of its own.

**No configuration reaches the mismatch today.** `resolve_kv_cache_mode_for_model` (#1350) already resolves every quantized mode to fp16 for the MLA-latent families, `glm4_moe_lite` among them, on both the `--kv-cache-mode` and the `--kv-bits` route. The `--kv-cache-mode int8` validation run confirms it from the outside: the banner reports `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)` and the classic and MTP streams are id-identical over 64 tokens. So #1350's substitution is why the gap was never operator-visible, and closing it holds the invariant if this family ever gains a calibrated quantized mode.

The test is built so a blanket application cannot pass it: the injected table sets every layer to Int8 except the last, which is left at Fp16, and it checks both install sites rather than the constructor alone.

---

## 6. Prefix caching and MTP do not combine on this family

The two features are mutually exclusive here and the docs now say so rather than implying that adoption works normally.

The Gemma 4, Qwen 3.5 and Inkling adapters run on model-owned sequence state, which is exactly where a prompt-cache adoption restored the prefix, so they forward only the suffix (#518). `glm4_moe_lite` keeps its classic caches in `CachePool` and its MTP adapter runs on a separate model-owned slot: the adopted prefix is not reachable from that slot, and finishing such a request through MTP would leave the pool cache holding only the prefix while the finalizer donates it under the full `prompt ++ generated` key.

`mtp_adopted_prefix_reusable(model)` is the one-line policy, and both the burst and the tick slice consult it when `prefill_start_offset > 0`, before the drafter is taken from the slot. The decline is safe rather than merely tolerable: no partial entry can reach the prompt-cache store on either the dense or the paged arm. The cost is the lost optimization, and `docs/supported-models.md` now names the trade directly, telling an operator to serve without `--model-draft` when prefix reuse across turns matters more than the decode speedup. An MTP-served request donates nothing, so a later turn finds no prefix to adopt.

---

## 7. Other review findings, and what was done about them

### 7.1 Fixed

**The tick slice dropped the drafter on every park and every finished request.** `park_speculative_slice` and `finalize_speculative_slice` carried no `LoadedModel::Glm4MoeLite` arm, so both fell into the defensive `_ => None` and dropped the drafter handle instead of returning it to the worker slot, reloading the split-out drafter from disk once per request. The tree's own guard, `every_mtp_dispatch_site_covers_every_capable_variant` (#1165), reads `mtp_capable_target`'s body and requires every variant named there to appear at all seven dispatch sites, and it was already failing on both. It did not fail visibly because the narrow test filters this branch was validated with never reach `server::batch::speculative_burst_tests`. Every site has a `_ =>` arm, so a missing family silently declines rather than failing to compile, which is exactly the shape of defect that guard exists for.

**`decompose_kv_b_proj` converts its geometry with `i32::try_from`** and range-checks the head-dim product, covered in section 2.4.

**`split_mtp_dir` refuses an `--output` that aliases the source**, and `split_mtp` refuses an already-quantized nextn layer, both covered in section 2.3. Both carry unit tests, and the alias test asserts not only that the refusal fires but that the source `config.json` and `chat_template.jinja` are still intact afterwards.

Smaller corrections, none of them behavior an operator would have noticed before hitting the edge: `num_nextn_predict_layers + 1` is a saturating add in both the surgery op and the drafter config, so a `u64::MAX`-shaped field cannot wrap to a block size of 0; a config missing both `block_size` and `num_nextn_predict_layers` is now told which field is missing instead of being blamed for a `block_size` nobody wrote; `prefill_from_target_hidden` clears runtime state before its empty-prompt early return, so a recycled drafter cannot inherit the previous session's seed token and hidden (the server rejects empty prompts twice upstream, so this removes the dependence on those guards rather than fixing a reachable path); and `peek_drafter_model_type` gained the `Used by:` note that `docs/code-guidelines.md` requires now that it has a caller outside core.

### 7.2 Left open

**The probe measures one width.** `mtp_exactness_allows(block_size)` probes at the configured width, and this project's own benchmark guidance records that block-versus-chain disagreement varies strongly with width. The MTP arms have shared this shape since Gemma 4, so it is not a regression, but the contract is strictly measured only at the width probed.

**`ProbeKey` has no family discriminator.** It is `{block_size, hidden_size, num_hidden_layers}` and assumes a process serves one target model, which router mode does not. Pre-existing and shared with every other MTP arm and with #1751, so it belongs to a change that touches the memo itself.

**`--draft-block-size` is unbounded while `split-mtp --block-size` is capped at 16.** The cap bounds what a checkpoint can put into effect; the operator flag is not untrusted input, so the asymmetry is correct in kind, but the per-row verify cost that motivated the cap is reachable from a typo on the flag. The same observation was recorded for DSpark in #1751, and the two now share one answer when someone writes it.

**`rollback_speculative_cache_for_sequence` returns only layer 0's trim count** and the sole caller discards it. All layers advance in lockstep so a divergence should be impossible; nothing would notice if it were not.

---

## 8. The rebase onto #1751

LFM2 / LFM2.5 DSpark (#1751) landed in the same files and the conflicts were all in the speculative subsystem. Every one resolved by keeping both rather than choosing:

- The two block-size peeks key on different `DrafterKind`s. #1751's `peek_dspark_configured_block_size` is on the `Dflash` arm and parses a full `DFlashConfig`; this branch's `peek_glm4_moe_lite_mtp_configured_block_size` is the third `Mtp` peek and self-identifies by `model_type`. Core factored the shared body into `peek_configured_block_size_for(path, expected_model_type)` so a same-named field in an unrelated drafter shape cannot be read as a block-size hint, and a test pins both directions of non-interference.
- `DrafterError` carries both new variants as adjacent, unrelated arms: `GreedyOnly` from #1751 and `BinaryCrateDrafter` from this branch.
- In `speculative_burst.rs` the import splits. `load_drafter` now resolves to the binary-crate wrapper this branch needs, while `sampler_is_greedy` keeps coming from `mlxcel_core::drafter::dflash::drafter` where #1751 put it.
- `model_variant_label` and the burst's decline messages carry both families' arms.

---

## 9. Validation

Real checkpoints on Apple M5 Max, release binary rebuilt from this tree after the rebase. Target `models/glm-4.7-flash-4bit` (16 GB); drafter `models/glm-4.7-flash-mtp-4bit` (705 MB, 54 tensors, `block_size: 2`) produced by `mlxcel split-mtp --model models/glm-4.7-flash-bf16 --output models/glm-4.7-flash-mtp-4bit --q-bits 4`. Every CLI run is `mlxcel generate -m models/glm-4.7-flash-4bit -p "Explain in two sentences why the sky is blue." -n 128 --temp 0 --show-reasoning` with `MLXCEL_PRINT_TOKEN_IDS=1`, ids compared with `cmp`.

**Four-arm CLI identity.** Classic and MTP, `MLXCEL_FUSED_MOE` unset and `=0`, all under `MLXCEL_QMV_WIDE=0`: id-identical over all 128 tokens, md5 `f22d2a4b`. Re-run three times, before the review fixes, after them, and after the rebase, with the same md5 each time. `MLXCEL_FUSED_MOE` changes nothing in any arm because this family never dispatches the fused single-token MoE kernel.

**Acceptance.** Mean accepted length 1.753 (55 of 73 proposals), unchanged across all three runs. The pre-norm tap measured 1.716 (53 of 74) and emitted the same ids.

**The one divergence is the known kernel effect, not the MTP path.** A classic run with the default wide `qmv` kernel differs from every other arm at one near-tie position, and it also differs from the narrow classic run, which is the #1199 kernel-selection effect the exactness gate exists for. The gate logs it: `MTP exactness probe failed under qmv_wide ... passed without it. Disabling qmv_wide for this process`.

**Opt-in real-checkpoint gate.** `MLXCEL_TEST_GLM_MTP_TARGET=models/glm-4.7-flash-4bit cargo test ... real_checkpoint_verify_block_matches_decode_chain`: 128 positions, 0 byte mismatches, worst abs diff 0.0, 0 argmax flips. The test is teacher-forced so nothing is lost to divergence, pins the process narrow with `set_qmv_wide(false)` exactly as the gate does on generation 15 and newer, and asserts on disagreement at decided positions rather than on byte identity, per `docs/benchmarks.md`. This is the test that caught the unmaterialized query-view drift.

**Server.** Port 19326, both sides under `MLXCEL_QMV_WIDE=0`: `mlxcel-server -m models/glm-4.7-flash-4bit --model-draft models/glm-4.7-flash-mtp-4bit` against the same server with no drafter, `POST /completion` with the chat-templated prompt at `temperature: 0`, `n_predict: 128`, `return_tokens: true`. Both stop at EOS after 36 ids and the two id streams are identical. Startup logs `MTP exactness probe passed: verify block is byte-identical to the single-token chain block_size=2`; the request served through the tick slice at 18 rounds, 16 accepted, `emitted_per_verify` 1.889, and `/v1/internal/mtp-policy` reports `profiling` with `acceptance_rate: 0.889`.

**Quantized KV.** `--kv-cache-mode int8`: classic and MTP id-identical over 64 tokens, banner `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)`, which is #1350's MLA-latent substitution and the reason the slot mismatch of section 5 was never operator-visible.

**Suites.** `cargo test --profile test-fast` on `-p mlxcel-surgery` (160), root `--lib models::glm4_moe_lite` (33), `-p mlxcel-core drafter::` (179) and `mla::` (41), `--bin mlxcel commands::split_mtp` (2). `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` at root and for both member crates, plus `cargo clippy --bins`, all clean; `cargo fmt --all -- --check` clean.

**What is not established.** The full `--workspace` gate was not run locally (the machine is shared with other agents, so narrow scopes only), and CI covers it. Batched `B > 1` windows decline to classic by design and were not exercised. The acceptance comparison between the two hidden taps is one prompt on one host; it agrees with the DeepSeek-V3 convention, which is what carries the choice. The exactness contract is measured at `block_size = 2` and at whatever width the probe runs, not across widths.

---

## 10. Change summary

| Item | Value |
|------|-------|
| Files changed | 30 |
| Lines added | +5263 |
| Lines deleted | -145 |
| New tests | 47 |

| Area | Summary |
|------|---------|
| Surgery | `ops/split_mtp.rs` (new, 617 lines): the rename, the `kv_b_proj` decomposition through the shared helper, expert stacking, the bf16 pass with its float32 exemption, optional affine quantization with its router exemption, and the drafter `config.json` writer; the index-only nextn probe and the shard-filtered load; `--block-size` bounded at 16, an output aliasing the source and an already-quantized nextn layer both refused. `safetensors` moved from dev-dependencies to dependencies for the writer |
| Core | `mla/kv_b_split.rs` (new): `decompose_kv_b_proj` and `KvBProjGeometry`, lifted verbatim out of `glm4_moe_lite_sanitize.rs` (which drops from 167 lines to 84) with the `as i32` casts replaced by `i32::try_from` and a `checked_mul` range check; `GLM4_MOE_LITE_MTP_MODEL_TYPE` and the kind-map entry, `DrafterError::BinaryCrateDrafter`, `peek_drafter_model_type` made public, and the two narrow block-size peeks factored onto one body |
| Drafter | `glm4_moe_lite_mtp_drafter.rs` and `_config.rs` (new): the stateful drafter over the decoder's own `TransformerBlock` at `model.mtp_block`, fail-closed weight inventory, the trim-and-extend accept arithmetic, the seed fast path, and `runtime_block_size` capped at the trained depth; `drafter_loader.rs` (new) is the binary-crate entry point every caller goes through |
| Target | `glm4_moe_lite_mtp_hooks.rs` and `_target.rs` (new): the absorbed-MLA verify forward with per-row materialized queries, the model-owned per-sequence slot, trim-based rollback, the block-vs-chain probe, and the `MtpTarget` adapter with its post-final-norm hidden tap; `Glm4MoeLiteModel` gains `num_nextn_predict_layers`, `forward_with_hidden`, `TransformerBlock::from_weights_with_prefix`, the `KvCacheLayerModes` table and both mode trait methods |
| Server / CLI | `Glm4MoeLite` admitted at all seven MTP dispatch sites plus the tick slice and the speculative bench; `mtp_adopted_prefix_reusable` declines an adopted prompt-cache prefix in both run arms; the offline path injects the resolved KV mode table before dispatch; `split-mtp` subcommand with its `--force` checkpoint-marker guard |
| Docs | `supported-models.md` MTP row and family-list entry (including the APC non-combination), `mtp-policy-api.md` family list, `environment-variables.md` rows for `MLXCEL_GLM_MTP_HIDDEN_TAP` and `MLXCEL_PRINT_TOKEN_IDS`, README bullet and usage block |

---

## 11. Follow-up

- `--draft-block-size` remains unbounded on the CLI while `split-mtp --block-size` is capped at 16. A warning rather than a gate is the right shape, and #1751 left the same note for DSpark, so one change can answer both.
- Resolved by #1763: the `MAX_BLOCK_SIZE` refusal message carried two runs of 14 literal spaces because a wrapped string literal had lost its `\` continuations and rustfmt joined the line; the message now renders as one sentence and its test asserts no run of two or more spaces.
- Resolved by #1763: `--force` now refuses rather than deletes when the output directory holds any `*.safetensors` other than the drafter's own `model.safetensors` (a previous checkpoint's `model-*.safetensors` shards, or a `consolidated.safetensors`), naming the file and leaving every file untouched; it still clears a stale index when no such file remains.
- Resolved by #1763: `split-mtp` now validates `--q-bits` against `SUPPORTED_AFFINE_BITS` (`{2, 3, 4, 5, 6, 8}`) and `--q-group-size` against `SUPPORTED_AFFINE_GROUP_SIZES` (`{32, 64, 128}`) before any tensor work or directory change, so the help text and the enforced set agree; the shared load-time `validate_quantization_params` bounds check (`1..=32`) is unchanged.
- Batched `B > 1` MTP for this family stays out of scope, and so does drafting deeper than `num_nextn_predict_layers` by default.
- The hidden-tap acceptance gap (1.753 against 1.716) rests on one prompt. If this pairing ever gets a benchmark entry, measure the tap across several prompts there rather than re-deriving it from the module doc.
- Issue #1289 (order-preserving streamed `qmv`) is the route to passing the probe on the wide kernel rather than pinning the process narrow. Until then the pairing is byte-identical only under `MLXCEL_QMV_WIDE=0`, which the gate applies inside a server process and which a standalone `mlxcel generate` comparison has to set by hand.
