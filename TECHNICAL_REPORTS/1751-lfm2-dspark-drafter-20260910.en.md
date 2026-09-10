# Technical Report: PR #1751 - feat(speculative): LFM2 / LFM2.5 DSpark drafter on the DFlash loop

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation plus review follow-up cycle
**Status**: Completed (validated on Apple M5 Max / Metal against `models/lfm2.5-2.6b-bf16` and `models/lfm2.5-8b-a1b-bf16` with their published drafters; the `--workspace` gate was not run locally because the machine is shared, and the narrow scopes plus CI cover it)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a second family joins the DFlash round loop, the server's Qwen-only DFlash target contract becomes a trait, and two LFM2 loader defaults change for checkpoints that were previously unloadable)

---

## Executive Summary

LiquidAI publishes "DSpark" drafters for the LFM2.5 family. A DSpark drafter is a 5-layer DFlash-style block-parallel transformer fed five of the target's residual streams through an `fc` projection, plus a learned rank-256 low-rank token-transition head that turns the block's parallel logits into a sequential chain. mlxcel already served `lfm2` and `lfm2_moe`, and already had the DFlash round loop, the drafter trait and the target trait, so this change is four missing pieces rather than a new subsystem: the Markov head, the DSpark config fields, the LFM2 target adapter, and a rollback for a state that cannot be trimmed.

The last of those is the interesting one. Every speculative target so far rewinds a partial accept by shortening something. An attention KV cache trims: `offset` moves back and the next append overwrites. A short-conv state cannot, because it is not a prefix of anything; it is the last `L_cache - 1` rows of the layer's gated input, and after a block of `bs` rows it holds rows `bs - 2` and `bs - 1`. Truncating the block at `n` committed rows means the state should hold the last two rows of `concat(previous_state, bx_block[:, :n])`, which is a different tensor, not a slice of the one in hand. So the verify forward records what it would need to rebuild any prefix, and the rollback recomputes rather than trims.

The second structural change is that the server's `Qwen35DFlashTarget`, a private trait with one method, becomes `DFlashTargetModel` with four. The refactor is deliberately conservative: every hook a second family needed got a default that reproduces the Qwen 3.5 behaviour exactly, so the Qwen path's control flow through the generic driver is the old function with the names changed.

---

## 1. What the checkpoint actually says, and what mlxcel read

### 1.1 Four config fields the DFlash loader dropped

`DFlashConfig::from_json` has no `deny_unknown_fields`, so a DSpark `config.json` parsed without complaint and lost everything that makes it a DSpark drafter:

```
block_size 9
dflash_config: { mask_token_id 125017, target_layer_ids [2, 9, 17, 21, 27], num_target_layers 30 }
markov_rank 256, rope_is_neox_style false, enable_confidence_head true, markov_head_type "vanilla"
```

`markov_rank`, `rope_is_neox_style`, `runtime_block_size` and `enable_confidence_head` are now fields with defaults chosen so an existing Qwen 3.5 DFlash checkpoint parses to exactly the values it parsed to before (`markov_rank: 0`, `rope_is_neox_style: true`, `runtime_block_size: None`, `enable_confidence_head: false`). `dflash_defaults_keep_neox_rope_and_no_markov_head` pins that against `json!({})`.

`num_target_layers` needed a second change. The Qwen 3.5 checkpoints declare it at the top level; the DSpark ones nest it inside `dflash_config` alongside `mask_token_id` and `target_layer_ids`, both of which the lift already handled. Without the third key a 30-layer LFM2.5-2.6B pairing silently read the Qwen default of 32, and the pairing gate that compares it against the target's layer count would then have rejected a correct pair.

### 1.2 `block_size` counts something different

This is the field most likely to be misread. A DFlash `block_size` counts verify rows: the bonus row plus the proposals. A DSpark `block_size` counts proposals only. The published value 9 therefore means gamma = 9 and a verify width of 10, not a verify width of 9.

`DFlashConfig::verify_width()` returns `block_size + 1` when `is_dspark()` and `block_size` otherwise, and `runtime_verify_width()` caps that by `runtime_block_size`, which the published checkpoints omit and which then reads as the eight-row default (seven proposals plus the anchor). `--draft-block-size 10` restores the trained width.

The consequence for the CLI is that the flat `DEFAULT_DFLASH_BLOCK_SIZE` of 16 is wrong twice over for a DSpark drafter: it would ask a 9-proposal head for 15 proposals. `resolve_draft_block_size` gains a third peek, `peek_dspark_configured_block_size`, which returns `None` for anything that is not a DSpark drafter so a Qwen 3.5 DFlash checkpoint keeps the flat 16 exactly as before.

### 1.3 Detection keys on the head, not the name

`is_dspark()` is `markov_rank > 0`, not an `architectures` match. The head's presence is what changes the draft step, and the rank is what sizes it, so the field that decides is the field that matters. `architectures: ["Lfm2DSparkDraftModel"]` is still accepted by `is_dflash_drafter_config`, which is the separate question of whether a directory is a drafter rather than a standalone model; that check is what makes `mlxcel generate -m <dspark-dir>` fail with an explanation instead of `Weight not found: model.embed_tokens.weight`.

---

## 2. The draft step

### 2.1 Every position is a proposal

A plain DFlash draft builds `[bonus, mask, ..., mask]` at `block_size` rows and throws position 0 away: that row is scaffolding whose logits restate what the target already decided. A DSpark draft builds `[anchor, mask * (gamma - 1)]` at gamma rows and keeps all of them, because in the DSpark layout the anchor row already sits one past the committed context, so position 0's logits are a proposal for the next token rather than a restatement of the last one.

Getting this wrong is not a crash. It silently costs one proposal per round and shifts every remaining one, which reads as poor acceptance rather than as a bug, so `dspark_draft_block_uses_all_positions` asserts it directly with position-indexed synthetic logits.

### 2.2 The Markov chain

The five-layer forward is non-causal over the block, so position `i + 1` was scored without seeing what position `i` will be. The head repairs that sequentially:

```
prev = anchor
for step in 0..gamma:
    step_logits = base_logits[:, step] + markov_w2(markov_w1[prev])
    prev = argmax(step_logits)
    draft[step] = prev
```

`markov_w1` is a `[vocab, rank]` embedding table and `markov_w2` a `rank -> vocab` projection, so a step costs one gather and one `[1, 256] x [256, 128000]` matmul. Both load through `UnifiedEmbedding` / `UnifiedLinear`, which means a quantized drafter's `.scales` / `.biases` siblings would be picked up without a second code path.

The chain is built as one lazy MLX graph and materialized once. Each step's argmax stays on the device and feeds the next step's gather directly, so a round costs one host synchronization instead of gamma of them. That is worth stating explicitly because the naive reading of a sequential chain is that it must round-trip per step.

`markov_chain_feeds_previous_token_into_next_step` is the test that would catch a chain that ignores its own output: with flat base logits and a `w2` that spikes `prev + 1`, the only way to emit `anchor+1, anchor+2, ...` is to actually feed the argmax forward.

### 2.3 RoPE pairing

`rope_is_neox_style: false` means the checkpoint rotates dimension `i` with `i + 1` (GPT-J style), which is MLX's `traditional = true`. `DFlashAttention` hard-coded `false` at all three `fast_rope` call sites. It now carries `rope_traditional: !config.rope_is_neox_style`, and the test asserts the choice reaches the output rather than just the struct field, by running the same input through both pairings and requiring the attention outputs to differ.

---

## 3. The rollback that cannot trim

### 3.1 Why a short-conv state is not a prefix

`Lfm2LayerCache::Conv` holds `[1, L_cache - 1, hidden]`, the last two rows (`L_cache = 3` on every published checkpoint) of the padded conv input. `ShortConv::forward` overwrote it and kept nothing, and the only existing trim hook reset it to `None`, which is correct for a cache eviction and wrong for a partial accept: it would restart the recurrence from zeros mid-sequence.

With `n = accepted + 1` committed rows, the state the next round needs is the last `L_cache - 1` rows of `concat(previous_state_or_zeros, bx_block[:, :n])`. Neither operand survives an ordinary forward, so `forward_with_capture` records both:

```rust
snapshots.push(ConvRollbackSnapshot {
    layer_idx,
    prev_state: conv_state.as_ref().map(|s| mlxcel_core::copy(s)),
    bx_block: mlxcel_core::contiguous(&bx, false),
});
```

The capture is placed after `bx` is computed and before `*conv_state` is assigned, and `mlxcel_core::copy` is `mlx::core::copy`, a real copy node rather than a handle alias. An aliased handle would have been subtly wrong in a way no shape check catches: the rollback would rebuild the state from the state the block just produced, which is exactly the value it is trying to undo.

The zeros fallback for a `None` prior state is `[bx.shape[0], L_cache - 1, bx.shape[2]]` in `bx`'s own dtype, which matches what `conv_padding()` left-pads a fresh causal forward by, `(l_cache - 1, 0)`.

### 3.2 The attention half, and what a full accept does

Attention layers take `KVCache::trim(block_size - n)`: `offset` moves back, the buffers stay, the next append overwrites. A full accept calls none of this; every cache stays as the verify forward left it, which is already the committed state.

### 3.3 What the test compares

`conv_rollback_matches_committed_prefix` does not assert shapes. It prefills five tokens, verifies a four-row block, rolls back at `accepted = 1`, and then compares against a reference model that consumed `prompt ++ verify[:2]` in one pass: the attention offset (7), the conv state (`allclose` 1e-5), and the next decode step's logits on both (`allclose` 1e-4). The last of those is what makes the test worth having, because a conv state can be close enough to pass a state comparison and still move a token.

`conv_rollback_from_fresh_caches_pads_with_zeros` covers the `prev_state: None` branch the same way, against a one-token prefill.

---

## 4. Hidden-state capture and the first draft

The drafter's `fc` reads `len(target_layer_ids) * hidden_size` features, so the target has to hand back one `[1, bs, 2048]` slab per captured layer, taken after that layer and before the final `embedding_norm`, concatenated on the feature axis to `[1, bs, 10240]`. `forward_speculative` is `forward_embeds_with_caches` with the capture and the snapshot sink threaded through: the same causal mask anchored on the first attention layer's offset, the same left-padded conv, the same MoE routing. That sameness is the exactness premise, and section 6 is about whether the kernels honour it.

One policy difference between the two families turned out to matter enough to name. The Qwen 3.5 DFlash path slices the prefill's captured hidden to the last prompt position, because its drafter starts its context cache from the bonus token onwards. A DSpark drafter appends every row it is handed to its own append-only context cache and wants the whole prompt before its first proposal. Feeding it one row instead is not a shape error anywhere; it silently leaves the drafter attending a one-row context and degrades acceptance. `FirstHiddenRows` makes the choice explicit and `first_hidden_rows_defaults_to_the_qwen_policy` pins both families' answers, so a third family that forgets to override is caught by a test rather than by a benchmark.

---

## 5. `DFlashTargetModel`

`DFlashGenerator` already accepts any `SpeculativeTarget`. What was still Qwen-shaped was the server side of a burst: allocating the target's heterogeneous cache vector, reading logits and hidden slabs off a family-specific `VerifyOut`, and deciding the first-draft row policy. `Qwen35DFlashTarget` carried the first of those and named the other two in its bounds.

The replacement carries four things: `make_dflash_caches`, `first_hidden_rows` (an associated function, since the policy belongs to the drafter family rather than to a loaded instance), `enable_speculative_buffers` (a no-op hook for a target whose caches need arming for a multi-row verify), and `exactness_allows`. A companion trait `DFlashVerifyOutput` gives the driver logits and hidden slabs without naming any family's output type.

Three properties made the refactor safe to land under live code:

- Every added hook has a default that is the Qwen 3.5 behaviour. `first_hidden_rows` defaults to `LastPromptPosition`, `enable_speculative_buffers` to a no-op, `exactness_allows` to `true`, and `Drafter::validate_target_compat` was already a no-op default. `Qwen35Model` overrides only `make_dflash_caches`.
- The two helpers that replaced inline code are equivalent, not similar. `concat_captured_hidden` is the old copy-then-fold loop expressed as `concatenate_many(refs, -1)`, and `first_hidden_for(LastPromptPosition, ..)` is the old slice verbatim.
- No inherent method shadows a new trait method. `Qwen35Model` has `mtp_exactness_allows`, not `exactness_allows`, so `m.exactness_allows(bs)` in the burst gate resolves to the permissive trait default and the Qwen DFlash path does not begin gating on a probe it never gated on.

The one thing the trait deliberately does not carry is a per-round block-size override. `DFlashGenerator` holds one `block_size` for the whole run, so a hook here would read as supported and do nothing.

---

## 6. The exactness gate, and what the host said

The temperature-0 contract is that the emitted stream is identical to classic greedy decode. That holds if and only if every verify row's logits equal what a single-token decode step at that position would produce, and on quantized checkpoints the `M = bs` matmuls and the `M = bs` MoE dispatch are not guaranteed bitwise equal to `M = 1`. LFM2's MoE block in particular takes its fused single-row expert kernel only when `x_flat.shape[0] == 1`, so a verify block never takes the path a decode step takes.

Rather than assert the premise, the change reuses the block-versus-chain probe the MTP arms use: three synthetic draws, a block of `bs` tokens against the same tokens decoded one at a time from the same prefilled state, logits compared bitwise. `mtp_exactness_gate` owns the memoization, the decline log line, the `qmv_wide` retry and `MLXCEL_MTP_ALLOW_INEXACT`, whose meaning is unchanged. The burst adds its own named warning on decline, because the shared gate's verdict line says "MTP declined", which an operator who configured `--draft-model` with a DSpark drafter has no reason to connect to their request falling back.

**On the validation host (Apple M5 Max, GPU generation 17) the probe declines for both pairs.** That is the documented generation-15-and-newer condition and it is the shipped default: the burst falls back to classic decode and serves the baseline's tokens. The measurements below therefore ran under `MLXCEL_MTP_ALLOW_INEXACT=1`, and what they show is worth recording precisely because it is the thing the gate exists to prevent.

| Pair | Width | Mean accepted | tok/s | Classic tok/s | Token ids vs baseline |
|------|-------|---------------|-------|---------------|-----------------------|
| Dense 2.6B | 8 | 3.59 | 203.3 | 72.3 | 256 / 256 |
| Dense 2.6B | 10 | 3.75 | 212.1 | 72.3 | 256 / 256 |
| MoE 8B-A1B | 8 | 2.61 | 115.5 | 104.3 | 153 / 256 |
| MoE 8B-A1B | 10 | 2.88 | - | 104.3 | 153 / 256 |

The MoE divergence at position 153 is a tie the two arms resolve differently: the baseline takes `':'` at logprob -1.0312 and the block path `' formula'` at -1.0000, and on a shorter prompt both candidates read -1.4375. It reproduces at the same position across both widths and across runs, while the fused and unfused classic arms agree with each other. That is precisely the byte-identity the probe declined, observed after overriding the decline, and it is the reason the override is not the default.

Greedy-only enforcement is a separate gate and it works on this host: `"temperature": 0.7` on the dense pair is served by classic decode with no `timings` and exactly one line, `DFlash speculative dispatch declined for seq seq-2: the drafter is greedy-only (DSpark) and the request samples with temperature 0.7 / top_k 50; serving it with classic decode`. The decline is decided after the drafter is resident and before it is taken, so the drafter stays in its slot for the next greedy request.

---

## 7. Two LFM2 loader fixes that came along

Both are prerequisites rather than scope creep: the bf16 originals cannot load or cannot load correctly without them, and the bf16 originals are what the published drafters pair with.

**`rope_theta` under `rope_parameters`.** The LiquidAI originals are transformers 5.x exports and carry no top-level `rope_theta`; it lives in a `rope_parameters` block (1e7 for LFM2.5-2.6B, 5e6 for 8B-A1B). The old field had a `#[serde(default)]` of 1e6, so those checkpoints ran every attention layer at the wrong base and produced plausible-looking wrong output. `ModelArgs::rope_theta()` now reads the top-level key first (the mlx-community conversions keep it, and it wins when both are present), then the nested one, then the 1e6 first-release default. All three branches are tested.

**Unstacked per-expert MoE tensors.** The MoE originals ship `feed_forward.experts.{e}.w1/w2/w3`. The rename step only matched the dense `.feed_forward.wN.` form, so the expert tensors kept their `wN` names, the stacking probe that looks for `experts.{e}.gate_proj` never fired, and the load failed on `switch_mlp.gate_proj`. The rename now covers both forms. The mlx-community 4-bit conversions ship pre-stacked `switch_mlp.*` tensors and match neither form, so they are untouched.

---

## 8. Review findings, and what was done about them

The review found no CRITICAL and no HIGH. Of what it did find, the one MEDIUM that was a regression this branch introduced is fixed on the branch, along with three doc statements the review showed to be false and two unbounded checkpoint-supplied numbers found while fixing them. The rest is recorded so it is not rediscovered.

### 8.1 Fixed

**A mismatched non-DSpark pairing on an LFM2 target had no gate.** `validate_target_compat` returns `Ok` immediately when the drafter is not DSpark, and `DFlashDraftModel::forward` does not check its `fc` input width, so an LFM2 target paired with a genuine Qwen 3.5 DFlash drafter reached an MLX shape throw through a non-`Result` cxx shim, which aborts the process rather than failing the request. The hazard is pre-existing in kind (a Qwen 4B target with a 27B DFlash drafter fails the same way), but LFM2 targets used to decline at the variant gate and after this branch they do not, which makes it a regression rather than an inherited gap.

The drafter cannot make this call. It reads the target as a `LanguageModel`, which carries no architecture string, so a DFlash drafter has no way to tell an LFM2 target from the Qwen 3.5 one it was published for. The target is the side that knows its own family, so the policy went on the trait as `DFlashTargetModel::requires_dspark_drafter()`, an associated function for the same reason `first_hidden_rows` is one: it is a property of the family, not of a loaded instance, so the test pins it without a checkpoint. It is `true` for `Lfm2Model` and `Lfm2VlModel` and the permissive default everywhere else. Both run arms read it before any forward and answer with one operator-facing message naming `--model-draft` and the published pairings. The broad alternative, an `fc`-width check for every DFlash drafter, would change which Qwen pairings are accepted and stays a maintainer call.

**Two checkpoint-supplied numbers reached a buffer index or a round size unbounded.** Both are now in the DSpark pairing gate, beside the `fc`-width and `target_layer_ids` checks that were already there. `mask_token_id` indexes the target's embedding table, because a DSpark drafter ships none of its own, and MLX range-checks no positive gather index: an id past the last row read whatever followed the table in the buffer and fed it to the logits. The target half of the same gate already pins the target's vocabulary to `vocab_size`, so bounding the id against `vocab_size` bounds the gather. Separately, a `runtime_verify_width()` below two rows proposes nothing and would emit one token per burst, and `verify_width()` now saturates rather than wrapping, since `usize::MAX` there panics in a debug build and wraps to a zero-row width in a release one.

**The Markov head's factors were never measured against the config that sizes them.** `markov_w1` is gathered at a token id once per chain step, so a table with fewer rows than the vocabulary the chain draws from read past its own buffer for the same reason the mask id did; a `markov_w2` of the wrong width threw inside MLX instead, which crosses the bridge as a process abort rather than a load error. `VanillaMarkovHead::from_weights` now takes the vocabulary as well as the rank and checks both factors before either becomes a layer. A quantized factor bit-packs `rank` along its last axis only, so its row count reads the same either way while its stored width is a function of the bit depth; the rank check is therefore skipped for a packed table and the row check is not.

**Three doc statements the review showed to be false**, no behavior change in any of them. `ShortConv::forward_with_capture` claimed the snapshot costs no copy while the code takes an explicit one; it has to, because `conv_state` is reassigned at the end of the call and an alias would read back as the post-block state. `configured_block_size` implied a DFlash round-loop block-size policy that does not exist. And the `dflash_target` module doc claimed a new family costs one `impl` block plus a match arm.

### 8.2 Left open

**A third family touches three match arms, not one.** The module doc now says so. The arms are the exactness gate, the `drive!` dispatch and the batched gate; the fourth the review counted, `model_variant_label`, is the pre-existing project-wide label table (#1613) and not specific to DFlash. The arms exist because the burst reaches the target as a `LoadedModel` enum and only a match recovers the concrete type the trait is implemented on, so they cannot be removed, only made uniform. Two of the three already are. The batched gate is not: it still carries the LFM2 B = 1 decline as a hardcoded arm, which is the one piece of per-family policy living outside the trait. A `supports_batched()` hook is the fix, and it belongs to the next family that needs it rather than to this branch.

**The probe measures one width.** `dflash_exactness_allows(block_size)` probes at the configured verify width, but the round loop narrows `bs` near the token budget (`bs = block_size_cfg.min(remaining_plus_one)`), and this project's own benchmark guidance records that block-versus-chain disagreement varies strongly with width. The MTP arms have the same shape, so this is not a regression, but the contract is strictly only measured at the wide width.

**`ProbeKey` has no family discriminator.** It is `{block_size, hidden_size, num_hidden_layers}`, and its docstring assumes a process serves one target model. Router mode does not. Pre-existing, and widened by a third family reaching the same memo.

**Inert block-size hooks.** `configured_block_size()` and `prefer_requested_block_size()` are implemented on `DFlashDrafter` as the issue asked, but only the MTP generator and its batched round loop read them; the DFlash round loop uses its constructed `block_size` directly. The effective width for DSpark comes from `resolve_draft_block_size`, which peeks the drafter config before the drafter is loaded and passes the same number in. The requirement that the drafter never backs off on low acceptance holds, just vacuously. Both are kept implemented, with the situation now stated on them, so the two agree if the DFlash loop ever grows an adaptive width.

**Benchmark numbers stayed in the PR body.** The issue asked for tok/s at both widths in `docs/benchmark_results/`. The PR records them in its body instead, and given that the host served other work and each throughput figure is one sample, keeping single-sample numbers out of the benchmark corpus is the right call under this project's benchmark discipline. Recording the deviation here is the point.

---

## 8b. Security review, and what was done about them

A second pass looked specifically at what a downloaded checkpoint controls. Everything in this section shares one premise: pointing `--model-draft` at a repository makes every field of that repository's `config.json` and every weight shape in it a runtime input, MLX range-checks no positive gather index, and an MLX C++ exception crossing the cxx bridge aborts the process rather than failing the request. It found one HIGH, four MEDIUM and three LOW. Six are fixed; two are left with the reason.

### 8b.1 Fixed

**HIGH: the verify width was bounded below and not above, and this branch is what made it a control input.** Before it, the DFlash runtime block size came only from `--draft-block-size` or the flat constant 16, both operator-supplied. `resolve_draft_block_size` now peeks the drafter checkpoint through `peek_dspark_configured_block_size` and hands `runtime_verify_width()` to the scheduler as the server-wide block size. That width is `min(block_size + 1, runtime_block_size)`, and a config omitting `runtime_block_size` is capped at the eight-row default, so a large `block_size` alone was harmless; a config setting both escaped the cap entirely.

What it reaches first is the worst of the two reach points. The exactness gate runs `exactness_allows(bs)` on the first LFM2 request, before the drafter is even loaded, and `probe_one_draw` runs one single-token target forward per row, three draws over, doubled again by the `qmv_wide` retry, on the scheduler thread. At a million rows that never completes, and because the verdict is memoized on `ProbeKey` it never retries either. It also runs before `validate_target_compat`, so the pairing gate could not have refused the config in time.

The fix is therefore in two places on purpose. `runtime_verify_width()` clamps at `DSPARK_MAX_VERIFY_WIDTH` (32, against 8 to 10 on the published checkpoints and 16 for flat DFlash), which is the bound that actually holds ahead of the probe because every caller goes through that one function. The pairing gate separately refuses a config whose `requested_verify_width()` (the unclamped number, exposed for exactly this) is above the ceiling, so an operator with a broken drafter is told rather than quietly served at 32. The test pins both halves, including the `usize::MAX` case the saturating add in `verify_width()` exists for.

**MEDIUM: the `mask_token_id` bound was vacuous for a self-contained drafter.** The bound added in the review-fix commit argued its soundness from the target half of the pairing gate pinning the target's vocabulary to `vocab_size`. That argument holds only on the lazy-bind path, which is what every published DSpark checkpoint uses. A checkpoint shipping its own `embed_tokens.weight` sets `needs_embed_binding()` false, takes the other arm of `bind`, and its table was never compared to anything, so an eight-row table under a declared `vocab_size` of 128000 still gathered row 125017 out of bounds. `LmHead::Own` had the same gap one step later: its width was never compared to `vocab_size`, which the Markov head IS measured against, so a mismatch failed broadcasting inside `ffi::add`. `from_weights` now measures both against `vocab_size` for a DSpark config. A plain DFlash checkpoint is deliberately left alone: it has no Markov head and no mask id indexing a borrowed table, and checking there would change which Qwen pairings load.

**MEDIUM: the Markov rank check was skipped, not narrowed, for a quantized factor.** The row check (the half that bounds the gather) held either way, but the width check did not run at all when `.scales` was present, so two internally consistent but mutually mismatched factors reached `quantized_matmul` and threw inside MLX. Every mlx-community conversion of these drafters is quantized, so the skipped case is the common one. MLX packs the last axis u32-wise (`packed_in * 32 == bits * in_features`), so the width is derivable: the check now accepts a packed width when some supported bit depth explains it, rather than trusting the loader's declared `bits`, which a per-tensor override can contradict.

**MEDIUM: `first_hidden_for` deep-copied the whole prompt hidden.** `mlxcel_core::copy` is a real MLX `Copy` primitive, not a handle clone, and the every-row arm called it only to satisfy a borrowed signature: a second full `[1, S, len(target_layer_ids) * hidden]` slab per request, 168 MB at an 8k prompt on LFM2.5-2.6B and 671 MB at 32k. It now takes the array by value and returns it unchanged, which also ends the caller's retention of the original across the whole round loop. The slicing arm is unaffected, since a slice holds its own reference to its input.

**MEDIUM: the short-conv rollback snapshots were captured on the prompt prefill.** Nothing reads them there, because a prefill is never rolled back. Each snapshot holds the layer's gated input at the forward's own width, and LFM2 is conv-dominant, so at prompt length this pinned roughly one prompt-sized buffer per conv layer alive through the eval that materializes the prefill: about 670 MB at 8k on a 30-layer 2.6B checkpoint, about 2.7 GB at 32k, on top of the model and the caches. `SpeculativeTarget` gains `prefill_forward_with_capture_layers`, defaulting to the verify hook so no other family changes, and LFM2 and LFM2-VL override it to skip the capture. `forward_speculative` takes the capture as a flag, and the new test pins that the flag changes what is kept and not what is computed, on the logits, the captured hidden and the cache offsets at once. Getting that wrong would break the temperature-0 contract silently, since the prefill and the verify rounds would then disagree.

**LOW: empty prompt and empty window.** `run_dflash_on_target` computed `last_pos = len - 1` and sliced at it; the batched arm indexed `prompts[0]`. Both were inherited unchanged from the Qwen arms, and neither is reachable from the scheduler today, but the LFM2 arms are new callers of the same code and a negative slice start reaches MLX as a process abort. Both are now request errors.

### 8b.2 Left as-is

**`sample_block_array`'s B = 1 invariant stays a `debug_assert`.** In release a `[B > 1, gamma, vocab]` input would silently chain from row 0 and drop the rest. It is not reachable: `draft_block_batched` returns `DraftFailed` for a DSpark drafter and the batched burst declines LFM2 before the take. The function is `pub` on a `pub` type, which is the real argument for hardening it, but the available hardening is a release panic in a request handler, which is itself a denial of service, and narrowing the visibility would be a breaking change to the library crate for a case no caller can reach. Recorded rather than changed.

**The mismatched-pairing decline stays `BurstOutcome::Error`.** `DeclineToClassic` would serve the request rather than fail it, which is more available. It is not chosen because the arm directly above it, `validate_target_compat`'s failure, is an `Error`, and because a silent fallback turns an operator's misconfigured `--model-draft` into a permanent unexplained slowdown. The message names the flag and the fix, and the warmup surfaces it at startup rather than only on the first request.

---

## 9. Change summary

| Item | Value |
|------|-------|
| Files changed | 29 |
| Lines added | +4331 |
| Lines deleted | -578 |
| New tests | 34 |

| Area | Summary |
|------|---------|
| Drafter core | `markov.rs` (new), DSpark config fields and `verify_width` / `runtime_verify_width`, RoPE pairing threaded into `DFlashAttention`, DSpark draft step and pairing gate, `DrafterError::GreedyOnly`, `Drafter::greedy_only`, `Drafter::is_dspark`; the pairing gate also bounds `mask_token_id` against the vocabulary and the verify width below two rows, and `VanillaMarkovHead::from_weights` measures both factors against the config before either becomes a layer |
| LFM2 target | `lfm2_speculative.rs` (new): verify forward with capture, conv rollback, exactness probe, `SpeculativeTarget`; `rope_parameters` and per-expert MoE rename fixes |
| Server | `dflash_target.rs` (new): `DFlashTargetModel`, `DFlashVerifyOutput`, `FirstHiddenRows`, both generic drivers; burst gate extended to the three LFM2 variants with a B = 1 restriction; `requires_dspark_drafter()` declines an LFM2 target paired with a non-DSpark DFlash drafter in both run arms before any forward |
| CLI | DSpark block-size peek in `resolve_draft_block_size`; offline rejection message names both drafter shapes |
| Docs | `supported-models.md` DSpark row, `speculative-acceptance.md` greedy-only decline, README |

Verified after the review and security fixes, rebased on `origin/main`: `cargo check` and `cargo clippy -- -D warnings` clean for the root package AND for `-p mlxcel-core` separately, plus `cargo fmt --all -- --check`. Checking only `--lib --tests` at the workspace root resolves to `-p mlxcel` and does not compile `mlxcel-core`'s test target, which is how two test-only compile errors reached a run of the suite; both packages are checked from here on. `-p mlxcel-core drafter::dflash` 74 passed (69 at review time), and one `--lib` run over `models::lfm2` / `server::batch::speculative_burst` / `server::batch::dflash_target` / `cli::speculative_args` / `models::detection` 184 passed (180 at review time).

---

## 10. Follow-up

- Issue #1343 (Muse Glimmer assistant drafter) is the next `DFlashTargetModel` implementor. It needs `enable_speculative_buffers` for its rotating cache, which is present and called by both drivers, and a rotating-cache `rollback_partial`, which is already per-family. Add `supports_batched()` while doing it, and decide there whether Muse wants `prefill_forward_with_capture_layers`: its rotating caches make the prefill the same shape of question LFM2's conv snapshots were.
- Issue #1289 (order-preserving streamed qmv) is the route to passing the probe on the kernels that decline today. Until then the DSpark burst is a measured decline on generation 15 and newer, not a measured speedup.
- Batched (B > 1) DSpark and a sampled acceptance rule are explicitly out of scope and remain so.
- The confidence head is loaded and never called; no early-exit policy is built on it.
- `ProbeKey` still has no family discriminator, which router mode makes wrong rather than merely narrow. It is pre-existing and shared with the MTP arms, so it belongs to a change that touches the memo itself.
- The `DSPARK_MAX_VERIFY_WIDTH` ceiling bounds what a CHECKPOINT can put into effect. `--draft-block-size` is still unbounded, which is correct in kind (an operator flag is not untrusted input) but means the probe's per-row cost is reachable by a typo. Worth a warning rather than a gate.
