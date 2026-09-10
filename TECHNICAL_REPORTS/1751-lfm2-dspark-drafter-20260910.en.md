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

## 8. Review findings

The review found no CRITICAL or HIGH issues. Nothing was changed on the branch beyond this report. What follows is recorded so it is not rediscovered.

**A mismatched non-DSpark pairing on an LFM2 target has no gate.** `validate_target_compat` returns `Ok` immediately when the drafter is not DSpark, and `DFlashDraftModel::forward` does not check its `fc` input width, so an LFM2 target paired with a genuine Qwen 3.5 DFlash drafter reaches an MLX shape throw through a non-`Result` cxx shim. The hazard is pre-existing in kind (a Qwen 4B target with a 27B DFlash drafter fails the same way), but LFM2 targets used to decline at the variant gate and now do not. The narrow fix is a burst-gate decline when the target is LFM2 and the drafter is not DSpark; the broad fix is an `fc`-width check for every DFlash drafter, which would change which Qwen pairings are accepted and so is a maintainer call rather than a review edit.

**A third family touches more than one match arm.** The module doc says "one `impl` block plus a match arm in the burst gate". The real count is four: the exactness gate, the `drive!` dispatch, the batched gate (which hardcodes the LFM2 B = 1 decline), and `model_variant_label`. A `supports_batched()` method on the trait would move the third of those into the trait where it belongs, which is worth doing when the next family lands.

**The probe measures one width.** `dflash_exactness_allows(block_size)` probes at the configured verify width, but the round loop narrows `bs` near the token budget (`bs = block_size_cfg.min(remaining_plus_one)`), and this project's own benchmark guidance records that block-versus-chain disagreement varies strongly with width. The MTP arms have the same shape, so this is not a regression, but the contract is strictly only measured at the wide width.

**`ProbeKey` has no family discriminator.** It is `{block_size, hidden_size, num_hidden_layers}`, and its docstring assumes a process serves one target model. Router mode does not. Pre-existing, and widened by a third family reaching the same memo.

**Inert block-size hooks.** `configured_block_size()` and `prefer_requested_block_size()` are implemented on `DFlashDrafter` as the issue asked, but only the MTP generator reads them; the DFlash round loop uses its constructed `block_size` directly. The effective width for DSpark comes from `resolve_draft_block_size`. The requirement that the drafter never backs off on low acceptance holds, just vacuously.

**Benchmark numbers stayed in the PR body.** The issue asked for tok/s at both widths in `docs/benchmark_results/`. The PR records them in its body instead, and given that the host served other work and each throughput figure is one sample, keeping single-sample numbers out of the benchmark corpus is the right call under this project's benchmark discipline. Recording the deviation here is the point.

---

## 9. Change summary

| Item | Value |
|------|-------|
| Files changed | 27 |
| Lines added | +3080 |
| Lines deleted | -577 |
| New tests | 25 |

| Area | Summary |
|------|---------|
| Drafter core | `markov.rs` (new), DSpark config fields and `verify_width` / `runtime_verify_width`, RoPE pairing threaded into `DFlashAttention`, DSpark draft step and pairing gate, `DrafterError::GreedyOnly`, `Drafter::greedy_only` |
| LFM2 target | `lfm2_speculative.rs` (new): verify forward with capture, conv rollback, exactness probe, `SpeculativeTarget`; `rope_parameters` and per-expert MoE rename fixes |
| Server | `dflash_target.rs` (new): `DFlashTargetModel`, `DFlashVerifyOutput`, `FirstHiddenRows`, both generic drivers; burst gate extended to the three LFM2 variants with a B = 1 restriction |
| CLI | DSpark block-size peek in `resolve_draft_block_size`; offline rejection message names both drafter shapes |
| Docs | `supported-models.md` DSpark row, `speculative-acceptance.md` greedy-only decline, README |

Verified locally at review time: `cargo check --lib --tests`, `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, and the narrow test scopes `-p mlxcel-core drafter::dflash` (69), `--lib models::lfm2` (29), `--lib server::batch::speculative_burst` (67), `--lib server::batch::dflash_target` (2), `--lib cli::speculative_args` (22), `--lib models::detection` (60). All clean.

---

## 10. Follow-up

- Issue #1343 (Muse Glimmer assistant drafter) is the next `DFlashTargetModel` implementor. It needs `enable_speculative_buffers` for its rotating cache, which is present and called by both drivers, and a rotating-cache `rollback_partial`, which is already per-family. Add `supports_batched()` while doing it.
- Issue #1289 (order-preserving streamed qmv) is the route to passing the probe on the kernels that decline today. Until then the DSpark burst is a measured decline on generation 15 and newer, not a measured speedup.
- Batched (B > 1) DSpark and a sampled acceptance rule are explicitly out of scope and remain so.
- The confidence head is loaded and never called; no early-exit policy is built on it.
