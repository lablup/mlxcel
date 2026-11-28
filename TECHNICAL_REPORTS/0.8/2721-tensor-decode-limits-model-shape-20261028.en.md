# PR #2721: Tighten tensor decode limits to the model shape

**Date**: 2026-10-28 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at aaa4daa7. The implementation review found one HIGH item (a stale test expectation), fixed in a91c64a8; the security and performance review found no CRITICAL, HIGH or MEDIUM issues. The local gate passed on GB10 (spark-102) at aaa4daa7. main was merged in at 2f137091 (bringing #2719 and #2720); the pre-merge gate on the merged tree and hosted CI were still running at the time of writing
**Languages**: Rust, Markdown
**Risk Level**: Medium (the frame format is unchanged, but every peer-facing pipeline decode now refuses frames that were accepted before, a stage fails to load when no inbound activation width can be determined, the empty-shape element count changes from 0 to 1, and `reassemble_layer_payload` and `LayerTransferHeader` change their signatures)

## Executive Summary

PR #2721 (closes #2716) makes the pipeline receivers say what they expect. Before it, every tensor a remote stage or the coordinator decoded was bounded only by the header the sender wrote, so an authenticated but compromised peer could hand a stage a ~21 MiB frame that decompressed to 4 GiB (and was then copied again into an MLX array) and could hand the stage executor a tensor of any rank, batch, sequence length or width. The PR adds a header-level limit check (`TensorDecodeLimits`, checked by `deserialize_tensor_with_limits` right after `TensorHeader::decode` and before the payload is sliced, copied, decompressed or allocated), makes the activation path refuse the COMPRESSED and QUANTIZED flags outright, and derives the bounds from the loaded model: hidden states must be `[1, n <= 2^20, width]` where `width` is the stage's inbound activation width, entry token ids `[1, n <= 2^20]` int32, masks rank 2 to 4 with every axis at most 2^20, and coordinator logits `[1, <= input_seq_len, vocab_size]`.

The width is not simply `hidden_size`. Gemma 4 checkpoints with per-layer inputs (E2B, E4B) pack the per-layer inputs of all remaining layers into the stage-boundary activation, so the PR adds `inbound_activation_width` to the stage executor traits with a Gemma 4 override that `unpack_stage_hidden` now also uses, so the packing width is computed in one place.

The other five items from #2716 are low-severity wire defects fixed in the same pass: the streamed KV layer header gains `original_bytes` and stops miscounting elements for odd byte counts, an unquantized layer must decode to exactly that many bytes, an empty shape is a 1-element scalar as in MLX, the decoder no longer copies a payload it is about to decompress, and `MAX_DECODED_TENSOR_BYTES` is a `u64` so it const-evaluates on 32-bit targets. All of these need an authenticated cluster member, so this is compromised-peer hardening, not a fix for an unauthenticated hole.

## 1. Problem Statement

#2701 (issue #2691) turned every peer-controlled size in the tensor wire format into a checked value, but the remaining bounds were the ones the sender itself declared. #2716 listed six residuals from section 7 of the #2701 report, all re-checked on main at 752db4f9.

### 1.1 Activation decode trusted the sender's header (medium)

The only production caller of `deserialize_tensor` was the pipeline activation path (`deserialize_wire_tensor` to `ActivationMessage::deserialize_activation`). Honest senders never set COMPRESSED or QUANTIZED, but the decoder honored both. Under a self-consistent header, the decompression cap was `min(header bound, MAX_DECODED_TENSOR_BYTES)`, and `decompress` accepts up to 256x the block size, so a ~16 MiB LZ4 block (~21 MiB after base64, well under the 256 MiB TCP frame cap) produced a 4 GiB buffer that `reconstruct_mlx_array` then copied into MLX: about 8 GiB of allocation per frame. Nothing checked rank, batch, sequence length or width before the tensor reached the executor. The peer-facing sites were the entry stage's `input_ids` and attention mask, a non-entry stage's hidden states and mask, and the coordinator's logits, which `validate_logits_shape` checked only after the full decode.

### 1.2 Streamed KV element count (low)

`prepare_layer_payload` counted `keys.len() / 2 + values.len() / 2`. For a 1-byte dtype with odd key and value byte counts this is one element short per tensor, so the level-None receiver, which bounded decompression at `original_num_elements * 2`, rejected a valid layer as 2 bytes over the limit. The quantizers count `(k + v) / 2` over the concatenation, so the same miscount made the Int8 and Int4 header disagree with the quantized layout.

### 1.3 Uncompressed level-None length unchecked (low)

`reassemble_layer_payload` returned `wire_data.to_vec()` for an uncompressed level-None layer without comparing it with anything in the header.

### 1.4 Empty-shape semantics disagreed (low)

`TensorHeader::num_elements` and `validate_data_size` treated `[]` as 0 elements, while `validate_raw_tensor` at the end of the wire path folds from 1, as MLX does. A `[]` header with 0 bytes passed `deserialize_tensor` and failed only later, and the encoder refused a real 2-byte float16 scalar. The overflow message in `validate_data_size` also embedded 18 spaces because a line-continuation backslash was missing.

### 1.5 Copy before decompress (low)

`deserialize_tensor` copied the payload with `to_vec()` before decompressing or dequantizing it, so every flagged frame paid for one extra full-size copy.

### 1.6 32-bit portability (low)

`pub const MAX_DECODED_TENSOR_BYTES: usize = 1 << 32` is a const-evaluation overflow on a 32-bit `usize`, so the crate could not compile there even in principle.

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `tensor_serialize.rs`, `distributed/mod.rs` | `DimBound { Exact, AtMost }`, `ShapeBound { Axes, Uniform { rank, max_dim } }`, `TensorDecodeLimits { allow_compressed, allow_quantized, dtypes, shape }` with `unrestricted()` and `check_header`. `deserialize_tensor_with_limits` calls `check_header` immediately after `TensorHeader::decode`; `deserialize_tensor` wraps it with `unrestricted()`. Errors name the flag, the dtype list, or the axis, its value and its bound. All re-exported next to `deserialize_tensor`. |
| 1 | `pipeline/activation_transfer.rs` | `deserialize_activation_with_limits` clears `allow_compressed` and `allow_quantized` whatever the caller passed; `deserialize_activation` wraps it, so the in-process path also refuses both flags. |
| 1 | `pipeline/wire_tensor.rs` | `MAX_STAGE_TOKENS_PER_FRAME = 1 << 20`; `StageWireLimits { activation_width, vocab_size }` with `hidden_states()`, `input_ids()`, `attention_mask()`, `logits(input_seq_len)`; `deserialize_wire_tensor_with_limits`. A `None` width or vocab becomes `AtMost(u64::MAX)` on that axis. `deserialize_wire_tensor` stays as the unrestricted in-process entry point. |
| 1 | `pipeline/stage_executor/`, `models/gemma4.rs`, `loading/mod.rs` | `inbound_activation_width()` on `StageExecutor` and `FamilyStageExecutor`, default `None`. `LoadedStageExecutor` stores the family override or `loading::read_hidden_size` (`text_config` first, then top level) and fails the load if both are absent. `Gemma4StageModel::inbound_activation_width` returns `hidden_size + (num_hidden_layers - layer_range.start) * hidden_size_per_layer_input` when the per-layer field is non-zero, and `unpack_stage_hidden` slices with it. |
| 1 | `pipeline/remote_service.rs` | `RemoteStageService::load` builds `PeerDecodeLimits { hidden_states, input_ids, attention_mask }` once from the executor's width and uses it at the four peer-facing decode sites (`handle_run_entry` ids and mask, `handle_activation` hidden states and mask). |
| 1 | `pipeline/runtime.rs` | `RemotePipelineRuntime` decodes the `RunResult` logits under `logits(max(input_seq_len, 1))` with its `vocab_size`, keeps `validate_logits_shape` for the exact "input length or 1" rule, and now calls `cleanup_request` when the bounded decode fails. |
| 1 | `kv_cache_transfer/streamed.rs` | `reassemble_layer_payload` takes `max_layer_bytes` and rejects `header.original_bytes > max_layer_bytes` before decompressing; its doc comment gives the receiver-side derivation `2 * n_kv_heads * head_dim * element_size * token_budget`. |
| 2 | `kv_cache_transfer/{mod,streamed,parallel}.rs` | `LayerTransferHeader.original_bytes` (no `serde(default)`; the header has no production receiver), filled by both senders. `num_elements = raw_bytes.len() / 2`. Int8 and Int4 refuse an odd-length layer. Int8 and Int4 receivers require `original_num_elements * 2 == original_bytes` before decompressing or dequantizing. |
| 3 | `kv_cache_transfer/streamed.rs` | New `reassemble_unquantized`: level None must yield exactly `original_bytes`, compressed (decompression capped at that value) or not (length compared before the copy). |
| 4 | `tensor_protocol.rs`, `tensor_serialize.rs` | `TensorHeader::shape_num_elements(&[u64])` folds from 1, so `[]` is one element; `num_elements` and `validate_data_size` share it. The 18-space message is fixed. |
| 5 | `tensor_serialize.rs`, `kv_cache_transfer/streamed.rs` | The payload is a `Cow<[u8]>`: decompress and dequantize read the borrowed slice, and only an unflagged payload is copied, once, by `into_owned`. The quantized KV path reads its uncompressed wire bytes the same way. |
| 6 | `tensor_protocol.rs` | `MAX_DECODED_TENSOR_BYTES: u64` plus `max_decoded_tensor_bytes() -> usize`, saturating to `usize::MAX`; used in `tensor_serialize.rs` and `streamed.rs`. |
| Docs | `docs/distributed.md` | The "What the secret protects" list describes the receive-side bounds. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 25 (14 source, 10 test, 1 docs) |
| Lines added / deleted | +1362 / -101 against main |
| Tests added | 21 `#[test]`/`#[tokio::test]` functions in 7 files, including the new `tensor_serialize_limits_tests.rs` and `gemma4_stage_width_tests.rs`; 3 existing tests adjusted (`test_header_num_elements_empty_shape_is_zero` renamed to `..._is_one`, `forward_rejects_logits_with_wrong_vocab_dimension`, and the `validate_logits_shape` case list) |
| Commits | 27ac0f02 (the six items), a91c64a8 (logits test expectation), aaa4daa7 (precomputed peer limits, docs, `validate_logits_shape` case), 2f137091 (merge of main) |

The merge of main (2f137091) brings in #2719 and #2720. #2720 rewrote the failure paths of `RemoteStageService::handle_activation`; git merged `remote_service.rs` cleanly, and the only conflict was in `remote_service_tests.rs`, where both PRs appended tests at the end of the file. The resolution keeps #2716's two limit tests and #2720's two error-routing tests. Section 5 covers how the two changes interact.

## 3. Technical Decisions

### 3.1 Check the header, not the payload

The limit check runs on the decoded `TensorHeader` and does no arithmetic, before the payload slice exists. That is the only point at which a refusal costs nothing: after it, the decoder may already have copied, decompressed or allocated the size the peer claimed. The tests pin this ordering by sending header-only buffers (`data_len` set, no payload bytes): a bounded decode must fail with the limit error, not with "buffer too short", which would mean the payload length was consulted first.

The mechanism is split in three layers. `tensor_serialize.rs` owns the generic vocabulary (`DimBound`, `ShapeBound`, `TensorDecodeLimits`) and knows nothing about pipelines. `wire_tensor.rs` owns the pipeline policy (`StageWireLimits` and its four builders). `remote_service.rs` and `runtime.rs` own the model facts (width, vocab, input length). A future receiver of another tensor kind (KV, weights) can reuse the first layer with its own policy.

### 3.2 Remove the amplification instead of tuning its cap

Instead of a tighter decompression cap for activations, `deserialize_activation_with_limits` refuses COMPRESSED and QUANTIZED regardless of the limits passed, and the refusal also covers the in-process path through `deserialize_activation`. `serialize_activation` has never set either flag, so no honest frame changes. With both flags gone, the largest buffer an activation decode allocates is bounded by the bytes actually received, so the 256:1 ratio disappears rather than being moved.

### 3.3 A frame-level token ceiling, not the context window

`MAX_STAGE_TOKENS_PER_FRAME` is 2^20, the same ceiling as `MAX_HANDOFF_PROMPT_TOKENS`. The context window was considered and rejected: rope-scaled checkpoints legitimately run past `max_position_embeddings`, so a context-window bound would refuse honest long prompts. The token ceiling exists to stop absurd shapes; the exact width and vocab axes do the precise work.

### 3.4 The width comes from the executor, and an unknown width fails the load

Using `config.json`'s `hidden_size` alone would have rejected every honest Gemma 4 E2B/E4B frame, because `pack_hidden_for_downstream` appends the per-layer inputs of all remaining layers to the hidden state. The trait method lets a family state its true inbound width; the default `None` means "use `hidden_size`". The PR's family audit found that every other adapter (llama, mistral, mixtral, qwen3, qwen35 including VLM, qwen3_next, gemma3, the glm4 family, glm_moe_dsa, deepseek_v3, gpt_oss, jamba, llama4, nemotron_h) sends plain `[1, L, hidden_size]`.

`read_hidden_size` reads `text_config` first because VLM and Llama 4 configs nest the language model there, and a top-level `hidden_size` may describe another tower. When neither the family nor the config yields a width, `LoadedStageExecutor` refuses to load rather than silently accepting any width: fail closed at startup, where the operator sees it, rather than at runtime on the first crafted frame.

The Gemma 4 formula lives in one `pub(crate)` method that both the limit and `unpack_stage_hidden` call, so a change to the packing layout cannot leave the receiver's bound behind. `per_layer_input_inbound_width_matches_upstream_packing` checks it against what an upstream stage actually emits.

### 3.5 Logits get a header bound and keep the exact rule

The coordinator's bound is `[1, <= input_seq_len, vocab]`, which is coarser than the real rule (the sequence axis is either the input length or 1). The header check stops the allocation; `validate_logits_shape` still enforces the exact rule after decoding. A side effect is that `validate_logits_shape` is now reachable only for a sequence length strictly between 1 and the input length, which aaa4daa7 covers with a new case. The PR also adds `cleanup_request` on a failed bounded decode, so a rejected `RunResult` does not leave the request's stage state behind.

### 3.6 Limits are built once per stage

The first implementation built `TensorDecodeLimits` per frame, which costs two or three small `Vec` allocations per token. aaa4daa7 precomputes them in `RemoteStageService::load` as `PeerDecodeLimits`. The coordinator's logits limits still depend on the request's input length and are built per forward call; next to the payload copy this is negligible.

### 3.7 Empty shape is a scalar, not an error

Rejecting `[]` was considered and dropped: the wire path already ends in `validate_raw_tensor`, which reads `[]` as a scalar, and MLX does too. Adopting the same semantics in `TensorHeader` makes the header and the final check agree, lets the encoder send a real scalar, and makes `[]` with 0 bytes fail early in `deserialize_tensor` instead of late.

### 3.8 Streamed KV gets an exact byte count

Adding `original_bytes` to `LayerTransferHeader` gives the receiver the one number it needs for level None (an exact decoded length) and a cross-check for Int8 and Int4 (`elements * 2 == bytes`). No `serde(default)` was added because the header has no production receiver; the production disaggregated handoff uses `kv_cache_serde` without compression and already anchors each slab to `ExpectedBlockGeometry`. `max_layer_bytes` is a parameter, not a constant, because only the receiver knows its expected geometry.

## 4. Verification

### 4.1 Failing-first evidence

Every new test targets an API that main lacks, so each was shown to fail by a mutation check on the branch that reverts the specific guard to 752db4f9's behavior:

| Test | Mutation | Failure |
|---|---|---|
| `limits_refuse_flags_dtype_and_shape_before_the_payload`, `hidden_states_shape_is_checked_before_the_payload`, `entry_tensor_limits_bound_ids_and_masks`, `logits_limits_bound_sequence_and_vocab` | drop `check_header` | "buffer too short" instead of the limit error |
| `activation_decode_refuses_compressed_and_quantized_frames` | stop forcing the flags off | LZ4 expansion error, no flag named |
| `run_entry_with_out_of_bounds_ids_answers_limit_error`, `activation_outside_stage_limits_never_executes` | unrestricted limits at the call sites | no limit error, or the executor ran |
| `per_layer_input_inbound_width_matches_upstream_packing` | width ignores per-layer inputs | 256 != 304 |
| `read_hidden_size_reads_text_config_then_top_level` | top level first | 1152 != 2560 |
| `reassemble_rejects_original_bytes_above_max_layer_bytes` | drop the bound | reached `decompress` |
| `reassemble_round_trips_odd_byte_layer_compressed` | per-tensor halves count | 4095 != 4096 |
| `prepare_rejects_odd_byte_layer_for_quantization` | drop the odd check | Ok |
| `reassemble_rejects_quantized_header_bytes_mismatch` | drop the bytes check | Ok, 512 bytes |
| `reassemble_rejects_uncompressed_length_mismatch` | drop the length checks | Ok, 511 bytes |
| `test_header_num_elements_empty_shape_is_one`, `float16_scalar_round_trips_and_empty_scalar_is_rejected` | `[]` is 0 elements | 0 != 1, the encoder refuses the scalar |
| `decode_does_not_copy_the_payload_before_decompressing` | `to_vec()` before decompress | assertion |
| `test_max_decoded_tensor_bytes_helper` | constant back to `usize` | E0308 |

### 4.2 Local gate

GB10 spark-102 at aaa4daa7, `--profile test-fast`, built outside the GPU lock and run under `gpu-lock` with `--test-threads=1`, real-model targets with `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`:

| Target | Result |
|---|---|
| `cargo fmt --check`, `cargo clippy --features cuda,test-utils --lib --tests -D warnings` | clean |
| `--lib distributed::` | 1512 passed, 4 ignored |
| `mlxcel-core from_bytes` | 6 passed |
| `distributed_integration`, `pipeline_e2e` | 20, 30 passed |
| `pipeline_remote_real_models --ignored` | 10 passed (llama including RDMA/Thunderbolt, gemma3, gemma4, glm4, gpt_oss, qwen3, qwen3.5; `glm_moe_dsa` skipped) |
| `pipeline_server_remote_real_models --ignored` | 7 passed (`glm_moe_dsa` skipped) |
| `pipeline_ci_multi_stage_real_models`, with and without `--ignored` | 2 + 2 passed, including the three-host parity test |
| `disaggregated_e2e`, `disaggregated_serving_e2e`, `disaggregated_router_e2e` | 51, 2, 7 passed |

The gemma4 runs use `gemma-4-e2b-it-4bit` (`hidden_size_per_layer_input = 256`), so honest packed frames pass the per-layer width bound on a real checkpoint, which is the acceptance criterion that no honest frame is rejected. `glm_moe_dsa` needs the 418.6 GB `glm5-4bit` checkpoint, which cannot load on a 128 GB box. A 32-bit build cannot be run (mlxcel-core does not build for 32-bit), so item 6 is verified by the const now evaluating and the helper test, not by a 32-bit compile. No performance measurement was taken; the per-token path changes only by one header check per decode and one fewer copy for flagged frames.

The gate above ran before the merge of main. The pre-merge gate on the merged tree (2f137091) was running when this report was written.

## 5. Remaining Gaps and Risks

- **A forward activation refused by the new limits still ends in a coordinator timeout (LOW).** After merging #2720, `handle_activation` routes execution failures and failed downstream sends through `report_activation_failure`, which sends the `Error` of the `RunEntry` exchange to the coordinator. The two bounded decodes at the top of `handle_activation` (hidden states and mask) still exit with `?`, before `mark_request_started`; the service loop logs `dropping frame` and the coordinator waits out `stage_timeout`. This is the same gap #2720's report lists for the old `deserialize_wire_tensor` failure, but #2716 adds many new ways to reach it (any shape, dtype or flag outside the bounds). The entry stage is not affected: `handle_run_entry` answers a rejected `input_ids` or mask with an immediate `Error`. `activation_outside_stage_limits_never_executes` accepts either outcome, so it will not notice when this is fixed. The fix is to match on the two decodes and call `report_activation_failure` with the error, which also finishes the request's caches from earlier steps; the test should then require the `Error` with the activation's nonce.
- **`DimBound::AtMost` admits 0 (LOW).** `[1, 0, width]` hidden states and `[1, 0]` input ids pass the limits and reach the executor. This predates the PR; a minimum of 1 on the sequence axis would close it.
- **The mask bound caps every axis at 2^20 (LOW).** An explicit mask whose key axis exceeds 1,048,576 cached tokens would be refused. No current server path sends masks over the remote pipeline.
- **`validate_data_size`'s Int4 branch uses `num_elements.div_ceil(2) as usize` (LOW).** It can truncate on a 32-bit target. It predates the PR, and the activation path refuses Int4 anyway.
- **`max_layer_bytes` is untested against a real receiver.** `reassemble_layer_payload` still has no production caller, and `LayerTransferHeader` has no compressed flag, so a future streamed receiver cannot choose the `compressed` argument from the header. Both are out of scope in #2716 and should be settled when that receiver is wired in.
- **Production handoff still passes `CacheIngestLimits::default()` (16 GiB frame).** Out of scope in #2716; no amplification, separate follow-up.
- **Breaking API changes inside the crate.** `reassemble_layer_payload` gained a parameter, `LayerTransferHeader` gained a required field, and `MAX_DECODED_TENSOR_BYTES` changed type. All callers in the tree were updated; any out-of-tree user of these public items would need the same change.

## 6. Learning Points

### 6.1 Bounds belong to the receiver

A size check derived from the sender's own header only ensures the header is self-consistent; it says nothing about whether the receiver wanted a tensor that large. The pattern this PR applies is: decode the fixed-size header, compare it with what the receiver knows from its own state (model width, vocab, input length), and only then touch the payload.

```rust
let (header, header_len) = TensorHeader::decode(buf)?;
limits.check_header(&header)?; // flags, dtype, shape; before any slice or allocation
```

### 6.2 One formula for a packed layout

When a sender packs extra features into a tensor and a receiver unpacks them, the width is a protocol fact. Computing it in one method that both the unpacker and the bound call keeps a future layout change from producing a receiver that rejects every honest frame, or one that accepts a mis-sized one.

### 6.3 Remove a feature nobody uses on a hostile path

The COMPRESSED and QUANTIZED flags were part of the generic tensor format, but no activation sender used them. Refusing them on that path removed the whole amplification class with a two-flag check, which is simpler and stronger than any cap on the decompressed size.

### 6.4 `Cow` for conditional ownership

The decoder needs owned bytes only when it transforms them. Starting from `Cow::Borrowed` and replacing it with `Cow::Owned` after decompress or dequantize, then calling `into_owned()` at the end, gives one copy for an unflagged payload and none before a transformation, without two code paths.

### 6.5 Size constants that exceed 32 bits are `u64`

A `usize` constant of `1 << 32` is a compile error on a 32-bit target. Declaring the constant as `u64` and converting at the use site with a saturating helper keeps the value exact on 64-bit and the build valid elsewhere.

## 7. References

- Issue #2716: tighten tensor decode limits to the model shape
- PR #2701 (issue #2691): bound tensor wire decode sizes; section 7 of its report is the source of #2716
- PR #2720 (issue #2712): routes non-entry stage failures to the coordinator through `report_activation_failure`
- PR #2719 (issue #2715): router parity residuals, merged into this branch at 2f137091
- PR #2708 (issue #2694): final stage sends `RunResult` to the coordinator with `reply_nonce`
- #2650 (cluster secret), PRs #2674 and #2684 (authenticated handshake, per-frame AEAD): why every item here requires an authenticated member
- #2654, #2671: related distributed transport hardening
