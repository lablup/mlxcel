# PR #2671: Keep remote pipeline stages on the dense kernel path

**Date**: 2026-10-27
**Status**: Completed (report written on the PR branch before merge)
**Languages**: Rust, YAML, Shell
**Risk Level**: Medium (changes the numeric path of every remote pipeline stage boundary for float16 activations on CUDA and of every Gemma 3 and Gemma 4 prefill stage; dense single-process paths are unchanged; includes a HIGH fix for a heap over-read reachable from an authenticated peer)

## Executive Summary

A two-stage remote pipeline gave different logits from the single-process model for llama, gemma3 and gemma4 on GB10 (issue #2654). Comparing hidden states at the split boundary before changing anything showed two independent defects. First, the wire decoder rebuilt float16 activations with `from_bytes_f16`, a checkpoint loader that widens float16 to float32 on every backend except Apple, so the downstream stage and the LM head ran in float32 while the dense model stayed in float16. Second, `Gemma3StageModel` and `Gemma4StageModel` built explicit causal and sliding masks for every prefill, while the dense forwards pass no mask inside the window and take MLX's maskless causal SDPA, a different kernel with a different reduction order. The PR rebuilds wire tensors in the dtype they were sent in, moves the Gemma prefill mask decision into helpers shared by the dense model, the stage model and `Gemma3Backbone`, and routes wire decoding through the length-checked `kv_cache_serde::reconstruct_mlx_array`, which closes a pre-existing heap over-read on a short payload. All three families now produce byte-identical logits to dense, in-process and over the wire, with no tolerance changed.

## 1. Problem Statement

### 1.1 Background

Issue #2654 recorded four `#[ignore]` real-model tests that failed deterministically on GB10 (spark-102, CUDA):

- `tests/pipeline_remote_real_models.rs`: the llama, gemma3 and gemma4 `*_real_model_parity_and_cleanup` tests, failing `allclose(&full_prefill, &remote_prefill, 1e-4, 1e-4)` on prompt `[2, 3]` (llama `[128000, 9906]`), split at `num_layers / 2` (gemma4 at 13).
- `tests/pipeline_server_remote_real_models.rs`: `pipeline_server_remote_coordinator_gemma4_matches_dense_baseline`, failing `assert_eq!(dense.text, remote.text)`.

The qwen3 and qwen3.5 variants passed in the same runs. The failure was identical on main before and after #2650 (the cluster handshake) and unchanged with `MLXCEL_SDPA_DETERMINISTIC=1`, so it was neither the transport change nor run-to-run noise. CI never ran these tests.

### 1.2 What the issue got wrong, and why it matters

The issue reasoned from the code that "the wire keeps the source dtype and raw bytes, with no implicit cast", and, because llama failed while qwen3 passed through the same generic stage loop, pointed at llama-specific stage construction. Both readings were wrong in an instructive way. The bytes on the wire were preserved, but the function that turned them back into an array changed the dtype. qwen3 passed only because its activations are bfloat16, which `from_bytes_f16` keeps native on CUDA (unless `MLX_BF16_NATIVE=0`). The llama stage construction was never at fault.

### 1.3 Root cause evidence

The diagnosis compared the dense hidden state after layer `split - 1` with stage 0's output, then compared in-process stage logits (no wire) with dense logits, then added the wire round trip. The numbers below are from the PR body, measured on GB10.

| Model (CUDA activation dtype) | Stage 0 output vs dense at split | In-process stage logits vs dense | Over the wire vs dense | Defect |
|---|---|---|---|---|
| llama-3.2-1b-4bit (float16) | byte-identical | byte-identical | max abs diff 1.5e-2, dtype float32 | wire widening |
| gemma-3-1b (float16) | not reported | max abs diff 9.4e-2 | not reported | masks and wire widening |
| gemma-4-e2b, split 13 (bfloat16) | not reported | max abs diff 3.4e-1 | not reported | masks |
| qwen3 (bfloat16) | not reported | not reported | passes | none |

### 1.4 Risk of leaving it

| Risk | Impact | Likelihood |
|------|--------|------------|
| Remote pipeline serving of float16 checkpoints on CUDA silently runs the downstream stages in float32 (more memory traffic, output not equal to dense) | Medium | High for any float16 checkpoint |
| Gemma 3 and Gemma 4 pipeline output diverges from dense output, visible as different generated text (the gemma4 server test) | High | High |
| A peer that passed the cluster handshake sends a short payload with a large shape and reads past a heap buffer in the receiving stage | High | Low (requires an authenticated peer) |

## 2. Technical Decisions

### 2.1 Rebuild wire tensors in the sent dtype, at the reconstruction site

Before, `deserialize_wire_tensor` special-cased the half types:

```rust
match tensor.dtype {
    TensorDtype::Float16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, false)),
    TensorDtype::BFloat16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, true)),
    other => {
        let dtype = tensor_dtype_to_mlx(other)?;
        Ok(mlxcel_core::from_bytes(&tensor.data, &shape, dtype))
    }
}
```

After, every dtype goes through the same path:

```rust
let dtype = tensor_dtype_to_mlx(tensor.dtype)?;
let raw = RawTensorData { data: tensor.data, shape, dtype };
reconstruct_mlx_array(&raw)
```

`from_bytes` with MLX dtype ids 9 (float16) and 12 (bfloat16) reinterprets the 2-byte halves on every backend, so the array that leaves stage N is the array that enters stage N+1. The implementation review checked the edge cases: Metal float16 and native bfloat16 were already lossless, and under `MLX_BF16_NATIVE=0` the weights themselves load as float32, so activations remain consistent with the dense model there too.

| Option | Pros | Cons |
|---|---|---|
| Relax the parity tolerance (the issue allowed this with evidence) | No code change | The drift was not inherent: it came from a dtype change and a kernel choice, both fixable. A looser tolerance would also hide the next real divergence |
| Keep `from_bytes_f16` and cast back to the sent dtype | Small diff | A round trip through float32 for nothing, and it keeps a checkpoint loader on the activation path where its backend-dependent behavior does not belong |
| **Chosen: one dtype-preserving path for all dtypes** | Byte-identical boundary on every backend; one code path to reason about; picks up the existing length validation | Relies on `from_bytes` covering every wire dtype; the review noted it decodes int16 through the `uint8_t` default branch (activations never use int16) |

### 2.2 Share the dense prefill mask decision instead of reimplementing it

The stage models were written as separate forward passes and drifted from the dense forwards as the dense code gained the maskless causal path, the native sliding-window route (#2621) and the long-prefill cutoff (#672). The table shows what each path built for a text-only prefill:

| Prefill case | Dense model (unchanged) | Stage before | Stage after |
|---|---|---|---|
| Global (full-attention) layer | `None`, maskless causal SDPA | explicit `create_causal_mask(seq_len, global_live_len)` | `None` |
| Sliding layer, `live_len + l <= window` | `None` | explicit band (both families) | `None` |
| Sliding layer past the window, native route planned | `None` (#2621) | Gemma 3 `None`; Gemma 4 explicit band | `None` |
| Sliding layer past the window, native route not planned | dense band | dense band | dense band |
| Gemma 4, `l > 4096` | both `None` (#672) | both explicit | both `None` |

An explicit bottom-right causal mask and the maskless causal flag describe the same attention pattern, but they select different SDPA kernels on CUDA, and the two do not agree bit for bit. The parity tests use a 1e-4 tolerance, which this kernel difference exceeded on the real checkpoints.

The fix makes the dense decision the only decision:

- `gemma3::text_prefill_sliding_mask(sliding_attn, h, caches, seq_len, sliding_live_len, window)` returns the sliding mask or `None`. Global layers never take a mask on this path. It is called by `Gemma3Model::forward_with_caches_and_embeddings`, `Gemma3StageModel::execute_hidden` and `Gemma3Backbone::forward_embeds`. `sliding_prefill_plans_native` becomes private to it, and the stage's `first_global_cache_index` is removed.
- `gemma4::text_prefill_masks(sliding_attn, h, caches, l, window)` returns the `(global, sliding)` pair, including the `l > DENSE_PREFILL_MASK_MAX_TOKENS` early return that used to be a separate branch in the dense forward. `Gemma4TextModel::sliding_prefill_plans_native` becomes a free function taking `Option<&Attention>` so the stage model, which holds `(global_idx, layer)` pairs, can pass its own first sliding layer.

The review confirmed the dense branches are boolean-equivalent to main: the Gemma 4 long-prefill branch is still reached only for `l > 4096` with no padding or vision overlay, and the external-mask, padded, batched-decode and verify branches are untouched.

Trade-off: the stage model now inherits the dense text-only preconditions (no caller mask, no resident padding, no vision overlay, no per-row valid ends). The stage did not handle those cases before either; it built plain causal masks regardless, so this is not a regression, but a future multimodal or padded stage path must not reuse these helpers unchanged.

`Gemma3StageModel` also gains `pipeline_hint(&hidden, layer_idx, n)` after each layer, matching the dense loop. It does nothing unless `MLXCEL_PIPELINE_GRANULARITY` is set.

### 2.3 Validate the wire payload length before the FFI copy (HIGH)

`mlxcel_core::from_bytes` copies `product(shape) * element_size` bytes out of a raw pointer and never looks at the payload length. The wire header's shape and the payload both come from the peer, so a short payload with a large shape read past the heap buffer. This predates the PR (the old `from_bytes_f16` branch did not check either), but the PR sends every dtype through `from_bytes`, so the security review fixed it in the same PR rather than widening the exposure.

The fix reuses `kv_cache_serde::reconstruct_mlx_array`, whose `validate_raw_tensor` (in `kv_cache_serde/types.rs`, added for the paged KV handoff in #186 and unchanged here) rejects negative dimensions, computes the element count and the byte count with checked multiplication (an empty shape is a 1-element scalar), and requires the payload length to match exactly. `deserialize_wire_tensor` already converted each `u64` dimension with `i32::try_from`. The dtype table it uses covers every dtype `tensor_dtype_to_mlx` produces, so no previously accepted wire dtype is newly rejected. The exposure was limited to peers that completed the cluster handshake (#2650, #2674).

### 2.4 GB10 CI coverage decision

The issue asked for a recorded decision on running remote parity in CI. The real-model tests are not added: the PP workflow's two-host job runs on `ubuntu-latest` (CPU) with only the qwen3 fixture, and no GB10 job runs the PP suites or holds llama or gemma checkpoints. Instead:

- `scripts/ci/run-pp-two-host-logical.sh` adds `models::gemma3::gemma3_stage_tests::` and `models::gemma4::gemma4_stage_tests::` to its lib-test filter. Its existing `distributed::pipeline::` filter already picks up `wire_tensor::tests`.
- `.github/workflows/pipeline-parallel-ci.yml` adds `src/models/gemma3.rs`, `gemma3_backbone.rs`, `gemma3_stage_tests.rs`, `gemma4.rs` and `gemma4_stage_tests.rs` to its `pull_request` path filter. The implementation review pointed out that main CI never runs mlxcel lib tests on Linux, so without this a dense-mask change that touches no PP path would never run the new parity suites.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 9 |
| Lines added | +1,035 |
| Lines deleted | -195 |
| Tests added | 14 (8 wire, 3 Gemma 3 stage, 3 Gemma 4 stage) |

- **Wire**: `src/distributed/pipeline/wire_tensor.rs` (dtype-preserving, length-checked decode) and the new `wire_tensor_tests.rs`.
- **Gemma 3**: `src/models/gemma3.rs` (`text_prefill_sliding_mask`, dense and stage callers, `pipeline_hint` in the stage loops), `src/models/gemma3_backbone.rs` (uses the helper), new `gemma3_stage_tests.rs`.
- **Gemma 4**: `src/models/gemma4.rs` (`first_sliding_attention`, free `sliding_prefill_plans_native`, `text_prefill_masks`, dense and stage callers), new `gemma4_stage_tests.rs`.
- **CI**: the PP workflow path filter and the two-host logical script.

New tests:

- `wire_tensor::tests`: `float16_activation_round_trips_in_float16`, `bfloat16_activation_round_trips_in_bfloat16`, `float32_activation_round_trips_in_float32`, `token_ids_round_trip_in_int32`, `hand_built_bfloat16_buffer_round_trips`, `float16_short_payload_is_rejected`, `float32_short_payload_is_rejected`, `scalar_shape_with_empty_payload_is_rejected`. The round-trip fixture uses values not exactly representable in half precision, so any widening or narrowing cast would change the bytes.
- `gemma3_stage_tests` and `gemma4_stage_tests`, each with `two_stage_split_matches_dense_prefill_and_decode_bitwise` (2- and 8-token prompts, then decode), `two_stage_split_matches_dense_over_window_prefill_bitwise` (`WINDOW + 8` tokens) and `two_stage_split_matches_dense_continuation_prefill_bitwise` (a second chunk on a non-empty cache, 4+3 inside the window and 12+8 across it, then decode). The synthetic model has 4 layers, window 16, head dim 256 (a size the fused SDPA kernels serve, so the masked and maskless paths are distinct kernels), and splits so stage 1 opens on a global layer. Gemma 3 runs in float16 and Gemma 4 in bfloat16, the dtypes the real checkpoints use on CUDA. Every comparison is bitwise, not `allclose`.

| Hash (rebased) | Original | Message |
|---|---|---|
| `6336cd04` | `2513d03f` | fix(distributed): keep remote pipeline stages on the dense kernel path |
| `4a59dd91` | `25e93ded` | fix(distributed): validate wire tensor length before from_bytes |
| `06cd3d3f` | `1205a313` | test: cover continuation prefill and run Gemma stage suites in PP CI |

The third commit addresses the implementation review's two MEDIUM items (no continuation prefill case, Gemma files missing from the PP path filter) and its LOW item (the Gemma 3 test header called the float16 fixture bf16).

## 4. Validation

All results come from the PR body and review comments, recorded on the pre-rebase heads on GB10 (spark-102). This report did not re-run any of them.

- **Failing first.** The float16 wire round trip and both 2-token split tests fail against main's sources (logit diffs 8.8e-3 and 2.3e-2, and a float32 dtype after the wire). The PR does not report the remaining new tests against main.
- **Real models.** With `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1 --test-threads=1` at the final head: `pipeline_remote_real_models` 6/6 (llama, gemma3, gemma4, qwen3, qwen3.5 parity, llama drain), `pipeline_server_remote_real_models` 5/5, and `pipeline_multi_stage_two_host_logical_smoke`. `pipeline_stage_executor_real_models` 10/10 at the first commit; its gemma variants and llama worker loop also failed on main, so the PR fixes more than the four tests the issue listed.
- **Lib suites and lint.** `distributed::pipeline::`, `models::gemma3`, `models::gemma4`, `models::gemma3_backbone` and `kv_cache_serde` pass, with and without `MLXCEL_SDPA_DETERMINISTIC=1`; after the security fix, `wire_tensor::` (8) and `kv_cache_serde::` (41) pass. Clippy with `-D warnings` and fmt are clean.
- **Rebase.** The orchestrator rebased onto main `359a335f`; `git range-diff` shows all three commits patch-identical. The upstream range includes #2674, which bumped the cluster handshake to wire version 2 and changed `tcp_transport.rs`, the transport the real-model remote tests run over. The GB10 real-model runs above predate that rebase; a pre-merge check on the rebased head was running when this report was written.
- **Not verified.** The new lib tests had not yet run on a CPU build; the PP CI run on this PR is the first. Metal was not checked; on Apple the float16 wire path was already native, so only the mask change applies there. No throughput or memory was measured. Other stage families (gpt-oss, glm4, glm5) were not run because their checkpoints are not on the host.

## 5. Residual Risks and Follow-ups

- **No GB10 CI for real-model parity.** The weight-free split tests guard the two mechanisms found here, but a new divergence specific to a real checkpoint (quantization layout, a different head dim) would still only show up in a manual GB10 run.
- **Pre-existing wire hardening gaps (authenticated peers only).** From the security review, outside this diff: (MEDIUM) `tensor_serialize::deserialize_tensor` sizes the `dequantize_int8/int4` output from `header.num_elements()`, which saturates to `u64::MAX` on overflow and is then multiplied by 2 unchecked, so a crafted header panics or aborts on allocation; (MEDIUM) `tensor_compress::decompress` trusts the LZ4 size prefix (up to 4 GiB) with no cap tied to the shape; (LOW) `from_bytes` decodes int16 through the `uint8_t` default branch. No open issue tracked these when this report was written.
- **Trimmed caches in a stage.** The removed stage code sized the global prefill mask from `live_len` after a `--max-kv-size` trim (#430). Global stage layers now take the maskless path, exactly like the dense model, so the stage is correct whenever the dense model is, but no stage test exercises a trimmed cache.
- **Gemma 4 stage with no sliding layer.** `text_prefill_masks` builds the sliding band whenever `window < l <= 4096` and the native route is not planned. For a stage that owns only full-attention layers (one-layer stages in practice), that band is allocated and never used. The cost is one mask per prefill; outputs are unaffected.
- **Helpers assume text-only prefill.** See 2.2: a future stage path with padding, a vision overlay or per-row valid ends needs its own mask branch, as the dense forward has.

## 6. Learning Points

- **Bisect a parity failure by boundary before forming a theory.** Three byte comparisons (stage 0 output vs dense hidden state, in-process stage vs dense, wire vs in-process) separated two unrelated defects that the issue's code reading had attributed to a third place. Equality checks at each seam are cheap and leave no room for interpretation.
- **"The bytes are preserved" is not "the tensor is preserved".** A serializer can be lossless while the deserializer's constructor applies backend policy. Loader helpers such as `from_bytes_f16` encode decisions that suit checkpoints (widen float16 where the backend prefers float32) and are wrong for activations, which must round-trip exactly.
- **Equivalent masks are not equivalent kernels.** An explicit causal mask and the causal flag mean the same thing mathematically, but they select different SDPA implementations with different reduction orders. When two code paths must agree bit for bit, they must make the same kernel-selection decisions, which is easiest when they call the same decision function.
- **Duplicated forward passes drift.** The stage models reimplemented the dense forward, then missed three later dense-path changes (#2621, #672, the maskless causal path). Extracting the decision into a helper used by every caller is what keeps the next such change from reopening the gap.
- **Test parity with exact equality on a synthetic model.** A small random model with the real dtype and a kernel-relevant head dim reproduces the kernel split without downloading weights, so it can run in CPU CI and fail loudly; `allclose` would have hidden exactly the class of bug fixed here.
- **Validate untrusted sizes before crossing the FFI.** A C++ copy that trusts a shape is a memory-safety boundary even in a Rust codebase. Reusing an existing validator (`validate_raw_tensor`) kept the fix to one call site.
