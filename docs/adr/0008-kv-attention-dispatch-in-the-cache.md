# ADR 0008: Attention dispatch is a property of the KV cache

**Status:** Accepted (2026-10-07). Epic #2166 Phase 4a (issue #2171). Amends [ADR 0007](0007-unified-batch-native-engine.md): it fills in the "KV storage" row of the component table (one KV interface over dense and paged storage) and records the kernel-selection rule and the single-sequence storage default that row left to this phase. ADR 0007's decision table stands unchanged.

## Context

The CLI and the server ran different attention kernels for the same model and the same request, and the choice was made in each model forward by the call site. `src/models/qwen3.rs` and `src/models/llama3.rs` each carried `l == 1 && mask.is_none() && cache.is_paged_backed()` in the single-sequence forward and a batched twin that read a scheduler-supplied `DecodeBatchContext` (`is_paged_decode()`, `paged_block_size`, `use_native_paged_kernel`). Two families could disagree on when the paged kernel applies, a storage-policy change meant editing model files, and a numerics fix in one kernel (#2128, #2162 in cuDNN SDPA) could reach one front end and not the other. ADR 0007's "same input, same output" invariant cannot hold across B=1 dense and B=1 paged until the storage behind a cache, not the model, picks the kernel.

Measured inputs (ADR 0007, GB10, 2026-10-07): a lone sequence on the paged backend decodes 6.5 to 23 percent slower than on dense storage in every cell (Qwen3-1.7B and Llama-3.2-1B, 256 and 8192-token prompts), while paged TTFT at 8192 tokens was 8 percent lower on Llama and unchanged on Qwen3.

## Decision

### One entry, storage decides

`KVCache::attend(&mut self, q, k, v, scale, mask)` appends the step's K/V and runs attention through the kernel that suits the storage behind the cache; `cache::attend_batched(q, k, v, &mut [&mut KVCache], scale, mask)` is the batched counterpart over one layer's caches. Both live in `src/lib/mlxcel-core/src/cache/attend.rs`. The models hand over projected, rotated Q/K/V and read nothing back to choose a kernel.

The rule:

| storage behind the cache | single-token, unmasked step | any other step |
|---|---|---|
| pool-backed paged (`KVCache::new_paged`, what `CachePool` wires for a sequence on the paged backend) | `paged_batch_decode_attention`: the fused v2 kernel above the token floors, else the ADR 0001 gather-then-SDPA fallback, both chosen inside that entry | pool append, gathered visible window, fused SDPA |
| dense Turbo4Asym / Turbo4 / Turbo4Delegated | the dequant-first SDPA variants the cache mode and its env gates already select (ADR 0002) | dense update, full dequant, SDPA |
| dense FP16 / Int8 | dense update, fused SDPA | dense update, causal SDPA (no mask) or masked SDPA |

A masked step (prefill, speculative or MTP verify) never takes the paged single-token kernel. A family with `supports_paged_decode_backend() == false` never gets a pool-backed cache, so it always runs dense. In a batch, a single-token unmasked step over pool-backed caches is one whole-batch launch; any other shape, and any batch the pooled entry declines before writing, runs `attend` per row, so a row decodes through the same route whether it was scheduled alone or in a batch.

Dispatch is a match on the cache's own fields. A `trait KvStorage` with a virtual call per layer per token was rejected, as ADR 0004 rejected per-op dynamic dispatch on the hot loop; the cache already resolves its quantization modes the same way.

### The storage policy is set where the cache is built

`CachePool::allocate_with_layout` decides storage once per sequence: pool-backed caches for a dense-natural family on the paged backend with an FP16 layout, dense caches otherwise. That allocation is the per-sequence storage policy. For the families on `attend`, `DecodeBatchContext` no longer selects a kernel; the scheduler still builds it for the model-owned families (Gemma 3, Llama 4, and the `model_owned` dispatch helpers), whose per-sequence state is not a `KVCache` and which keep their dense-pointer paged kernels until they move onto the entry.

The dense-pointer paged route (`paged_decode_attention_dense_compat` over dense `keys`/`values` with a block table derived from the batch metadata) is removed from the Qwen3 and Llama 3 batched forwards. With an FP16 paged layout the pool wires pool-backed caches, which that route already skipped, so it only ever served the FP16 layers of a non-FP16 paged layout, whose caches are dense: the Boundary-V layers a Turbo mode keeps at FP16 (`MLXCEL_KV_BOUNDARY_V_LAYERS`, two at each end by default) and the last layer `--kv-skip-last-layer` (default on) keeps at FP16 under `--kv-bits`. Those layers now run `attend` per row (dense update, then SDPA through `attention_from_ptr`), the route the quantized layers of the same batch already took. The removed C++ entry was itself a per-row loop of block slices, a concat and a direct `fast::scaled_dot_product_attention` call, so this is the same attention without the concat copy, but not bit-identical to it. The kernels stay for the model-owned families.

### Single-sequence storage default

Dense is the single-sequence KV storage, as ADR 0007 measured. It applies wherever the engine has one sequence: the CLI, `--max-batch-size 1`, and the Phase 4b/5 engine when a request runs alone. The server's storage policy is unchanged in this phase: `--decode-storage-backend` with `auto` resolving to paged for batched serving (`max_batch_size > 1`, a batching family that supports the paged backend), so a lone request on a batched server still decodes on the paged pool. Migrating a lone sequence between storages as the batch grows and shrinks is not in scope; it would need a detach-and-adopt across storages per transition, and ADR 0007's numbers do not yet say whether the win at B=1 pays for that copy.

## Consequences

- `rg 'is_paged_backed\(\)|is_paged_decode\(\)' src/models` returns nothing for Qwen3, Llama 3 and the families that reuse their attention (Qwen2, Qwen2.5, Helium, the VLM text backbones). The paged-vs-dense decision has one home, so the ADR 0007 parity harness compares one dispatch under two storages.
- Batched decode over Turbo caches now takes the same dequant-first variants as single-sequence decode instead of a full dequant per step; the result is the same exact attention through a different op order, so it is not bit-identical to the previous batched route. Pool-backed routes and FP16 dense routes on the dense backend are op-for-op what the models ran before, pinned by `cache::attend::attend_tests` (which also pins a batched Turbo4Asym row to the single-sequence dequant-first call); the FP16 boundary layers of a quantized paged batch changed kernel as described above.
- MLA latent caches keep their own attention and expose the same entry: `MlaLatentCache::attend` appends the step's latent and rope rows and runs the absorbed step, Stage 2 split-KV first when enabled; DeepSeek V2's absorbed decode calls it (`mla::decode_tests` pins it), while its prefill keeps up-projecting through the model's `kv_b_proj`.
- `RotatingKVCache` (Gemma 3's sliding window) is untouched; its decode undo log (#2182) and the model-owned families are the follow-up this phase leaves for 4b. Closed by [ADR 0009](0009-engine-step-api.md) (#2172): `RotatingKVCache::attend` and `ChunkedKVCache::attend` keep `update_and_fetch` as the write path, the model-owned enums implement `KvAttention`, and `DecodeBatchContext` is removed.
- Prompt-cache detach and adopt, the disaggregated handoff (`is_paged_backed()` in `src/distributed/disaggregated/handoff_impl.rs`) and `CachePool::sync_paged_state_with_dense` are unchanged: they read storage, which is still the cache's property.

## References

- Epic #2166, issue #2171, and the Phase 0 baseline in `docs/benchmark_results/unified-engine-baseline-gb10-2026-10-07.md`.
- [ADR 0001](0001-paged-attention-gather-vs-fused-kernel.md) (gather fallback), [ADR 0002](0002-turbo-kv-split-dequant-vs-fused.md) (Turbo dequant-first), [ADR 0004](0004-compute-backend-session-seam-and-stablehlo-family.md) (no per-op dynamic dispatch), [ADR 0007](0007-unified-batch-native-engine.md) (the engine this phase serves).
- `src/lib/mlxcel-core/src/cache/attend.rs`, `src/lib/mlxcel-core/src/cache/paged_batch_decode.rs` (#899), `src/lib/mlxcel-core/src/cache.rs` (`CachePool::allocate_with_layout`).
