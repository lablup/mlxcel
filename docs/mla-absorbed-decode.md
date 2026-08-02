# MLA matrix-absorbed decode

Multi-head latent attention (MLA) is the DeepSeek-family attention that compresses K and V into a single latent vector per token. This page covers mlxcel's matrix-absorbed decode path, which serves attention directly against that latent instead of up-projecting it first.

Status: opt-in, off by default. Verified on synthetic tensors with MLA geometry. Not yet verified against a real DeepSeek checkpoint.

## The identity

MLA stores one latent `c` of width `kv_lora_rank` (512 in every shipping checkpoint) plus a shared rope stream `k_pe` of width `qk_rope_head_dim` (64). `kv_b_proj` up-projects `c` into per-head keys and values. Naming its two row blocks `W_UK` and `W_UV`:

```
score:  q_nope . (W_UK c)  ==  (W_UK^T q_nope) . c
output: attn @ (W_UV c)    ==  (attn @ c) @ W_UV
```

Folding `W_UK` into the query projection and `W_UV` after the attention removes the up-projection from the decode graph without changing the arithmetic. Attention then runs against `c` with `num_heads` query heads sharing one latent KV head, and the score becomes the sum of two dot products, one over `kv_lora_rank` and one over `qk_rope_head_dim`.

## What it buys

The KV cache stops scaling with head count.

| Geometry | Decompressed bytes/token/layer (f16) | Latent bytes/token/layer (f16) | Ratio |
|---|---|---|---|
| DeepSeek-V2-Lite (16 heads) | 10240 | 1152 | 8.9x |
| DeepSeek-V3 (128 heads) | 81920 | 1152 | 71.1x |

These come from `mlxcel_core::mla::decompressed_bytes_per_token` and `latent_bytes_per_token`, which are the same functions the benchmark harness prints, so the table cannot drift from the code.

The cost is fixed weight memory. The absorbed contractions are dense batched matmuls and MLX has no batched quantized matmul that keeps a 3-D per-head quantized operand, so the fold dequantizes `kv_b_proj` at load time and holds it dense. For DeepSeek-V2-Lite that is about 113 MiB in f16 across 27 layers against about 28 MiB for the 4-bit original. The trade only pays off past the context length where the cache saving exceeds it; the harness prints both numbers so the crossover is visible.

## Flags

| Variable | Effect |
|---|---|
| `MLXCEL_MLA_ABSORBED=1` | Cache `(ckv, kpe)` and run absorbed decode (Stage 1). Default off. |
| `MLXCEL_MLA_SPLIT_KV=1` | Additionally cut the latent range into chunks and merge the partial states (Stage 2). Implies the above; ignored without it. |

Both accept `1`, `true`, `on`, `yes` in any case. Anything else, including an unset variable, is off, so a typo degrades to the decompressed path rather than silently enabling one nobody asked for.

With `MLXCEL_MLA_ABSORBED` unset, no fold is built, no weight is dequantized, and the family runs the pre-existing decompressed path unchanged.

## Where it applies

| Family | Before this work | Now |
|---|---|---|
| `deepseek_v2` | Decompressed cache | Absorbed under the flag |
| `deepseek_v3` | Already absorbed (`embed_q` / `unembed_out` decomposition in `sanitize_weights`) | Unchanged |
| `deepseek_v32` | Already absorbed, same decomposition | Unchanged |
| `minicpm3` | Decompressed cache | Unchanged, still decompressed |
| `dots1` | Decompressed cache | Unchanged, still decompressed |

V3 and V3.2 have carried their own copy of the absorption transform since before this work. The shared `mlxcel_core::mla` module is where they can converge, and where MiniCPM3 and dots1 can be folded next.

## Cache layout and what stays working

The latent cache is a `KVCache` in FP16 mode whose two slots hold asymmetric tensors:

| Slot | Holds | Shape |
|---|---|---|
| `keys` | `ckv` | `[B, 1, L, kv_lora_rank]` |
| `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]` |

The FP16 update path already reads each side's head dimension from its own shape, so this needs no new cache type. That matters: a genuinely new type would force the family into the `ModelOwnedSequenceState` escape hatch the SSM hybrids use, which costs batching, the paged decode backend, and the generic prompt-cache donation path. Instead `offset`, `live_start`, `trim`, `trim_front`, `nbytes`, and `can_trim_prompt_cache` all keep their existing meanings on latent rows.

The slot assignment matches what `src/models/deepseek_v3.rs` has always done, so V2 folded onto the shared type and V3's existing code describe the same buffer.

Declined, with a message, falling back to the decompressed path:

- Any non-FP16 KV cache mode (`int8`, `fp16+turbo4`, `turbo4`, `fp16+turbo3`, `turbo4-delegated`). Their per-token quantization is calibrated for a per-head K/V row, not for a 512-wide latent whose reconstruction error is amplified by every query head that reads it.
- Paged backing. The pool allocates `[num_blocks, page_size, Hkv, head_dim]` with one head dimension for both sides, which the asymmetric `(512, 64)` split cannot fill.

The decline is decided per cache, and the answer is fixed for a cache's lifetime, so a run that declines on the first step declines on every step and a cache never mixes the two layouts.

## Prefill

Prefill stays on the decompressed path and up-projects the live latent window through the still-loaded `kv_b_proj`. Absorption is a decode transform: it trades one up-projection of the whole window for a fold applied to every query, which only wins when the cached window is large relative to the number of queries.

Keeping the original `kv_b_proj` for this means the absorbed prefill is the same arithmetic on the same quantized weight the decompressed path uses. The cost is that a chunked prefill re-up-projects the rows it already handled on the previous chunk, because the cache no longer holds them decompressed.

Speculative and MTP multi-token verify steps take this same path, so they keep working against the latent cache rather than falling back.

## Stage 2: split-KV

Absorbed decode is one query row per head against the whole cached window, so parallelism is bounded by `batch * num_heads` and adding context adds none. Stage 2 cuts the latent range into chunks whose partial softmax states are combined afterwards, the same observation issue #898 made about paged decode.

The merge is issue #898's `paged_attention_merge_states`, reused with no change to the kernel, the FFI signature, or the C++ launcher. Its contract is deliberately paging-agnostic: `v_in [N, H, D]` partials already divided by their own denominator, `lse_in [N, H]` in **log2** units, and `o_indptr [M + 1]` grouping. MLA fits it as `H = num_heads`, `D = kv_lora_rank`, `M = batch`, `N = batch * chunks`.

The log2 requirement is the one clause that fails silently. A natural-log LSE still merges and returns a plausible wrong weighted average, so the conversion is a named constant (`mla::split_kv::LOG2_E`) and `merge_rejects_natural_log_lse_units` is a negative control that asserts the natural-log variant does *not* match the reference.

The partial producer is currently composed from MLX ops. It is correct and it is the reference a fused partial kernel would be validated against, but it runs `C` small matmuls where Stage 1 runs one large one, so it is expected to be slower than Stage 1. Do not quote a Stage 2 throughput number before a fused partial kernel exists.

## Benchmark

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer
cargo run --release --features metal,accelerate --example mla_absorbed_decode_bench -- \
    --contexts 4096,16384,32768 --batches 1,4 --steps 64 --warmup 16
```

It measures one MLA attention block per decode step over three arms and prints the KV-memory table. Synthetic tensors, not a real checkpoint, so the numbers are the throughput of the layer that changed, not end-to-end model throughput.

Every row ends in a `paths=` field from `mlxcel_core::mla::stats`, taken and reset around the measured region, and the harness prints a `WARNING` on any row whose counters disagree with the arm's label. **Do not report a row whose `paths=` field does not name its own arm.** Issue #899 shipped a fused decode path that never activated and whose before/after benchmark compared the fallback against itself; that null looked clean and was nearly accepted.

For a real checkpoint, `MLXCEL_MLA_ABSORBED=1 mlxcel generate -m models/<deepseek> ...` prints one line at load stating how many layers folded and the bytes/token before and after. A run whose line reads `0/27 layers` is running the fallback. The line goes to stdout rather than through `tracing` because the `mlxcel` CLI installs no tracing subscriber, so a `tracing::info!` on this path emits nothing at any `RUST_LOG`.

## Related

- `src/lib/mlxcel-core/src/mla/` for the implementation.
- `src/lib/mlxcel-core/src/paged_v2/` for the merge kernel and its contract.
- [`docs/turbo-kv-cache.md`](turbo-kv-cache.md) for the KV cache modes this path declines.
