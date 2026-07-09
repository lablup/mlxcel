# Quantized KV cache modes on CUDA (GB10) status matrix

Date: 2026-07-10
Hardware: NVIDIA GB10 (sm_121, unified memory), CUDA build
Model: `llama-3.1-8b-4bit` (dense Llama, GQA 8 KV heads, head_dim 128, 32 layers)
Issue: #635 (epic #623)

## Summary

All six `KVCacheMode` variants run on CUDA without crashing or aborting. There is no `[metal_kernel] No Metal back-end` abort path in any mode: every custom Metal kernel dispatch in the Turbo and Sparse-V paths is gated behind `kernel_enabled()`, which is compile-time `false` on non-macOS targets, so each Turbo mode transparently falls through to a plain-MLX graph path on CUDA.

The one real defect was in the single-stream Int8 path: `quantize_per_token` floored every per-token scale to a minimum of `1.0` (`maximum(scale, 1.0)`). Because per-token KV absmax is almost always well below 127, the true scale (`absmax / 127`) is far below 1.0, so the floor collapsed Int8 quantization into round-to-nearest-integer and produced degenerate greedy decodes. The fix substitutes `1.0` only where the scale is exactly zero (all-zero token) and otherwise keeps the true scale. After the fix, Int8 greedy output is byte-identical to Fp16 on the validation prompt. The batched server 4-bit path (`cache/batch_quant.rs`) was never affected because it uses MLX-native affine `quantize`; the 8-bit uniform batched configuration (`--kv-bits 8 --kv-quant-scheme uniform`, which maps `BatchKvQuantConfig::base_mode()` to `KVCacheMode::Int8`) reuses `quantize_per_token` and is also fixed by this change.

## Audit status matrix

| Mode | CLI flag | Runs on CUDA | Output quality (greedy, llama-3.1-8b-4bit) | Decode perf vs Fp16 | Notes |
|------|----------|--------------|--------------------------------------------|---------------------|-------|
| Fp16 | `fp16` | yes | baseline | 1.00x | default |
| Int8 | `int8` | yes (fixed) | byte-identical to Fp16 on the probe prompt after the scale fix | slower, worsening with context (see table) | plain MLX quantize/dequantize; memory win only |
| Turbo4Asym | `fp16+turbo4` (`turbo4-asym`) | yes | coherent, small drift (lossy 4-bit V) | ~0.6x at short ctx | default decode path = dequant-first native SDPA (plain MLX) |
| Turbo3Asym | `fp16+turbo3` (`turbo3`) | yes | coherent, near-identical on probe | ~0.6x at short ctx | 3-bit V host round-trip dequant |
| Turbo4 (sym) | `turbo4` | yes | coherent | ~0.5x at short ctx | non-allowlisted models fall back to Turbo4Asym |
| Turbo4Delegated | `turbo4-delegated` | yes | byte-identical to Fp16 on probe | ~1.0x at short ctx | FP16 K + FP16 hot-V tail keeps it near-lossless |

Fallback paths were also exercised and are crash-free on CUDA:

- `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0` forces the Sparse-V graph reference (`attention_sparse_v_turbo4`, plain MLX) - runs.
- `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA=0` forces the steel -> cold-only -> graph fallback order; steel/cold-only return `None` on non-macOS, so it lands on `delegated_graph_attention` (plain MLX) - runs.

The Turbo decode tok/s values above are short-context spot checks from the functional probe (48-token greedy generations that still pay first-run CUDA kernel compilation on the first mode); they are indicative, not benchmarked steady-state numbers. Only Int8 was benchmarked with warmup (below), because Int8 is the mode this issue targets for long-context memory relief.

## Int8 vs Fp16 decode benchmark (warmed, `mlxcel-bench-decode`)

Synthetic prompt of the stated length, 128-256 measured decode tokens after a 16-token warmup pass, one process per configuration. Contexts >= 8k use `MLXCEL_PREFILL_CHUNK=2048`.

| Context | Mode | Decode tok/s | Decode ratio (int8/fp16) | MLX peak (GB) | Peak delta |
|---------|------|--------------|--------------------------|---------------|------------|
| 2048 | fp16 | 51.56 | - | 7.08 | - |
| 2048 | int8 | 38.23 | 0.74x | 5.65 | -1.43 |
| 8192 | fp16 | 44.18 | - | 7.18 | - |
| 8192 | int8 | 23.33 | 0.53x | 6.42 | -0.76 |
| 32768 | fp16 | 27.95 | - | 10.37 | - |
| 32768 | int8 | 8.62 | 0.31x | 8.18 | -2.19 |

Raw CSV: `benchmarks/cuda_gb10_issue635_kvquant_2026-07-10.csv`.

## Why Int8 decode is slower, not faster, on this path

The issue anticipated a 15%+ decode speedup at 32k from halved KV read traffic. The measured result is the opposite: Int8 decode is 26% slower at 2k, 47% slower at 8k, and 69% slower at 32k, worsening monotonically with context.

The reason is the shape of the current Int8 path. `update_and_fetch` in Int8 mode dequantizes the entire live INT8 K/V window back to a full FP16 tensor on every decode step, then runs the standard FP16 SDPA against it. So the attention kernel still reads a full FP16 KV stream (no bandwidth saved at the kernel), while the per-step full-window dequant adds work that grows linearly with context. There is no fused INT8-KV SDPA kernel that reads the packed INT8 directly, which is what would actually cut the KV read bandwidth. Building such a kernel is out of scope here and overlaps the quantized paged-attention work (#634).

The peak-memory column is the whole-run MLX high-water mark and at short context is dominated by prefill transients rather than resident KV, so the 2k/8k deltas are noisy. At 32k the KV cache dominates and the -2.19 GB delta matches halving the ~4.3 GB FP16 KV cache almost exactly.

## CUDA recommendation

On GB10, Int8 KV is a memory-capacity lever, not a decode-speed lever. Use it when an FP16 KV cache would not fit (very long single-stream contexts, or packing more concurrent sequences into unified memory), accepting the decode-rate cost above. Do not enable Int8 KV to make decode faster: it is slower at every context measured, and the gap widens with length. For pure speed, keep Fp16 (default). A future fused INT8-KV SDPA / paged-attention kernel (#634) is the prerequisite for turning the halved KV footprint into a decode-throughput win.

Turbo modes carry additional V-quantization quality loss and no decode-speed advantage on CUDA today (their fused kernels are Metal-only and fall back to graph paths here), so they are not recommended on CUDA beyond experimentation; Fp16 or Int8 (for memory) are the practical CUDA choices.

## Server integration (Int8)

Int8 serving works for the common case and was verified end to end on CUDA:

- Chunked prefill + decode on a 520-token prompt returns coherent output identical to Fp16.
- Prompt-cache donate/adopt works: two requests sharing a long system prefix reuse the Int8 KV prefix. The second request reported `cached_tokens: 256` and produced coherent output, so the detached Int8 handle (INT8 buffers + scale sidecars) round-trips correctly through the pool. Existing tests `cache_pool_detach_adopt_preserves_int8_round_trip` and `property_detach_adopt_decode_matches_fresh_prefill_int8_within_tolerance` cover the handle round-trip; `int8_kv_fetch_recovers_values_within_one_quant_step` locks in the scale-fix accuracy.

The KV-cache layer's Int8 front-trim is correct: `trim_front` handles `Fp16 | Int8` and slices the INT8 scale sidecars (`key_scales` / `val_scales`) in lockstep with the INT8 buffers (Turbo modes return 0, a documented no-op). The unit tests `int8_kv_trim_front_then_fetch_matches_untrimmed_tail` and `int8_kv_prefill_grow_trim_then_decode_tracks_fp16` (prefill across the buffer-grow boundary, front-trim, then decode appends) confirm the dequantized visible window tracks the Fp16 reference within one quant step.

Known limitation (separate, pre-existing bug): `--max-kv-size N` with a prompt longer than N produces degenerate output on the DENSE batched-decode backend. This is NOT Int8-specific: `--kv-cache-mode fp16 --decode-storage-backend dense --max-kv-size 256` on a 520-token prompt produces the same garbage, while `fp16` on the default paged backend (where the dense front-trim is a no-op and capacity is a pool-side concern) stays coherent. Int8 always uses the dense backend, so it always exposes this path. Because the KV-cache-layer Int8 trim is proven correct and Fp16-dense fails identically, the defect is in the dense batched-decode orchestration after a trim (model-forward level), not in the Int8 cache. Recommendation until that is fixed: do not combine `--max-kv-size` with the dense decode backend (and therefore with Int8). Tracked as follow-up issue #718.
