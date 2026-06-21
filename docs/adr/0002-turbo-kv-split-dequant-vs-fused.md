# ADR 0002: Turbo KV decode is split dequant plus native SDPA, and the upstream sparse-V speedup does not carry over

**Status:** Accepted (2026-06-20); amended 2026-06-21 with the #370 result. Follows #369 (Turbo4Asym routed through dequant-SDPA), motivated #370 (fuse V dequant into the attention kernel). Backed by the 2026-06-19 sweep under `benchmarks/turbo_kv/` and the A/B measurements below. The 2026-06-21 addendum records that the #370 fused-kernel lever did not pan out and why; the decision stands and is now measured on both ends.

## Context

TurboQuant KV modes store the V cache (and for symmetric `turbo4` the K cache) as a rotated, codebook-quantized 4-bit tensor. Decode attention has to turn that back into something the dot product can consume. The TurboQuant code is a Rust port of [turboquant_plus](https://github.com/TheTom/turboquant_plus); its [sparse-V paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md) reports **+22.8% decode at 32K** from skipping the V dequant for positions whose post-softmax attention weight is below `1e-6`, with zero perplexity change. Two questions drove this ADR: is mlxcel's port faithful (is the skip exact, or does it cost quality), and does the +22.8% carry over to mlxcel's decode path.

mlxcel has two V-handling strategies for `Turbo4Asym`, both reachable at runtime:

- **sparse-V skip.** Graph FP32 `Q@K^T` and `softmax` produce the attention weights, then a JIT Metal kernel (`src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp`) does the V weighted-sum and `continue`s past any position below threshold. Selected with `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0`; the kernel is on by default.
- **dequant-then-SDPA.** Dequantize the whole V cache to FP16, then call native MLX `scaled_dot_product_attention` for the entire attention. This is the #369 default.

Both are *split*: scores and softmax run outside the V-handling step. The upstream is a single fused flash-attention where dequant, scores, softmax, and accumulation interleave in one kernel. That structural difference decides the outcome.

## Findings

### Sparse-V is exact, not lossy

`DEFAULT_THRESHOLD` is `1e-6`, the paper's value, and the source notes any threshold in `[1e-8, 1e-4]` gives identical PPL. A skipped position contributes its attention weight (`< 1e-6`) times a bounded V vector to the weighted sum, below FP16 round-off, so dropping it does not change the output. The `sparse_v_kernel_threshold_zero_matches_graph` correctness test holds at threshold 0. The skip is exact at the default; an earlier internal note calling sparse-V "lossy" was a mischaracterization, not a port bug.

### The sparse-V skip buys no decode speed, and +22.8% does not reproduce

A/B on `qwen3-30b-a3b` (standard attention) at 16K and on `qwen3.5-35b-a3b` (the paper's exact model, `Qwen3.5-35B-A3B`) at 32K, M1 Ultra, decode tok/s, ratio against the same model's fp16:

| Path (turbo4-asym) | qwen3-30b-a3b @16K | qwen3.5-35b-a3b @32K |
|---|---|---|
| fp16 (reference) | 40.34 (1.00x) | 49.83 (1.00x) |
| dequant-SDPA (#369 default) | 10.51 (0.26x) | 54.56 (1.10x) |
| sparse-V kernel, tau=1e-6 | 10.47 (0.26x) | 55.34 (1.11x) |
| sparse-V graph, kernel off (full dequant) | 10.55 (0.26x) | 54.59 (1.10x) |
| sparse-V kernel, tau=1.0 (skip all V) | 6.08 (0.15x) | 54.39 (1.09x) |

The paper's +22.8% is the sparse-V kernel against its own full-dequant baseline. The mlxcel analog is the kernel row against the graph (full-dequant) row: +1.4% at 32K (55.34 vs 54.59), inside run-to-run noise, against the paper's +22.8%. The skip earns nothing measurable even on the exact model in the 32K regime the paper studied.

### Root cause: the V step is a negligible decode fraction in a split design

Skipping the *entire* V weighted-sum (`tau=1.0`) lands at 54.39 on the 32K hybrid, the same as full dequant (54.59). Removing all V-sum work changes nothing, because scores and softmax run in graph FP32 ops before the V kernel and the FFN runs regardless, so the V weighted-sum the kernel optimizes is a small slice of the step. The paper's +22.8% lives where V dequant is a large, eliminable fraction, which is inside a fused flash-attention. mlxcel does not have that kernel. (The `tau=1.0` cell on the standard-attention 16K model reads 6.08, slower than the working configs; that all-dead-mask path is a degenerate config whose output is zeros and whose timing is not representative, included only to bound the V-sum work.)

### Quantized KV is slower than fp16 on cache-bound models, neutral on small-KV hybrids

On the four standard-attention sweep models every quantized KV mode decodes slower than fp16 (`qwen3-30b-a3b` 16K turbo4-asym 0.26x), because the per-token dequant cost exceeds the cache-read bandwidth the 4-bit V saves. `qwen3.5-35b-a3b` is a hybrid: `full_attention_interval: 4` puts a real KV cache on only about 10 of its 40 layers (2 KV heads each), so it is not cache-bound, decode stays near 50 tok/s, and turbo4-asym is roughly fp16-neutral. The +9% there sits within run variance and the small-KV effect, not the paper's cache-bandwidth win.

## Decision

Keep dequant-SDPA (#369) as the `Turbo4Asym` decode default. Do not revive sparse-V as a speed path: it is exact and worth keeping as a runtime option and correctness reference, but its skip buys nothing in mlxcel's split design. Treat quantized KV as a memory-footprint trade, not a decode-speed feature; PR #376 aligns the `--recommend-quant` advisor, the `bench_kv_cache.sh` gates, and `docs/turbo-kv-cache.md` to that.

The one lever that could in principle beat the current 0.24x to 0.40x decode ceiling is **#370**: fuse the V dequant into the attention kernel so V is never materialized and the dequant rides inside the scores plus softmax plus accumulate. That would capture all three things the split design cannot: the long-context cache-bandwidth win, a sparse-V skip that actually removes hot-path work, and no materialize round-trip. The 2026-06-21 addendum below records that mlxcel's only available fused kernel does not actually win, so this prediction did not hold.

## Addendum (2026-06-21): the fused-kernel lever does not pan out (#370)

#370 took the first route the issue proposed: route `Turbo4Asym` decode through the existing delegated steel-envelope kernel (`attention_turbo4_delegated_steel`). `Turbo4Asym` is structurally the all-cold case of `Turbo4Delegated` (FP16 K, every visible V token 4-bit packed, no hot FP16 tail), so the kernel serves it directly with `cold_offset = offset, hot_offset = 0`. The kernel fuses the cold-V dequant into the attention accumulate, never materializing the full V. Parity held: a 16-step `Turbo4Asym` decode through the steel kernel matched the dequant-SDPA reference at RMS < 5e-3 (threshold 0, no skip).

The speed did not. M1 Ultra, `qwen2.5-7b-4bit`, decode tok/s (`benchmarks/turbo_kv/2026-06-21_Apple_M1_Ultra_qwen2.5-7b-4bit_370_asym_fused_attempt.csv`):

| mode | 4K | 16K | 32K |
|---|---|---|---|
| fp16 | 94.83 (1.00x) | 68.95 (1.00x) | 50.62 (1.00x) |
| turbo4-asym, dequant-SDPA (#369) | 28.61 (0.30x) | 22.76 (0.33x) | 14.55 (0.29x) |
| turbo4-asym, steel fused (#370 attempt) | 10.68 (0.11x) | 4.41 (0.06x) | 2.15 (0.04x) |
| turbo4-delegated | 63.50 (0.67x) | 48.38 (0.70x) | 34.64 (0.68x) |

The fused kernel **regresses** asym decode 3x at 4K and 7x at 32K, the opposite of the goal. Two facts explain it:

1. **The custom fused kernel is slower than native SDPA over a materialized V.** The steel envelope computes scores in graph, then does a two-pass online softmax and a per-token nibble-unpack plus codebook-lookup plus accumulate sweep over every cold token in one Metal dispatch. At 32K that hand-written sweep loses badly to MLX's optimized flash-attention GEMM running on a transiently materialized FP16 V. Materializing the rotated V and calling native SDPA, which is exactly what the dequant-SDPA path does, is the faster arrangement on this hardware.

2. **`Turbo4Delegated`'s 0.66-0.80x does not come from the steel kernel.** The delegated decode path defaults to dequant-SDPA (`turbo4_delegated_dequant_sdpa_enabled()` is on by default; `update_and_turbo4_delegated_attention` returns from the dequant-SDPA branch before it ever reaches the steel try). The steel envelope is delegated's *slow fallback*, used only when dequant-SDPA is disabled. So the premise that motivated #370, that delegated is fast because it fuses V dequant into the kernel, was wrong: delegated is fast because it dequantizes only the cold body, keeps a hot FP16 ring, and hands the result to native SDPA.

There is no fuse-into-native-SDPA option, because MLX's SDPA kernel consumes a dense V; feeding it packed V would mean modifying the upstream kernel. The mlxcel-side fused kernel is the only fusion available, and it is slower.

### Decision (amended)

Do not route `Turbo4Asym` through the fused kernel. `Turbo4Asym` stays on the exact dequant-SDPA path (#369), the simple full-precision-K option that quantizes every V token with no hot ring, at roughly 0.3-0.5x decode. For the fp16-K + 4-bit-V use case that wants speed, the recommended mode is **`Turbo4Delegated`**: about 4x V compression at 0.66-0.80x fp16 decode, via cold-only dequant-SDPA plus a hot FP16 ring. The `--recommend-quant` advisor already promotes `Turbo4Delegated` here (post-#376); no advisor change is needed.

The remaining asym-versus-delegated decode gap (about 0.4x versus 0.7x) is the hot/cold split, not V-dequant fusion. Closing it would mean giving `Turbo4Asym` a hot ring, which is exactly what makes a cache `Turbo4Delegated`. Keeping the two modes distinct (delegated = fast with a hot ring, asym = simple and fully quantized) is the intended design, so the gap is left as is.

## Reproduce

All runs use `mlxcel generate ... -n 60 --profile` on a prompt repeated to the target token count, M1 Ultra, `--release --features metal,accelerate`. The cache mode is `turbo4-asym` throughout; the path is selected by environment variable:

```bash
# dequant-SDPA (#369 default)
MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=1 mlxcel generate -m models/<moe> -p "<32k prompt>" -n 60 \
    --kv-cache-mode turbo4-asym --profile

# sparse-V skip kernel, tau=1e-6 (paper-faithful skip)
MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0 MLXCEL_SPARSE_V_THRESHOLD=1e-6 mlxcel generate ... --kv-cache-mode turbo4-asym --profile

# sparse-V graph fallback, full dequant, no skip (the paper's baseline)
MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0 MLXCEL_SPARSE_V_THRESHOLD=1e-6 MLXCEL_SPARSE_V_KERNEL=0 mlxcel generate ... --kv-cache-mode turbo4-asym --profile
```

Models: `qwen3-30b-a3b-4bit` (standard attention, cache-bound) and `qwen3.5-35b-a3b-4bit` (the paper's model, hybrid linear attention). The full six-mode sweep that establishes the fp16-relative ratios for the standard-attention models is in `benchmarks/turbo_kv/`.

## Consequences

- **Advisor and docs.** PR #376 frames Turbo KV as a memory trade and promotes `int8` and `turbo4-delegated` as the fast quantized picks. This ADR is the measured why.
- **sparse-V kernel.** Kept as a runtime option (`MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0`) and the correctness reference for any future fused kernel, not recommended for speed.
- **#370 closed without a fused asym path.** The fused-kernel attempt regressed asym decode 3-7x (see the 2026-06-21 addendum); the mlxcel-side fused kernel is slower than native SDPA over a materialized V, and delegated's speed comes from dequant-SDPA, not the steel envelope. The decision is to keep `Turbo4Asym` on dequant-SDPA and steer the fp16-K + 4-bit-V speed use case to `Turbo4Delegated`.

## References

- [turboquant_plus sparse-V paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md), the +22.8% claim and its baseline framing.
- Issue #367 (the slow sparse-V bug report), PR #369 (the dequant-SDPA fix), issue #370 (fuse V dequant), issue #354 (the sweep that backs PR #376).
- `src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp` and `sparse_v_sdpa.metal`, the sparse-V skip kernel.
- `src/lib/mlxcel-core/src/cache/turbo/sparse_v.rs`, the threshold and dispatch.
- `benchmarks/turbo_kv/`, the 2026-06-19 four-model sweep.
- [ADR 0001](0001-paged-attention-gather-vs-fused-kernel.md), the parallel gather-vs-fused-kernel decision for paged attention.
