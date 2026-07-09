# ADR 0001: Paged-attention gather strategy and KV pool tensor layout

**Status:** Accepted (2026-05-31). Part of epic #116 (unified paged KV cache), Phase 0 (#117).

## Context

Epic #116 introduces a unified paged KV store: one global pool of fixed-size physical KV blocks, indexed by a radix trie, with blocks shared (copy-on-write) across sequences that share a prefix. The decode step for a sequence then has to read KV from blocks that are scattered across the pool rather than laid out contiguously, because two sequences sharing a prefix point at the same physical blocks and a sequence's own blocks are allocated on demand as it grows.

Two attention strategies can serve that scattered read:

- **(A) gather-then-SDPA.** Use `take` to pull the sequence's physical blocks out of the pool by index, `reshape` + `transpose` them into the `[batch, n_kv_heads, ctx, head_dim]` shape the fused kernel expects, then call the existing fused `scaled_dot_product_attention`. This is a small delta from the current dense decode path and reuses only existing FFI. The risk is the extra gather copy on every decode step.
- **(B) fused Metal paged-attention kernel.** A custom kernel, modeled on the fused Sparse-V SDPA kernel in `src/lib/mlx-cpp/turbo/sparse_v_sdpa.metal`, that takes a block table and reads scattered blocks directly inside the attention kernel, with no separate gather copy. This is the lower bound on decode cost but is a large, hard-to-tune piece of Metal to write and maintain.

Apple unified memory changes the calculus relative to a discrete-GPU PagedAttention deployment. There is no host swap-out and no PCIe copy: KV blocks already live in memory the GPU addresses directly. The win the unified store is chasing is therefore memory sharing (one physical copy of a shared prefix) and prefill avoidance (reuse a cached prefix instead of recomputing it), not avoiding host transfers. That reframes the decode gather cost of strategy (A) as the main risk to quantify: if gathering scattered blocks per step is cheap relative to the SDPA it feeds, (A) captures the sharing and prefill wins without the cost of building (B).

This ADR is backed by the synthetic op-level measurements in `examples/page_gather_microbench.rs`. The current decode path it compares against is `paged_decode_attention_dense_compat` (`src/lib/mlxcel-core/src/layers.rs`), which already materializes a dense per-sequence K/V before the fused SDPA. The pool itself is `PagedBlockPool` (`src/lib/mlxcel-core/src/cache/paged.rs`), whose block size is configurable (the `profile_paged_decode_kernel` example and the existing paged decode tooling default to 32).

## Decision

### Attention strategy

Adopt **(A) gather-then-SDPA** for Phases 1 through 5 (#118 through #122). Defer the **(B)** fused Metal paged-attention kernel to Phase 6 (#123), and build it only if the measured gather overhead at the target context lengths is material.

Rationale: the microbench shows gather overhead (layout A, measured against the contiguous-SDPA lower bound) stays under ~15% of SDPA time below ~4096 tokens of single-sequence context. Below that crossover the gather is dominated by the attention it feeds, so (A) is within noise of the fused-kernel lower bound while needing no new kernel. Overhead then grows with both context length and batch: single-sequence decode reaches ~56% at 16384 tokens and ~67% at 32768, and batched decode (batch 4) is already ~48% at 1024 tokens and runs 2x to 3x the contiguous SDPA cost past 4096. So (B) earns its complexity for long-context or batched serving, which is why it stays on the roadmap rather than being dropped; the concrete trigger for building it is single-sequence context past ~16384 tokens, or any sustained batched decode.

### Pool tensor layout

Two candidate per-layer pool layouts were measured:

- Layout A: `[num_blocks, block_size, n_kv_heads, head_dim]`.
- Layout B (head-split): `[n_kv_heads, num_blocks, block_size, head_dim]`.

Both reach the same `[batch, n_kv_heads, ctx, head_dim]` SDPA input after one `take`, one `reshape`, and one `transpose`; they differ in the `take` axis (0 for A, 1 for B) and in the `slice_update` shape used to append a fresh block each step. The recommendation, finalized from the `take` and `slice_update` numbers in the Results table, is **layout A**, `[num_blocks, block_size, n_kv_heads, head_dim]`. Its gather-then-SDPA step is on average 2.1x faster than head-split layout B (1.2x to 3.2x across the sweep): taking on axis 0 keeps each block's `[block_size, n_kv_heads, head_dim]` slab contiguous, and MLX folds the trailing `[0, 2, 1, 3]` transpose into the SDPA read. Layout B takes on axis 1 and needs a `[1, 0, 2, 3]` transpose across the leading head axis, a scattered pattern MLX does not fuse into SDPA as cheaply, so `gatherB_sdpa` runs from roughly 2x the contiguous baseline at 4096 tokens to more than 4x at 16384. Block-append cost is layout-insensitive (`slice_update` averages ~680us for A and ~700us for B, within noise), so it does not offset layout A's gather advantage.

### Block size

Keep the existing default block size (32) unless the data argues otherwise. Block size has little effect on the gather-then-SDPA cost at a fixed context length: layout A overhead varies by only a few points across 16 / 32 / 64 (batch 1 at 4096 tokens is 12%, 15%, and 16% for blocks 16, 32, 64). The swept context lengths are exact multiples of every block size, so `frag%` is 0 throughout the Results table; in production the internal waste is at most one partial block per sequence, below `block_size / ctx`. The tradeoff the sweep exercises is fragmentation against gather dispatch cost: a smaller block (16) cuts internal fragmentation (fewer wasted slots in the last partial block of each sequence) but raises the number of `take` indices and therefore the gather dispatch, while a larger block (64) does the reverse.

## Microbench methodology and reproduce

The bench is fully synthetic and op-level. It allocates fake K/V tensors with `zeros` (only timing matters, not values), loads no model, and times each decode-step body with a warmup loop followed by `synchronize_default()`, then a timed loop that evals each result, then a closing `synchronize_default()`. Per-call cost is the total timed wall time divided by the iteration count. This is the same eval-per-iteration harness used by `examples/bridge_overhead_microbench.rs`.

Paths measured per `(batch, ctx, block_size)`:

- `contig_sdpa`: fused SDPA over a contiguous per-sequence K/V. This is the lower bound and the effective cost of the current `paged_decode_attention_dense_compat` path.
- `gatherA_only` / `gatherB_only`: the `take` + `reshape` + `transpose` of K and V for each layout, with no attention, to isolate gather cost.
- `gatherA_sdpa` / `gatherB_sdpa`: the full gather-then-SDPA decode step for each layout.
- `sliceupd_A` / `sliceupd_B`: the per-step append of one fresh block into the pool via `slice_update`, for each layout.

To keep `reshape` valid when the block size does not divide the context length, the materialized length is padded to `ctx_pad = ceil(ctx / block) * block`, and every path (including the contiguous baseline) attends over `ctx_pad` keys so the comparison is apples to apples. The reported `frag%` is `(ctx_pad - ctx) / ctx * 100`, the internal fragmentation the block size induces. Block ids are assigned in reverse pool order over a pool sized at 2x the needed blocks, so the gather reads genuinely scattered (non-contiguous) physical ids.

Sweep: context lengths 1024 / 4096 / 16384 / 32768, batch sizes 1 / 4, block sizes 16 / 32 / 64. Dimensions: `head_dim` 128, `q_heads` 32, `kv_heads` 8, dtype f16.

Reproduce:

```text
cargo run --release --features metal,accelerate --example page_gather_microbench
```

Run it under `caffeinate -i` so the host does not idle-throttle the GPU mid-run, and let the machine cool between sweeps; Apple Silicon down-clocks under sustained load, so a hot machine inflates the larger-context rows. The numbers in the Results table are measured on the spike machine and reproduce by re-running.

## Results

**Hardware:** Apple M1 Ultra (Mac Studio), 128 GB unified memory, macOS 26.5 (build 25F71). Built `--release --features metal,accelerate`. Each cell is the minimum of two full sweeps (the cooler run), 50 timed iterations after 20 warmup, dtype f16.

| batch | ctx | block | frag% | contig_sdpa_us | gatherA_only_us | gatherA_sdpa_us | gatherB_only_us | gatherB_sdpa_us | sliceupd_A_us | sliceupd_B_us | overheadA% | overheadB% |
|------:|----:|------:|------:|---------------:|----------------:|----------------:|----------------:|----------------:|--------------:|--------------:|-----------:|-----------:|
| 1 | 1024 | 16 | 0.00 | 333 | 598 | 382 | 767 | 533 | 327 | 320 | 14 | 60 |
| 1 | 1024 | 32 | 0.00 | 341 | 627 | 375 | 747 | 452 | 303 | 304 | 10 | 33 |
| 1 | 1024 | 64 | 0.00 | 303 | 575 | 336 | 676 | 405 | 294 | 302 | 11 | 34 |
| 1 | 4096 | 16 | 0.00 | 341 | 618 | 383 | 895 | 661 | 320 | 320 | 12 | 94 |
| 1 | 4096 | 32 | 0.00 | 333 | 612 | 385 | 910 | 662 | 317 | 321 | 15 | 99 |
| 1 | 4096 | 64 | 0.00 | 335 | 598 | 390 | 902 | 661 | 321 | 321 | 16 | 97 |
| 1 | 16384 | 16 | 0.00 | 455 | 716 | 692 | 1861 | 1829 | 397 | 434 | 52 | 302 |
| 1 | 16384 | 32 | 0.00 | 437 | 733 | 686 | 1877 | 1843 | 426 | 424 | 57 | 322 |
| 1 | 16384 | 64 | 0.00 | 437 | 727 | 691 | 1884 | 1853 | 441 | 425 | 58 | 324 |
| 1 | 32768 | 16 | 0.00 | 646 | 986 | 1070 | 3322 | 3338 | 683 | 687 | 66 | 417 |
| 1 | 32768 | 32 | 0.00 | 635 | 987 | 1074 | 3360 | 3395 | 679 | 697 | 69 | 435 |
| 1 | 32768 | 64 | 0.00 | 646 | 983 | 1065 | 3413 | 3461 | 681 | 685 | 65 | 436 |
| 4 | 1024 | 16 | 0.00 | 339 | 609 | 513 | 946 | 732 | 334 | 316 | 52 | 116 |
| 4 | 1024 | 32 | 0.00 | 340 | 614 | 514 | 904 | 739 | 314 | 323 | 51 | 118 |
| 4 | 1024 | 64 | 0.00 | 362 | 643 | 516 | 892 | 710 | 312 | 317 | 42 | 96 |
| 4 | 4096 | 16 | 0.00 | 483 | 727 | 1186 | 1880 | 2511 | 432 | 433 | 146 | 420 |
| 4 | 4096 | 32 | 0.00 | 490 | 724 | 1209 | 1903 | 2552 | 437 | 433 | 147 | 421 |
| 4 | 4096 | 64 | 0.00 | 500 | 729 | 1197 | 1920 | 2228 | 436 | 444 | 140 | 346 |
| 4 | 16384 | 16 | 0.00 | 940 | 1383 | 3446 | 5987 | 7371 | 1039 | 1047 | 267 | 684 |
| 4 | 16384 | 32 | 0.00 | 1018 | 1379 | 3408 | 6051 | 7685 | 1203 | 1266 | 235 | 655 |
| 4 | 16384 | 64 | 0.00 | 1513 | 1874 | 4503 | 9275 | 9108 | 1202 | 1284 | 198 | 502 |
| 4 | 32768 | 16 | 0.00 | 1757 | 2090 | 6551 | 11371 | 14245 | 1792 | 1850 | 273 | 711 |
| 4 | 32768 | 32 | 0.00 | 2110 | 2947 | 6567 | 12089 | 15233 | 1817 | 2048 | 211 | 622 |
| 4 | 32768 | 64 | 0.00 | 2289 | 3041 | 6695 | 13334 | 18431 | 1763 | 1761 | 193 | 705 |

Reading the table: layout A's `gatherA_only` can exceed `gatherA_sdpa` at short context (batch 1 at 1024 tokens is ~600us vs ~380us) because timing the gather alone forces MLX to materialize the full contiguous K/V, while the gather-then-SDPA path lets MLX fuse the `take`, `reshape`, and `transpose` into the fused-SDPA read without ever materializing that intermediate. The decode-relevant cost is `gatherA_sdpa` and the `overheadA%` derived from it, not `gatherA_only`. That fusion is the main reason strategy (A) stays cheap at the common context lengths: the per-step gather does not pay for a separate full copy of the sequence KV.

## Consequences

Phases 1 through 3 inherit the following from this decision:

- **Phase 1 (#118), global block-pool tensor storage:** the pool is stored in the layout chosen above, and block append uses the `slice_update` shape measured for that layout. No new kernel is needed for this phase.
- **Phase 2 (#119), paged decode attention over real block tables:** decode reads blocks with `take` over the real block table, then `reshape` + `transpose` into the SDPA input, then the existing fused `scaled_dot_product_attention`. This is the `gatherA_sdpa` / `gatherB_sdpa` path the bench measured, so the decode hot path reuses only existing FFI (`take`, `reshape`, `transpose`, fused SDPA) and adds no new kernel in Phases 1 and 2.
- **Phase 3 (#120), paged prefill into the block pool:** prefill writes into the same pool layout, so it inherits the append path and the layout's `slice_update` characteristics.

The fused Metal paged-attention kernel (Phase 6, #123) stays deferred. The crossover context length above gives the downstream phases a concrete trigger: if real workloads run past it and the gather overhead shows up in end-to-end decode throughput, (B) is the planned next step, and the gather path built in Phases 1 and 2 remains the correctness reference and fallback for it.

## Phase 6 outcome (#123): fused kernel built, gather stays the default

Phase 6 implemented the (B) kernel and benchmarked it against the (A) gather path. The kernel (`src/lib/mlx-cpp/turbo/paged_attention.cpp`, JIT-compiled via `mlx::core::fast::metal_kernel`) is a split-K flash-decoding attention: one threadgroup per (batch, query head), `NumSplits` SIMD groups split the token range, and within each SIMD group the 32 lanes partition the head dimension so the per-token QK dot product is a single barrier-free `simd_sum`. It reads scattered KV blocks straight from the pool through the block table, with no gather copy. It is correct to fp32 round-off: RMS about 2e-8 against `paged_decode_attention_pooled_fallback` over 200 fragmented decode steps, plus grouped-query and batched cases.

The throughput does not justify making (B) the default on Apple Silicon. Measured on the M1 Ultra spike machine (50 timed iterations after 20 warmup, f16 pool, head_dim 128, 32 q-heads, 8 kv-heads, block 32), the fused-vs-gather speedup (a value above 1 means the fused kernel is faster):

| ctx | b=1 | b=4 | b=8 | b=16 |
|----:|----:|----:|----:|-----:|
| 4096 | 0.50x | 1.30x | 1.01x | 1.36x |
| 16384 | 0.39x | 0.83x | 0.82x | 0.91x |

The fused kernel wins at 4096 tokens once decode is batched (batch >= 4), and it scales better with batch than the gather path, the behaviour this ADR predicted from gather overhead growing with batch. At 16384 tokens it runs about 0.8 to 0.9x of the gather path across batch sizes, so it loses in the regime this ADR named as the primary trigger. The cause is the one the Decision section anticipated for unified memory: there is no host copy to avoid, and MLX folds the gather's `take` / `reshape` / `transpose` into its tuned fused SDPA, so the (A) path already runs near the contiguous lower bound. A hand-written JIT kernel reading scattered f16 blocks does not beat that tuned SDPA on memory bandwidth at long context.

Phase 6 lands the kernel gated off. The native path is opt-in through the `MLXCEL_PAGED_ATTENTION_NATIVE` environment variable (or the `use_native_paged_kernel` argument on `paged_decode_attention_pooled`), and the default stays on the gather-then-SDPA path. The kernel keeps its value for the batched moderate-context regime where it wins, and as a starting point if a future device or a non-fused gather path shifts the trade-off. `examples/paged_attention_kernel_bench.rs` reproduces the table above.

## Chunked slab storage interaction (#235)

The pool later moved from one growable tensor per layer to a list of fixed-size slab tensors (32 blocks per slab), so growth appends a slab instead of reallocating and copying the whole layer. The fused kernel reads one contiguous pool buffer per side, so `paged_decode_fused` declines (returns `None`, the caller falls back to gather) on any layer that has grown past a single slab. Layers within their first slab keep the kernel available. Teaching the kernel per-slab base pointers is possible if the trade-off above ever flips; given the kernel is gated off by default and loses in the long-context regime, the decline is the accepted state. The gather path splits the block-row list into per-slab runs and concatenates the per-run `take` results, which keeps it byte-identical and within run noise of the single-tensor layout.

## Adaptive native-kernel selector (#331)

The `use_native_paged_kernel` request from the scheduler previously meant "attempt the fused kernel, fall back only when it declines". That let `paged_decode_attention_pooled` dispatch native in regimes the Phase 6 table shows it losing: at `b=1` a single-slab layer ran the kernel at ~0.9x of gather. #331 replaces the request-means-attempt behaviour with an adaptive selector, `select_pooled_paged_dispatch(batch_size, visible_len, slab_count, backend)` in `src/lib/mlxcel-core/src/layers.rs`, which dispatches the kernel only inside the island the Phase 6 table measured it winning and uses gather everywhere else.

The selector is native only when all four hold: the backend is Apple Silicon Metal (the kernel is Metal-only and the tables here are Metal); `batch_size >= 4` (Phase 6: `b=1` loses at 0.50x/0.39x, `b>=4` wins at 4096); `visible_len <= 4096` (Phase 6: 4096 wins, 16384 loses across every batch size, so the selector stops at the last measured winning context rather than extrapolating into 16384); and `slab_count <= 1` (the #235 decline above, hoisted into the selector so it never dispatches native for a layer the kernel would decline anyway). The decision is memoized in a last-key atomic cell: every layer in a decode step shares the same `(batch, visible_len, slab_count, backend)` key, so the pure selector runs at most once per distinct shape and later layers take a single relaxed atomic load. The pure function is a handful of integer comparisons, so the memo is a formality that pins "no per-token recompute" rather than a measured hot-path saving.

`MLXCEL_PAGED_ATTENTION_NATIVE` now overrides the selector both ways: the original force-on set (`1`/`true`/`on`/`yes` and uppercase) still pins the kernel, and #331 adds a symmetric force-off set (`0`/`false`/`off`/`no`) that pins gather. An unset or unrecognised value defers to the selector.

The chunked-slab reality (#235) narrows where native is reachable. Slab count is a per-layer property across all sequences sharing the pool, so a batched layer spans `B * ceil(visible_len / block_size)` rows; at `block_size` 32 any `B>=4` layer past ~256 tokens already exceeds one 32-row slab and the kernel declines. The batched moderate-context win the Phase 6 table recorded on the pre-#235 single-tensor pool is therefore unreachable today without the deferred multi-slab kernel. The reachable remnant is short-context batched decode (single-slab, `B>=4`), where the kernel still wins.

Reachability caveat: the in-server decode path does not currently call `paged_decode_attention_pooled` at all. Pool-backed model layers gate out of the native-kernel arm (`is_paged_backed()`) and route through the per-sequence `update_and_fetch` pool intercept, which gathers the visible window and runs standard SDPA. The pooled entry point, and with it this selector, is exercised today by the kernel bench, the unit/FFI tests, and external mlxcel-core API consumers; inside `mlxcel-server` it is latent. Wiring the pooled path into the scheduler decode (or retiring it) is tracked in #710.

**Hardware:** Apple M1 Ultra (Mac Studio), 128 GB, macOS 26.5. `--release --features metal,accelerate`, 50 timed iterations after 20 warmup, f16 pool, head_dim 128, 32 q-heads, 8 kv-heads, block 32. From `examples/paged_attention_kernel_bench.rs`. Fused = raw `paged_decode_fused`; speedup > 1 means the kernel beats gather:

| batch | visible_len | slabs | selector | gather_us | fused_us | speedup |
|------:|------------:|------:|:---------|----------:|:---------|--------:|
| 1 | 512 | 1 | gather | 355 | 386 | 0.92x |
| 1 | 1024 | 1 | gather | 352 | 391 | 0.90x |
| 1 | 4096 | 4 | gather | 577 | declined | — |
| 1 | 16384 | 16 | gather | 1201 | declined | — |
| 4 | 4096 | 16 | gather | 1355 | declined | — |
| 8 | 4096 | 32 | gather | 3442 | declined | — |
| 4 | 128 | 1 | native | 490 | 393 | 1.25x |
| 4 | 256 | 1 | native | 521 | 433 | 1.20x |
| 8 | 128 | 1 | native | 747 | 427 | 1.75x |

The table confirms the two design claims: single-slab `b=1` runs the kernel at a ~0.9x loss (which the selector now avoids by choosing gather, a small win over the old request-means-attempt path), and the only reachable batched island (single-slab `B>=4`) runs it at 1.20x–1.75x, where the selector chooses native. Everything at 1k/4k/16k with `B>=4` is multi-slab and declines to gather, so the selector and the kernel agree.

The multi-slab native-kernel spike stays out of scope. It is worth building only if live traces show sustained requests in the `B>=4`, ~4k, multi-slab regime; that evidence does not exist yet, so this ADR keeps the decline (#235) as the accepted state and the selector confines native to the single-slab island.

## Pooled entry point retired to a library-only API (#710)

#710 resolved the reachability caveat above. The pooled decode entry point (`paged_decode_attention_pooled` and its `select_pooled_paged_dispatch` selector) is retired to a library-only API rather than wired into the scheduler decode. The occupancy derivation below shows the selector's winning island is both unreachable under the shipped serving defaults and transient even when it is entered, so a wire-in would thread a new batched-decode arm through every transformer family (high blast radius, jitter-class parity risk) for a sliver of accelerated steps that real chat serving leaves within its first few generated tokens.

### Occupancy derivation (Apple M1 Ultra)

The kernel wins only on a single-slab layer (`slab_count <= 1`, the #235 decline hoisted into the selector). A slab is `POOL_SLAB_BLOCKS = 32` block rows (`src/lib/mlxcel-core/src/cache/paged.rs`) and a block holds `DEFAULT_PAGED_BLOCK_SIZE = 32` tokens (`src/server/batch/scheduler.rs`), so one slab is 32 x 32 = 1024 token rows per layer across every sequence resident in the shared pool: `PagedBlockPool::slab_count` returns the layer's whole slab-list length, not one sequence's rows. The pool stays single-slab only while the total allocated block rows satisfy `sum_i ceil(len_i / 32) <= 32`. For a batch of `B` equal-length sequences that is `B * ceil(len / 32) <= 32`.

Combined with the selector's `batch_size >= 4` floor and the shipped `--parallel 4` default (#714, `n_parallel = 4`), the reachable island at `B = 4` is `ceil(len / 32) <= 8`, i.e. `len <= 256` tokens per sequence counting the prompt, the chat-template framing, and generated tokens together. At `B = 8` it tightens to `len <= 128`, and at `B = 16` to `len <= 64`.

Two facts make that island negligible in production. First, it is unreachable under the shipped defaults: the #714 serving-throughput bench drives 512-token prompts, and at `B = 4` a 512-token sequence needs `ceil(512 / 32) = 16` blocks, so the pool holds `4 x 16 = 64` rows = 2 slabs and every layer is multi-slab from the first decode step. The #331 bench table above shows exactly this, with every `B >= 4` row at `ctx >= 512` (`ctx 4096`, 16-32 slabs) reading `declined` and only the `ctx 128 / 256` rows at `B = 4 / 8` staying single-slab and native. Real chat serving passes 256 total tokens per sequence almost immediately, since the system prompt and chat-template framing alone are tens of tokens before the first user turn. Second, the island is transient even when entered: the pool only appends slabs (#235: growth appends, existing slabs are never freed), so once total rows cross 32 the layer stays multi-slab for the rest of the request. A request that starts inside the island leaves it permanently within its first few dozen generated tokens and never returns, so the best case for a wire-in is a short burst of accelerated steps at the very start of short-prompt batched requests, after which every step is gather anyway.

An instrumented occupancy run adds nothing here. The pooled path has no server caller, so its measured production occupancy is definitionally zero today, and the geometry above is a closed-form bound on the hypothetical wired-in occupancy rather than a noisy sample that Apple Silicon thermal drift would blur.

### Decision

Retire `paged_decode_attention_pooled` and `select_pooled_paged_dispatch` (with their `MLXCEL_PAGED_ATTENTION_NATIVE` override and per-shape memo) to a library-only API: kept and tested for external mlxcel-core consumers and `examples/paged_attention_kernel_bench.rs`, and documented as not on the `mlxcel-server` decode path. The fused kernel (`PagedBlockPool::paged_decode_fused`), the selector, and the bench all stay, since they are tested, benchmarked library surface and the selector remains the correct dispatch gate for any consumer that does call the pooled entry point. `MLXCEL_PAGED_ATTENTION_NATIVE` remains a library-consumer control and an A/B pin for the bench, not a server knob. Nothing is deleted: the `use_native_paged_kernel` scheduler request is left in place because it also gates the live `paged_decode_attention_dense_compat` block-table decode path (`src/models/llama3.rs` and the other families), so it is shared plumbing, not dead code.

### What reopens this

- A serving workload that sustains `B >= 4` with total contexts at or under ~256 tokens per sequence (short-prompt, high-concurrency batched decode), where the single-slab island is entered often enough that a wire-in's transient burst pays for the cross-family blast radius.
- A multi-slab native kernel (the deferred #235 per-slab-base-pointer spike), which would lift the `slab_count <= 1` constraint and make the batched moderate-context win reachable at real context lengths. Live traces showing sustained `B >= 4`, ~4k, multi-slab decode would justify building it, per the #235 note above.

## References

- Epic #116, unified KV cache.
- Issue #117, this Phase 0 spike.
- `examples/page_gather_microbench.rs`, the microbench backing this ADR.
- `src/lib/mlx-cpp/turbo/sparse_v_sdpa.metal`, the fused-kernel model for strategy (B).
- `src/lib/mlxcel-core/src/layers.rs`, `paged_decode_attention_dense_compat`, the current dense decode path.
- `src/lib/mlxcel-core/src/layers.rs`, `select_pooled_paged_dispatch` and `paged_decode_attention_pooled`, the adaptive selector (#331).
- `src/lib/mlxcel-core/src/cache/paged.rs`, `PagedBlockPool` and `PagedKvLayout`.
- `examples/paged_attention_kernel_bench.rs`, the fused-vs-gather bench and selector cross-check (#123, #331).
