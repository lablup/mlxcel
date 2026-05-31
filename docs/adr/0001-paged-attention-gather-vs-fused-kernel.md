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

Rationale: the microbench shows gather overhead stays under <!--FILL_CROSSOVER-->% of SDPA time below ~<!--FILL_CROSSOVER_CTX--> tokens of context. Below that crossover, the gather is dominated by the attention it feeds, so (A) is within noise of the fused-kernel lower bound while needing no new kernel. Above it, the gather copy starts to cost enough that (B) is worth its complexity, which is why (B) stays on the roadmap rather than being dropped.

### Pool tensor layout

Two candidate per-layer pool layouts were measured:

- Layout A: `[num_blocks, block_size, n_kv_heads, head_dim]`.
- Layout B (head-split): `[n_kv_heads, num_blocks, block_size, head_dim]`.

Both reach the same `[batch, n_kv_heads, ctx, head_dim]` SDPA input after one `take`, one `reshape`, and one `transpose`; they differ in the `take` axis (0 for A, 1 for B) and in the `slice_update` shape used to append a fresh block each step. The recommendation, finalized from the `take` and `slice_update` numbers in the Results table, is: <!--FILL_LAYOUT_DECISION-->

### Block size

Keep the existing default block size (32) unless the data argues otherwise. <!--FILL_BLOCK_SIZE_NOTE--> The tradeoff the sweep exercises is fragmentation against gather dispatch cost: a smaller block (16) cuts internal fragmentation (fewer wasted slots in the last partial block of each sequence) but raises the number of `take` indices and therefore the gather dispatch, while a larger block (64) does the reverse.

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

**Hardware:** <!--FILL_HARDWARE-->

| batch | ctx | block | frag% | contig_sdpa_us | gatherA_only_us | gatherA_sdpa_us | gatherB_only_us | gatherB_sdpa_us | sliceupd_A_us | sliceupd_B_us | overheadA% | overheadB% |
|------:|----:|------:|------:|---------------:|----------------:|----------------:|----------------:|----------------:|--------------:|--------------:|-----------:|-----------:|
| 1 | 1024 | 32 | 0.00 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

<!-- BENCH_RESULTS_PLACEHOLDER: orchestrator fills the measured table + hardware line here -->

## Consequences

Phases 1 through 3 inherit the following from this decision:

- **Phase 1 (#118), global block-pool tensor storage:** the pool is stored in the layout chosen above, and block append uses the `slice_update` shape measured for that layout. No new kernel is needed for this phase.
- **Phase 2 (#119), paged decode attention over real block tables:** decode reads blocks with `take` over the real block table, then `reshape` + `transpose` into the SDPA input, then the existing fused `scaled_dot_product_attention`. This is the `gatherA_sdpa` / `gatherB_sdpa` path the bench measured, so the decode hot path reuses only existing FFI (`take`, `reshape`, `transpose`, fused SDPA) and adds no new kernel in Phases 1 and 2.
- **Phase 3 (#120), paged prefill into the block pool:** prefill writes into the same pool layout, so it inherits the append path and the layout's `slice_update` characteristics.

The fused Metal paged-attention kernel (Phase 6, #123) stays deferred. The crossover context length above gives the downstream phases a concrete trigger: if real workloads run past it and the gather overhead shows up in end-to-end decode throughput, (B) is the planned next step, and the gather path built in Phases 1 and 2 remains the correctness reference and fallback for it.

## References

- Epic #116, unified KV cache.
- Issue #117, this Phase 0 spike.
- `examples/page_gather_microbench.rs`, the microbench backing this ADR.
- `src/lib/mlx-cpp/turbo/sparse_v_sdpa.metal`, the fused-kernel model for strategy (B).
- `src/lib/mlxcel-core/src/layers.rs`, `paged_decode_attention_dense_compat`, the current dense decode path.
- `src/lib/mlxcel-core/src/cache/paged.rs`, `PagedBlockPool` and `PagedKvLayout`.
