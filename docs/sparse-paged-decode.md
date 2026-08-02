# Fused sparse decode via page indirection

Sparse-attention models decide which KV positions a query may see and then, historically, pay dense costs anyway. This document describes how mlxcel turns that selection into a page table for the fused v2 decode kernel instead, which family it is wired into, which family it is not and precisely why, and how to tell from a benchmark's output which path it actually ran.

Issue: #904, part of epic #909. Depends on #898 (the CSR page table and the v2 kernels) and #899 (the production dispatch policy).

## The reduction

The v2 decode kernel resolves visible token `i` of request `r` through `indices[indptr[r] + (first_page_offset[r] + i) / page_size]`. It never assumes those rows are adjacent, ordered, or complete: they are physical pool rows and the kernel simply reads them. A sparse selection is therefore not a new kernel, it is a **shorter page list**.

At `page_size = 1` each page holds one token, `indices` is the selected-position list itself, and the kernel's page indirection becomes a per-token gather executed inside the attention loop. No gathered copy is materialized at any point, and nothing is computed for a position that was not selected.

## Addressing a contiguous cache as a page pool

Neither family this issue targets uses `PagedBlockPool`. Both use a dense `KVCache`, whose storage is a `[B, H, Cap, D]` row-major allocation. `ContiguousCacheLayout` (`src/lib/mlxcel-core/src/cache/sparse_csr.rs`) bridges the two:

- At `page_size = 1` and one pool KV head the kernel's address arithmetic collapses to `row * D + d`, so the pool view of the allocation is `[B * H * Cap, 1, 1, D]`, a **pure reshape**.
- The reshape is of the **allocation**, not of the window `update_and_fetch` returns. That window is a token-axis slice of a step-padded buffer, so reshaping it would copy the whole cache and defeat the point. `KVCache::raw_kv_allocations` hands back the allocation plus the live length, and returns `None` for every mode whose rows are not raw fp16 K/V, for a pool-backed cache, and for a head-trimmed window.
- The row stride between heads is the **reserved capacity** `Cap`, not the live length. Using the live length aliases heads the moment the buffer has any slack, which is almost always.
- The KV-head axis is folded into the *request* axis: request `r = b * H_attn + h`. The query reshapes to match (`[B, Hq, 1, D]` to `[B * Hkv, n_rep, 1, D]`, also a pure reshape, because MLX GQA numbers query head `i` under KV head `i / n_rep`). One CTA still owns one KV head and a group of its query heads, so the work per CTA is unchanged.

The v2 kernel resolves K and V from the *same* `indices` entry, so a selection is only expressible when both allocations put `(b, h, t)` at the same row. `shared_row_mapping` enforces that. It holds automatically when the two allocations have the same shape. It also holds, for `B == 1` only, when they differ in head count, which is the MiniMax-M3 case: its index key rides at head `H_attn` of the K allocation, so K is one head wider than V, and at `b == 0` both bases collapse to `h * Cap`. Beyond batch 1 the sequence strides diverge and the launch is declined rather than mis-addressed.

## Token-exact, not block-granular

Issue #904 offers two granularities and asks for one to land with the other documented.

**Token-exact (`page_size = 1`) is what landed.** It attends to precisely the set the indexer chose, so greedy decode can be required to match the mask implementation exactly rather than accepting a measured quality delta.

**Block-granular (`page_size = C`) is not implemented here.** It requires the pool row of block `j` of request `r` to be `base(r) / C + j`, i.e. the per-request row stride must be a multiple of the block size. That holds for a `PagedBlockPool`, whose rows *are* blocks, and it is the natural path for a pool-backed cache: feed the selected block ids through as `indices` with the pool's own `block_size` as `page_size`, with no expansion step at all. It does not hold for a contiguous allocation, whose reserved capacity grows in `step` increments unrelated to `C`. Since both target families are contiguous-cache models, block-granular pages do not apply to either of them today.

A block-sparse model therefore still reaches a token-exact page table: `selection_from_blocks` expands block ids into token rows in one `O(selected)` device pass, which is a fraction of a percent of the attention traffic it replaces.

## No synchronization on the decode path

The selected *content* is a device value (an `argpartition` output). The selected *count* is not: a top-`k` selection picks `k` of something on every step by construction. That split is what keeps this off the critical path. `SparseCsrStructure` (`indptr`, `last_page_len`, `first_page_offset`, `seq_lens`) is derived on the host from `(requests, per_request)` alone, `indices` stays a lazy MLX array, and nothing is read back.

`SparseSelection::materialize` does read back and is reserved for tests and for `MLXCEL_SPARSE_PAGED_DUMP=1`, which prints each request's selected rows. Never enable the dump for a timed run.

## MiniMax-M3 block-sparse decode

`src/models/minimax_m3_layers.rs` tries the fused path at `l == 1` whenever the indexer is in its sparse regime (`kv_len > 2 * topk_blocks * block_size`), and falls back to the additive-mask path on any decline.

The scored set excludes the final block, which is appended unconditionally instead. It is the query's own block at decode, so the `local_blocks >= 1` force-keep already pins it; excluding it keeps the selected cardinality a function of `kv_len` rather than of the scores, and removes the only way a selected block could name a row past the live window (which, in a pool view, is another request's tokens rather than an out-of-bounds error).

`minimax_m3_tests::the_decode_selection_names_exactly_the_blocks_the_mask_path_keeps` asserts set equality between the page table's attended positions and the mask path's kept columns, across four context lengths including a partial final block. That is a stronger statement than comparing two attention outputs: a wrong block choice hides easily inside a plausible output tensor and not at all inside a set comparison.

The real 427B checkpoint does not fit on the development machine, so everything here is validated synthetically. This is stated in the code as well as here.

## DeepSeek Sparse Attention is **not** routed through this path

Issue #904 lists DSA (DeepSeek-V3.2, shared with GLM-MoE DSA) first. It did not land, and the reason is a property of the kernel rather than of the selection.

DSA decode computes, for selected position `t`:

```
score = scale * (q_latent . kv_latent[t])  +  scale * (q_pe . k_pe[t])
```

The second term reaches `mlxcel_core::layers::attention_from_ptr` as an **additive per-(head, position) mask** (`pe_scores`). The v2 kernel computes `scale * (q[h] . k_pool[row])` and has no input through which such a term can arrive. The selection plumbing in this document applies to DSA unchanged; the kernel does not.

Three ways out were considered:

- **Materialize the bias as a kernel input.** It has to be indexed by CSR slot, so it is `[B, Hq, S]`, and producing it needs the gathered `k_pe`, which is the gather being removed. Producing it over the full context instead is `[B, Hq, kv_len]` (8 MB at 16K with `Hq = 128`) and costs `Hq * kv_len * qk_rope_head_dim` MACs, eight times the gathered version at 16K. Strictly worse than the path it replaces.
- **Fold the positional term into the dot product** by making the pool row `concat(kv_latent, k_pe)`, so `q_cat . row` is the whole score and no bias exists. Numerically exact and needs no kernel change, but the DSA cache stores `concat(kv_latent, indexer_key)` as its K buffer and `k_pe` as its V buffer, two separate allocations, so building the concatenation costs an `O(kv_len * 576)` copy per layer per step (about 19 MB at 16K) against the 2 MB gather it would replace. Reordering the cache to store `concat(kv_latent, k_pe, indexer_key)` makes it free, but that changes the width and field order of the buffer that the Int8 and Turbo4 KV modes group and pack over, and DeepSeek-V3.2 (~685B) cannot be loaded to validate it.
- **Generalize the kernel** with a row stride plus a second K stream: inputs `k_extra_pool` and `q_extra` contributing `dot(q_extra, k_extra[row])` to the score, with template constants `RowStride`, `Dim`, `ExtraDim`. This is the shape FlashInfer's paged-MLA kernel uses (`head_dim_ckv` plus `head_dim_kpe`), and it composes with the selection plumbing above with **no cache change**: `k_pool` is the K allocation at `RowStride = kv_lora_rank + index_head_dim`, `Dim = kv_lora_rank`, and `k_extra_pool` is the `k_pe` allocation at `ExtraDim = qk_rope_head_dim`. Both reshape for free and both have one head, so they share a row mapping at any batch.

The third is the real answer and is the recommended follow-up. It was not attempted here because it modifies the kernel #899 just shipped into the production decode path, for a family whose only checkpoint cannot be run on any available machine, and a fused decode path that silently never activates is the most expensive failure mode this epic has produced.

One further fact that follow-up work will have to confront: with `Hkv = 1`, MLA's head folding yields **one** CSR request per sequence, so a batch-1 DSA decode faces the single-request dispatch floor of 4096 rather than the batched one. At the default `index_topk` of 2048 that launch is *below the floor* and would be declined even with the kernel in place. The floor has to be re-derived for `Hkv = 1` sparse launches before DSA can be expected to dispatch at all.

## Where the dispatch floor sits

#899 measured the fused kernel against gather and found exactly one losing shape, batch 1 at 1024 visible tokens (0.91x), which produced the two-regime floor in `paged_v2::dispatch`: 4096 visible tokens for a single request, 512 per request when batched. Two things move that floor for a sparse launch, in opposite directions:

- **The count that must clear the floor is the selected count, not the context.** The launch reads `S` rows per request, so a 32K context with a 4K selection is a 4K-token launch. Applying the floor to `kv_len` would dispatch launches whose real work is eight times smaller than the number that justified the dispatch.
- **The request count is multiplied by the KV heads.** Folding the head axis into the request axis turns a single sequence into `Hkv` requests, so a batch-1 sparse decode with `Hkv > 1` faces the *batched* floor. That is not a loophole: the losing measurement was caused by a plan that degenerates to a couple of pages per chunk and pays a merge for nothing, and `Hkv` requests of `S` pages each is the batched shape that measured 1.41x and 1.47x, not the batch-1 shape that measured 0.91x. It does **not** apply when `Hkv == 1` (MLA), as noted above.

For MiniMax-M3's shipped configuration (`Hkv = 4`, `topk_blocks = 16`, `block_size = 128`) a single-sequence decode is 4 requests of ~2048 rows, which clears the 2048-row floor with margin.

That part is derived, not measured. #899's table was taken on dense page lists. The floor is the same code and the same environment overrides (`MLXCEL_PAGED_V2_MIN_KV_TOKENS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`), so a re-measurement moves both together.

## A token floor is not enough: the sparsity gate

A token floor asks "is this launch big enough". It does not ask the question that decides a *sparse* launch, which is "is skipping worth the kernel you have to skip with". `mlx::fast::scaled_dot_product_attention` is a heavily tuned dense kernel; the v2 partial kernel is a scalar per-lane sweep. Reading half the data through a kernel with twice the constant factor is a loss.

`examples/sparse_paged_decode_bench` measures exactly that. MiniMax-M3 geometry (64 query heads, 4 KV heads, head_dim 128, `block_size` 128, `topk_blocks` 16), one repetition of 40 steps on an idle Apple Silicon host, so **indicative rather than a recorded result**:

| context | sparsity | fused vs mask | transient memory |
|---|---|---|---|
| 4096 | 2.0x | **0.67x** | -1.6 MB |
| 8192 | 4.0x | **0.77x** | -3.2 MB |
| 16384 | 8.0x | 1.22x | -16.2 MB |
| 32768 | 16.0x | 1.17x | -8.7 MB |
| 65536 | 32.0x | 2.06x | -16.4 MB |

Transient memory drops at every point, as expected: the mask path allocates a `[B, 1, 1, kv_len]` mask and lets SDPA stage the full window, while the fused path allocates only the workspace the plan sizes.

So a launch must also clear `MIN_SPARSITY_RATIO` (default **8**): the live window has to be at least that many times the selected count. The default sits **on top of the measured win** rather than interpolated into the unmeasured 4x-to-8x band. #899 argued the opposite way for its token floor, deliberately sitting below its weakest measured point, because what lay under that point was a missed opportunity. Here what lies under the point is a *measured regression*, and shipping a regression in a narrow band is worse than declining a modest win in one.

Two caveats on that table, both pointing the same way. The harness builds the mask on the host outside the timed region, while the real mask path rebuilds it on device every step through several `O(kv_len)` passes; and the harness hands the sparse arm a prebuilt block list, while the real path expands the selection in one `O(selected)` pass. Both flatter the mask arm, so the real crossover should sit at a lower sparsity than 8x. That makes the default conservative in the direction that cannot regress anything, and it is the number a proper measurement should revisit first.

For MiniMax-M3's shipped configuration the gate opens at a 16K context, which is exactly where issue #904 requires decode to improve. Between the point where `should_apply_sparse` fires (4097 tokens) and 16K, the mask path runs and says so.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MLXCEL_SPARSE_PAGED_ATTENTION` | enabled | `0`, `false`, `off`, `no` pin every caller back on its pre-#904 gather or mask path. Any other value enables the fused path, so a typo leaves the measured path in place. Read once per process. |
| `MLXCEL_SPARSE_PAGED_DUMP` | off | `1`, `true`, `on`, `yes` print each request's selected pool rows. **Synchronizes**; debugging only. |
| `MLXCEL_SPARSE_PAGED_MIN_SPARSITY` | `8` | Minimum `live_len / selected` ratio before the fused path is dispatched. `0` disables the gate, which is how a benchmark measures the regime the policy declines. A non-integer value falls back to the default. |

## Reading which path ran off the output

Every call returns a `SparseDecodeOutcome`, and `report_sparse_outcome_once` announces the first occurrence of each **kind** on stderr as well as through `tracing`. One flag per kind, not one global flag: a single one-shot reports only whatever happened first, which in a real server is a short warmup request, and a later permanent decline for a different reason then never surfaces.

stderr as well as `tracing` because the `mlxcel` CLI installs no tracing subscriber, so a `tracing::info!` on a CLI-only decode path prints nothing at any `RUST_LOG`. There are at most eight such lines in a process lifetime.

A fused launch reads:

```
sparse paged decode: fused sparse launch (4 request(s) x 2048 selected rows, 256 chunks, merge on)
```

and a decline names the numbers that produced it:

```
sparse paged decode: fallback: 128 selected rows across 4 request(s) is below the 2048-row dispatch floor
```

`mlxcel_core::paged_v2::sparse_decode_stats()` returns running totals for a harness that wants a count rather than a first-occurrence line.

## Benchmark harness

`examples/sparse_paged_decode_bench.rs` times one decoder layer's attention at MiniMax-M3's head geometry and sparse configuration, comparing the additive-mask arm against the fused sparse arm over the **same** selection.

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer

# Default sweep: 8K / 16K / 32K.
cargo run --release --features metal,accelerate --example sparse_paged_decode_bench

# Wider, more repetitions.
cargo run --release --features metal,accelerate --example sparse_paged_decode_bench -- \
    --contexts 16384,32768,65536 --reps 5 --steps 400

# Confirm the kill switch restores the mask path.
MLXCEL_SPARSE_PAGED_ATTENTION=0 cargo run --release --features metal,accelerate \
    --example sparse_paged_decode_bench
```

It is synthetic by necessity: DeepSeek-V3.2 and MiniMax-M3 are 685B and 427B, and neither fits on the development machine. Read the numbers as "what happens to the attention op", multiply by the sparse layer count to reason about a forward pass, and do not read them as end-to-end tokens per second.

Two guards make the output self-describing. The harness prints, by name, whether the sparse arm dispatched the fused kernel or fell back, together with the `SparseDecodeStats` delta, and it **refuses to report a timing comparison at all** when the sparse arm did not fuse. It also checks the two arms agree numerically before timing them, because a timing comparison between arms that disagree is not a comparison. #899 shipped a fused decode path that silently never activated and then benchmarked the fallback against itself; the resulting clean-looking null was nearly accepted as a real finding.

Run it serialized on an otherwise quiet machine and repeat it. A decode attention step at these sizes is well under a millisecond, so anything else resident on the GPU dominates.

## phi3-small and phimoe are also **not** routed

Issue #904 groups phi3-small and phimoe with MiniMax-M3 as "block-sparse". They are block-sparse, but not in the shape this page table can express.

`phi3small::build_blocksparse_mask` selects key block `kb` for query block `qb` of head `h` when `(qb >= kb) && ((qb - kb < local_blocks) || ((kb + h + 1) % vert_stride == 0))`. The selection is static (position-driven, not score-driven), which is fine, but the `+ h` term makes it vary **per query head**.

A CSR request owns one KV head and the `n_rep` query heads grouped under it, and the kernel derives a request's page range from `indptr[request]`. So all `n_rep` query heads of a request necessarily share one page list, and a per-query-head pattern is only expressible when `n_rep == 1`. Supporting it in general needs a per-query-head page range in the kernel, which is a larger change than #904's premise ("no dedicated sparse kernel is needed") allows for.

`kv_len <= local_blocks * block_size` already degenerates to plain causal attention in that family and is untouched.

## What is not covered

- **Prefill.** Both families keep their existing prefill paths (the DSA sparse additive mask, the MiniMax-M3 block mask). Multi-token steps are out of scope for #904 and are the natural follow-up.
- **CUDA.** Unvalidated. The v2 partial kernel's CUDA body was already marked unvalidated by #898, and nothing here was compiled or run against `nvcc`. The host-side selection code is backend-independent; the kernel it feeds is not.
- **DSA decode.** See above.
- **Real-checkpoint validation.** Neither target checkpoint exists on any machine available to this work.
