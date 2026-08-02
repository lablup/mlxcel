# Cascade attention (shared prompt prefixes)

Cascade attention computes a prompt prefix shared by several concurrent sequences **once per decode step** instead of once per sequence. It is the compute-side counterpart to the storage-side deduplication mlxcel already performs.

Status: opt-in, off by default. Correctness is verified against both a host f64 reference and the flat fused launch, on Metal, with f32 and f16 pools. Throughput on a real serving workload has not been measured yet, which is why it is off; see [Measuring it](#measuring-it).

## What is duplicated today

The paged pool refcounts blocks, and `CachePool::clone_detached_paged_prefix` hands the *same* block ids to every sequence that adopts the same APC entry. Four clients behind one 8K system prompt therefore hold one physical copy of that prompt's KV, not four.

The compute is not deduplicated. In a batched decode every sequence attends over the shared span independently, so the attention bandwidth spent on that span scales with the batch even though every byte read is identical. At batch 4 with an 8K shared prefix, three quarters of the KV bytes the attention reads are re-reads of the same pages.

## The decomposition

Softmax states compose. If a key range is partitioned into disjoint parts, and each part yields its own normalized output `V_i` together with its log-sum-exp `LSE_i`, then

```
m = max_i LSE_i
w_i = exp2(LSE_i - m)
V = (sum_i w_i V_i) / (sum_i w_i)
```

is exactly the attention over the whole range. That is the same algebra the cross-CTA split-KV decode uses, and it runs on the same kernel.

A cascade decode step is therefore:

1. **Level 0**, the shared span, attended once for the whole subgroup.
2. **Level 1**, each request's private suffix, attended per request.
3. One merge, pairing each member's two states.

The result is bit-comparable with the flat path up to f32 rounding, because the two levels partition exactly the key range the flat launch reads.

## Why level 0 is one request with many query heads

This is the part that makes it a win rather than two launches doing the same work.

The fused v2 partial kernel gives one threadgroup all the query heads of one KV head, for one `(request, page tile)` pair. That threadgroup loads each K and V element once and reuses it across every query head it owns. Handing the same page list to `M` separate requests would read the span `M` times; handing it to **one** request with `M` times as many query heads reads it once.

The kernel maps query head `h` to KV head `h / NRep`, so the stacking order is load-bearing. With `G = Hq / Hkv`, member `m` and group slot `g`, the level-0 head index must be

```
h0 = kv_head * (M * G) + m * G + g
```

which is a `[M, Hkv, G, D] -> [Hkv, M, G, D]` transpose on the way in and its inverse on the way out. Stack member-major instead and every query head silently reads the wrong KV head: same shapes, successful launch, wrong answer. `paged_v2::cascade_launch_tests::member_major_head_stacking_reads_the_wrong_kv_head` is the negative control that pins it.

No kernel was written or modified for this feature. Both levels are ordinary v2 launches and the merge is issue #898's `paged_attention_merge_states`, whose contract (`v_in [N, H, D]` already normalized, `lse_in [N, H]` in **log2** units, `o_indptr [M + 1]` grouping contiguous rows) is honoured rather than adjusted.

## How the shared span is found

Detection compares the CSR page table's `indices` positionally across requests, not the pool refcounts. Within one `PagedCsrView` every live block id resolves to exactly one physical pool row, and two live block ids never share a row, so two requests whose page `i` is the same row hold the same block, which means `refcount > 1` and means the bytes are identical. Reading it off the page table keeps the decision inside `mlxcel-core`, needs no scheduler plumbing, and catches sharing that arose any way at all (an APC adoption, a forked sequence, a future block-hash dedup) rather than only sharing the server knows it created.

Candidates are grouped by their first emitted page, and the subgroup that removes the most page reads (`shared_pages * (members - 1)`) wins.

## What is refused, and why

| Condition | Reason |
|---|---|
| A request whose window starts mid-page (`first_page_offset != 0`) | a sliding window has trimmed into the middle of the first page, so the span is no longer expressible in whole pages. Such a request is dropped from the subgroup; the rest may still cascade. |
| A span reaching a member's last page | the last page of a request is the only one that may be partially filled, and level 0 declares its pages full. The span is capped at `pages - 1` for every member. |
| Fewer shared pages than `MLXCEL_CASCADE_MIN_SHARED_PAGES` | the extra launches cost more than the duplicated reads they remove. |
| Fewer members than `MLXCEL_CASCADE_MIN_MEMBERS` | with one member there is no duplication. |
| Logit soft-cap, multi-token steps, non-pool-backed caches, multi-slab layers | declined upstream by the #899 whole-batch path, before cascade is consulted. |

Sequences outside the chosen subgroup are not excluded from the step. They keep their whole page range at level 1 and their merge group is a single row, which the merge kernel resolves to the identity. One launch serves a mixed batch.

## Flags

| Variable | Default | Meaning |
|---|---|---|
| `MLXCEL_CASCADE_ATTENTION` | off | `1`/`true`/`on`/`yes` enables cascade; `0`/`false`/`off`/`no` is the kill switch. Anything else, including unset, takes the shipped default. |
| `MLXCEL_CASCADE_MIN_SHARED_PAGES` | `16` | Whole pages a subgroup must share. 16 pages is 512 tokens at the default block size of 32. |
| `MLXCEL_CASCADE_MIN_MEMBERS` | `2` | Sequences that must share the span. |

All three are read once per process. With `MLXCEL_CASCADE_ATTENTION` unset the decode path does no page-table scanning at all and the flat launch is byte-identical to pre-#903 behaviour.

## Reading off which path ran

The whole-batch decode announces the first launch of each distinct outcome at `info`:

```
INFO ...: paged decode v2: fused v2 cascade launch (batch 4, 4 of them sharing 256 pages / 8192 KV tokens read once; 8 shared-span chunks, 4 suffix chunks)
INFO ...: paged decode v2: flat v2: the cascade decomposition was planned but its launch failed (...)
```

and `/metrics` carries the split:

```
mlxcel_paged_decode_launches_total{path="fused_v2"} 12288
mlxcel_paged_decode_launches_total{path="gather"}   0
mlxcel_paged_decode_launches_total{path="cascade"}  12288
mlxcel_cascade_shared_tokens_total     100663296
mlxcel_cascade_member_sequences_total  49152
mlxcel_cascade_failures_total          0
```

`cascade_shared_tokens_total / cascade_launches` is the mean shared-span length and `cascade_member_sequences_total / cascade_launches` the mean subgroup size. `cascade` is a subset of `fused_v2`, so a cascade count of zero while `fused_v2` climbs means the feature is enabled but never activating: either nothing is shared, or the thresholds are above the span the workload actually shares.

## Measuring it

The layer-level harness sweeps shared-prefix length against batch size and prints, for every cell, the launch statistics that attribute the arm:

```bash
cargo run --release --features metal,accelerate --example cascade_attention_bench -- \
    --shared 2048,8192 --batches 4,8 --tail 256 --steps 200 --warmup 40 --reps 5
```

It also runs the no-sharing overhead gate: the same shapes with nothing shared, timed with and without the detection scan in front of the flat launch.

End to end, against a running `mlxcel-server` with the prompt cache on:

```bash
python3 scripts/bench_serving_concurrency.py \
    --shared-prefix-tokens 2048 --prompt-tokens 32 \
    --concurrency 4,8 --max-tokens 256 --metrics
```

`--shared-prefix-tokens` makes every client send an identical system prompt and differ only in a short user question, which is what produces refcounted shared blocks; `--metrics` scrapes `/metrics` around each level and reports the paged decode path split, so a level that did not actually run cascade says so instead of being assumed. Run the same command with `MLXCEL_CASCADE_ATTENTION=1` and without it, on the same server build, for the before/after pair.

Record results under `docs/benchmark_results/cascade-attention-<hw>-<date>.md` following the format in [`benchmarks.md`](benchmarks.md). Until such a file exists, the default in `paged_v2::cascade::DEFAULT_CASCADE_ENABLED` stays `false`.

## Limitations

- One shared level. Nested sharing (a common system prompt plus a per-tenant sub-prefix) would need a third level and is out of scope.
- Detection runs per layer per decode step. The page table it reads is already cached across steps, but the scan itself is not memoized; the overhead gate above is what bounds its cost.
- The level-0 launch JIT-specializes on the member count, so the first decode step at each new subgroup size pays one kernel compilation.
- Chunk sizes for both levels come from the heuristic. The #906 autotuner is not wired to either cascade geometry.
- CUDA is untested. Both levels reuse the existing v2 kernels, whose CUDA bodies were already shipped unvalidated by #898; nothing here adds new CUDA code, but nothing here validates it either.
