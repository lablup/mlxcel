# Sparse paged decode via page indirection: Apple M1 Ultra, 2026-08-03

Validation run for issue #904 (epic #909).

**Outcome: the fused sparse path wins from 16x sparsity upward, reaching 2.99x at
32x, and sits at parity at the 8x routing threshold.** Correctness agrees with the
dense mask path to 4e-4, and transient memory is lower at every fused point.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-904-sparse-paged-decode` |
| Harness | `examples/sparse_paged_decode_bench.rs`, 400 steps, 5 reps per cell, median |
| Geometry | q_heads 64, kv_heads 4, head_dim 128, block_size 128, topk_blocks 16 |
| Load average | 4.7 to 5.9 across three invocations |

CUDA is unvalidated: the host-side selection is backend-independent, the kernel is
not, and there is no CUDA hardware here.

## The harness refuses to produce a misleading comparison

At sparsity below the routing threshold the sparse arm falls back, and the harness
prints the decline and **declines to print timings at all**:

```
== ctx 4096 ==   2.0x sparsity
   sparse arm: FELL BACK (nothing fused) -> ... below the 8x the fused kernel needs
   sparse arm counters: fused +0, fallbacks +1
   NOT COMPARABLE: the sparse arm did not run the fused kernel, so any timing
   below would compare the mask path against itself.
```

That is the failure this epic hit three times: a benchmark comparing a path
against itself and reporting a clean-looking null. Here it is structurally
impossible to misread, because no number is printed. Every fused row carries its
own counters (`fused +1, fallbacks +0`) and the launch description.

## Results

Three independent invocations. Speedup is mask median over sparse median, so above
1.0 favours the fused sparse path.

| ctx | sparsity | routed | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|---|---|
| 4096 | 2.0x | no | not comparable | | | |
| 8192 | 4.0x | no | not comparable | | | |
| 16384 | 8.0x | yes | 0.94x | 1.02x | 1.04x | **1.02x** |
| 32768 | 16.0x | yes | 1.31x | 1.39x | 1.42x | **1.39x** |
| 65536 | 32.0x | yes | 3.15x | 2.14x | 2.99x | **2.99x** |

Absolute medians from run 1: at 32768, mask 1.0921 ms/step against sparse 0.8344;
at 65536, mask 1.8123 against sparse 0.5753.

### On the 8x threshold

`MIN_SPARSITY_RATIO` defaults to 8, and 8x measures at **parity**, not at a win:
median 1.02x over a 0.94x to 1.04x range. The first invocation alone read 0.94x,
which looks like a regression; two further invocations put it at break-even, so
the single-run reading was noise rather than signal.

The threshold was left at 8 rather than raised to 16 for two reasons. It does not
lose, so routing there costs nothing measurable. And the harness's own biases do
not point clearly in one direction: the mask arm's additive mask is built on the
host outside the timed region, while the sparse arm's block list is likewise
prebuilt, so both arms are flattered and the sign of the residual bias is not
established. Raising the floor to 16 would be justified if a production workload
shows a regression at 8x; the constant moves via `MLXCEL_SPARSE_PAGED_MIN_SPARSITY`
without a rebuild.

What is unambiguous is that the useful range starts at 16x, and that the payoff
grows steeply with sparsity rather than plateauing.

## Correctness

Agreement against the dense mask path at every fused point: max relative error
4.026e-4 at 16384, 4.406e-4 at 32768, 3.994e-4 at 65536, all in f16 storage.

The decisive correctness test is not the output comparison, though. It is
`minimax_m3_tests::the_decode_selection_names_exactly_the_blocks_the_mask_path_keeps`,
which asserts **set equality** between the page table's attended positions and the
additive mask's kept columns across four context lengths including a partial final
block. Comparing outputs can pass with a wrong block selection that happens to
produce a similar tensor; comparing the selected set cannot. On the kernel side,
`paged_v2::sparse_tests` matches a host reference restricted to the selection and
then **requires the result to differ from full dense attention**, so a kernel that
ignored the page list would fail rather than silently pass.

## Memory

Transient peak is lower on the fused path at every measured point, by 8.7 MB at
16384, 8.7 to 16.3 MB at 32768, and 17.4 to 33.7 MB at 65536. The gather copy the
issue set out to remove never existed on the MiniMax-M3 path, which was mask-based,
so this is the mask materialization going away rather than a gather.

## Scope corrections carried from the implementation

Two parts of the issue's premise did not survive contact with the code, and both
are recorded in `docs/sparse-paged-decode.md` rather than worked around:

- **DSA decode did not land.** Its decode score carries `scale * (q_pe . k_pe[t])`
  as an additive per-(head, position) term, and the v2 kernel has no input through
  which such a term can arrive. Materializing it requires the very gather the issue
  wants removed. The real fix is a row-stride plus second-K-stream kernel
  generalization, costed in the doc and not attempted.
- **phi3-small and phimoe cannot use this at all.** Their pattern varies per *query*
  head, while a CSR request shares one page list across the `n_rep` query heads
  under its KV head.

## Open items

- DSA routing, pending the kernel generalization above.
- Prefill on both families.
- Batch greater than 1 with MiniMax's side head is declined, because K has
  `Hkv + 1` heads and V has `Hkv`, so they share a row mapping only at `b == 0`.
- No real-checkpoint validation: neither the 685B nor the 427B model fits here.
- CUDA is unvalidated.
