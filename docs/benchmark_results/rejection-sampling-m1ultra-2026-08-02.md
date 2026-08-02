# Dual-pivot rejection sampling: Apple M1 Ultra, 2026-08-02

Validation run for issue #901 (epic #909).

**Outcome: the kernel ships routed only where it wins.** top-p sampling gains
1.17x to 1.82x in a pipelined decode loop; top-k alone and min-p alone lose and
keep the stock `argpartition` chain. End-to-end decode with top-p is 1.049x.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-901-rejection-sampling`, base `a93d365e` |
| Harness | `examples/rejection_sampling_microbench.rs`, 200 iters, 30 warmup |
| End-to-end model | `mlx-community/Qwen3-0.6B-4bit` (vocab 151936) |

CUDA was written but never compiled or run: no CUDA hardware, no `nvcc`.

## The routing policy, and why it exists

The kernel replaces a **sort**. Where the stock chain does not sort, it cannot
win. That single sentence predicts the whole table: top-p converges in one or two
rounds at every vocabulary and wins; top-k needs two to seven rounds and the count
grows with vocabulary, so it loses; min-p is one round but the stock chain has no
sort to remove there either.

| configuration | measured | routed |
|---|---|---|
| top-p alone | 1.17x - 1.82x pipelined, every vocabulary | **yes** |
| top-k alone | 0.31x - 0.97x | no |
| min-p alone | 0.47x - 0.98x | no |
| top-k + top-p, vocab <= 32768 | 1.21x - 1.82x pipelined | **yes** |
| top-k + top-p, vocab > 32768 | mixed, batch 4 loses reproducibly | no |

**The llama-server default (top-k 40 with top-p 0.9) keeps the argpartition chain
on Qwen, Llama-3 and Gemma vocabularies.** It routes only on 32K-vocab models
(Mistral, Mixtral, Llama-2). That is the honest consequence of the measurement and
it means the headline configuration is unchanged on most models.

The 65536 joint case is excluded by measurement rather than omission: across three
repetitions batch 1 won (1.31/1.28/1.29) and batch 8 won (1.29/1.26/1.16) but
batch 4 lost every time (0.94/0.79/0.80).

## Two measurement modes, because one was misleading

The harness reports two speedups per row.

- **`iso`** times the sampler in isolation, synchronizing around each iteration.
- **`pipe`** reproduces the decode loop: build the next step, `async_eval` both,
  read the previous step's tokens, synchronize once at the end.

This matters because the first version of this kernel measured **1.17x isolated
and 0.58x end-to-end**. The eager entry point read the per-row convergence flags
back to the host every token, which absorbed the entire outstanding GPU queue and
serialized the decode loop. The isolated harness could not see it, because it
already synchronized and the forced sync was therefore free.

That is now a structural test rather than a timing observation:
`the_production_sampling_call_never_synchronizes` enqueues chained matmuls, calls
the sampler, and times the drain that follows. Pre-fix it recorded 86.293 ms of
"build" time against 63.3 us of outstanding GPU work. A call that only builds a
graph returns in microseconds, so the ratio is a hardware-independent witness.

The production path now evaluates nothing. The bit-space bracket bounds the loop
at 31 rounds, so the 32-round cap is unreachable; convergence flags are inspected
on a later call once MLX's non-blocking `is_available()` reports the launch
landed. An unconverged row has already returned its row argmax, which the kernel
guarantees lies inside the filtered support, so worst-case degradation is one
greedy draw, counted and announced rather than silent.

## Routed configurations, rebased branch

| config | vocab | batch | iso | pipe |
|---|---|---|---|---|
| top-p=0.9 | 32768 | 1 | 2.25x | 1.31x |
| top-p=0.9 | 32768 | 4 | 1.02x | 1.47x |
| top-p=0.9 | 32768 | 8 | 2.04x | 1.46x |
| top-p=0.9 | 65536 | 1 | 1.54x | 1.17x |
| top-p=0.9 | 65536 | 4 | 1.59x | 1.43x |
| top-p=0.9 | 65536 | 8 | 2.04x | 1.67x |
| top-p=0.9 | 152064 | 1 | 1.37x | 1.49x |
| top-p=0.9 | 152064 | 4 | 1.23x | 1.41x |
| top-p=0.9 | 152064 | 8 | 1.80x | 1.64x |
| top-k+top-p | 32768 | 1 | 1.61x | 1.21x |
| top-k+top-p | 32768 | 4 | 1.51x | 1.82x |
| top-k+top-p | 32768 | 8 | 1.45x | 1.29x |

Every routed row is above 1.0 in both modes. Unrouted rows read close to 1.00x in
`pipe` because both arms then execute the same code, which doubles as a live check
that the routing policy is in force rather than merely declared.

## End-to-end

Qwen3-0.6B-4bit, `--top-p 0.9 --temp 0.8 --seed 99 -n 200`, three repetitions.

| arm | runs (tok/s) | median |
|---|---|---|
| `MLXCEL_SAMPLING_REJECTION=1` | 277.72, 278.81, 279.83 | **278.81** |
| `MLXCEL_SAMPLING_REJECTION=0` | 265.86, 265.69, 265.48 | 265.69 |

**1.049x**, with spreads of 0.8% and 0.15%. The gain is small because sampling is
a minor share of a decode step on a 0.6B model, the same reason the Gumbel-max
kernel in #900 measured +2.3% end to end against much larger op-level gains. Before
the synchronization fix this same comparison read 145-155 against 258-261, a 1.7x
regression.

## Semantic note carried from the rebase

The kernel evaluates top-k and top-p against the untruncated row, while the stock
chain renormalizes between them, so for that one combination the kernel's support
is a superset. Issue #902's contract is that `fused_sample_probs` describes what
`fused_sample` draws from, and left alone that contract would have broken silently:
the speculative accept test would have received a `p` missing part of the target's
support and rejected tokens the target could genuinely produce. No existing test
covered it, because #902's cases are all configurations #901 does not route.

`fused_sample_filter_logits` now takes a flag that reorders the same stages,
applying the top-k and top-p masks independently against the untruncated row and
intersecting with `minimum`, which is exact because the masked fill is `lowest()`.
The flag is inert outside that one combination. Two guards were added, one checking
the reported distribution covers exactly what is sampled across six routed and
declined configurations, and one proving the two orderings genuinely differ so the
first cannot pass vacuously.

## Open items

- CUDA is unvalidated: written, never compiled, never run.
- The single-threadgroup-per-row design is what makes top-k lose; fixing it needs
  multi-threadgroup cooperation, which requires a grid barrier MLX custom kernels
  do not expose, or host-controlled rounds costing a synchronization per round.
  Survivor compaction would not help, because round 1 already costs three
  full-row single-threadgroup sweeps against the baseline's one full-GPU
  `argpartition`.
- top-p with min-p is not in the matrix and is the one extrapolation in the
  routing policy.
- Batch 4 through `mlxcel-server` was not measured; only the CLI path was.
