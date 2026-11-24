# Paged sliding-window decode on GB10 (issue #3256)

Decode tok/s of Gemma 3 with its FP16 sliding layers stored as window-bounded paged blocks, measured against `main` at `56bfb92a`. The script is the interleaved null-arm `bench_decode.py`. Arms `main-a`, `branch` and `main-b` rotate each round, with prompt cache off, `--parallel 4` and `-c 65536`.

The copy in this directory adds `BENCH_ALLOW_EARLY_STOP=1`. With it, a row that ends before its token budget shortens the window in which every row streams, and no longer aborts the cell (see "The early stop" below). The rate was always computed over that window only. It also adds `BENCH_LOGIT_BIAS`, sent as the request's `logit_bias` on every arm; `{"106": -100}` keeps a greedy row from ending on `<end_of_turn>`, so both arms decode the same context range.

Host: GB10, shared. `uptime` is recorded at the start and end of each run.

## Current gate: B=1 still fails

### After the profile-driven fixes (commit `ea2036e4`)

Both runs use `BENCH_LOGIT_BIAS='{"106": -100}'` on all arms. The binary was built before two whitespace-only edits (a wrapped C++ line and an inlined Rust binding).

`gemma3-1b-gate-graph-budget.json`, 3 rounds x 3 reps, every cell. The load average started at 0.7, and a CI job (a release build, then a browser test holding the GPU lock between cells) ran on the host during round 2 and pushed it to 3.3. A cached `cargo clippy` of this branch also ran for about 40 seconds during round 0 of `main-a`.

| prompt | B | main-a | main-b | branch | null spread | branch vs main |
|---|---|---|---|---|---|---|
| 256 | 1 | 182.50 | 190.75 | 177.68 | 8.25 | -5.7% |
| 256 | 4 | 372.54 | 365.59 | 373.86 | 6.95 | +1.0% |
| 8192 | 1 | 182.27 | 178.72 | 171.63 | 3.55 | -4.4% |
| 8192 | 4 | 348.11 | 342.55 | 354.03 | 5.56 | +2.4% |

Every 256 x 4 row on every arm now decodes its full 1024 tokens, so that cell compares the same context range on both arms, and it is within the null spread. Both B=4 cells are at or above main.

`gemma3-1b-gate-graph-budget-b1.json`, the B=1 cells again with 6 rounds x 3 reps on a quiet host (load average 0.9 at start and end, nothing else running):

| prompt | B | main-a | main-b | branch | main range | null spread | branch vs main |
|---|---|---|---|---|---|---|---|
| 256 | 1 | 190.65 | 186.50 | 176.43 | 179.24 to 194.73 | 4.16 | -5.9% |
| 8192 | 1 | 179.76 | 179.56 | 168.21 | 162.92 to 190.44 | 0.20 | -6.4% |

The branch's per-round median is below both main arms in 11 of 12 rounds. B=1 went from 8% to 18% below main to about 6% below, and it is still outside the null spread in both cells. The profile below attributes what remains.

### First run: at 5fc1fc7b

`gemma3-1b-b1.json`, 3 rounds x 3 reps. In this run every pool-backed layer gathered its visible keys out of the pool on every B=1 step.

| prompt | B | main-a | main-b | branch | null spread | branch vs main |
|---|---|---|---|---|---|---|
| 256 | 1 | 194.13 | 190.56 | 161.82 | 3.57 | -16.6% |
| 8192 | 1 | 188.92 | 184.24 | 142.34 | 4.68 | -23.6% |

### Runs with the fused single-row read

At B=1 every pool-backed layer now reads its keys in place through the fused paged kernel at any length (`paged_single_decode_attention`), so the step does no gather. The server metrics confirm it: over 511 steps x 26 layers, 13286 fused launches and 0 gathers. Neither arm runs the lookahead pipeline at B=1.

Early experiments, 2 rounds x 2 reps each:

| run | prompt | B | main-a | main-b | branch | null spread | branch vs main |
|---|---|---|---|---|---|---|---|
| batch entry, default floor (`gemma3-1b-b1-gather-floor.json`) | 256 | 1 | 196.39 | 195.17 | 149.71 | 1.21 | -23.5% |
| batch entry, default floor | 8192 | 1 | 182.27 | 188.76 | 143.69 | 6.49 | -22.7% |
| floor forced to 1 (`gemma3-1b-b1-floor-1.json`) | 256 | 1 | 196.60 | 193.21 | 181.15 | 3.39 | -7.8% |
| floor forced to 1 | 8192 | 1 | 183.38 | 182.53 | 157.06 | 0.84 | -14.4% |

`gemma3-1b-gate-single-row.json` (commit `bab355a1`; the binary was built from the same tree before `cargo fmt`, which changed whitespace only), 3 rounds x 3 reps, load average 2 to 3:

| prompt | B | main-a | main-b | branch | null spread | branch vs main |
|---|---|---|---|---|---|---|
| 256 | 1 | 197.04 | 197.86 | 168.47 | 0.82 | -14.7% |
| 256 | 4 | 385.15 | 373.55 | 324.49 | 11.60 | -14.7% |
| 8192 | 1 | 188.97 | 186.85 | 155.43 | 2.12 | -17.1% |
| 8192 | 4 | 360.09 | 346.86 | 371.15 | 13.23 | +4.7% |

The 256 x 4 loss had a separate cause. Before a row fills its 512-token window, four rows hold fewer than the 2048 visible tokens the batched floor asks for, so each sliding layer gathered every row's window every step. With both floors forced to 1 (`gemma3-1b-b4-floor-1.json`, 2 x 2), the same cell gave main 389.63 / 390.32 against branch 408.65 (+4.7%). That experiment lifted the floor for every layer, full-attention ones included. Commit `31803cae` drops the batched floor only for layers that are windowed on every row; its own numbers are below.

`gemma3-1b-gate-windowed-floor.json` (commit `31803cae`), 3 rounds x 3 reps. The load average rose from 2 to 17 during this run because a CI runner was compiling on the host, so the B=4 cells were rerun below.

| prompt | B | main-a | main-b | branch | null spread | branch vs main |
|---|---|---|---|---|---|---|
| 256 | 1 | 190.66 | 190.07 | 171.06 | 0.59 | -10.1% |
| 256 | 4 | 326.71 | 324.94 | 387.53 | 1.77 | +18.9% |
| 8192 | 1 | 182.92 | 183.77 | 151.25 | 0.85 | -17.6% |
| 8192 | 4 | 325.21 | 315.33 | 369.12 | 9.88 | +16.1% |

B=4 rerun at commit `31803cae` (`gemma3-1b-gate-windowed-floor-b4-rerun.json`), 3 rounds x 2 reps. It started at a load average of 3.5 and ended at 1.2. A `cargo clippy` of this branch ran on the host for the first 9 of its 15 minutes, while every arm was being measured.

| prompt | B | main-a | main-b | branch | main range | null spread | branch vs main |
|---|---|---|---|---|---|---|---|
| 256 | 4 | 400.25 | 397.18 | 385.85 | 361.94 to 416.16 | 3.07 | -3.0% |
| 8192 | 4 | 365.10 | 363.11 | 371.98 | 357.52 to 381.21 | 1.99 | +2.3% |

At 8192 the branch is above main. At 256 it is 13 tok/s (3.0%) below main, about four null spreads, so that cell is unresolved: the overlapping clippy run and the early stop below may account for part of it, and neither was ruled out. With `BENCH_ALLOW_EARLY_STOP`, the branch's 256 x 4 window ends near 598 tokens in rounds 0 and 2 (the row that stops early), while main's runs to 1024, so the two arms are not measured over the same context range in that cell.

Before the profile, across four separate runs, B=1 was 8% to 15% below main at 256 tokens and 14% to 18% below at 8192. The profile below found most of that in graph commits caused by the pool slab's size, not in the pool write or the page table.

## B=1 decode profile

Gemma 3 1B 4-bit, one request, under Nsight Systems with `--cuda-graph-trace=node` (`profile/run.py`), exported with `nsys export -t sqlite` and summarized by `profile/win.py`. Each figure is per decode step over the last 400 steps of a 528-token generation, so prefill and warm-up are excluded. Tracing slows the host, so the traced wall times are longer than untraced ones; the untraced per-step times are in the next table. Arms: `main` at `56bfb92a`, `b4win` at `31803cae` (the code measured in the gates above), `graph` with the byte-budget overlay fix only, and `fix` with all three fixes below.

| per step | main 256 | b4win 256 | graph 256 | main 8192 | b4win 8192 | graph 8192 | fix 8192 |
|---|---|---|---|---|---|---|---|
| kernels | 557 | 711 | 688 | 570 | 711 | 688 | 619 |
| CUDA graph launches | 29 | 54 | 34 | 29 | 54 | 34 | 31 |
| GPU busy, us | 3765 | 3861 | 3887 | 3986 | 4171 | 4208 | 4149 |
| attention kernels, us | 501 | 403 | 418 | 675 | 701 | 713 | 712 |
| Q cast, output cast and scale fill kernels | 0 | 78 | 78 | 0 | 78 | 78 | 26 |
| KV write (`copy_gg_nd`, 52 launches on every arm), us | 57 | 72 | 80 | 69 | 84 | 90 | 87 |
| `cudaGraphExecUpdate` host time, us | 1218 | 1200 | 1734 | 1446 | 1444 | 1612 | 1621 |
| `cudaMallocAsync`, `cudaMemcpy*`, `cudaEventSynchronize` calls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Untraced, `profile/step.py`: median of three (528 - 16 tokens) / 512 differences per arm, arms run back to back. The highest main and b4win figures come from the first pair, run at a load average of 1.4.

| ms per step | 256 | 8192 |
|---|---|---|
| main (seven runs) | 5.12 to 5.98 | 5.41 to 6.21 |
| b4win (three runs) | 6.37 to 7.56 | 6.41 to 11.13 |
| graph (two runs) | 5.45, 5.45 | 5.89, 6.31 |
| fix (two runs) | 5.37, 5.58 | 5.79, 5.86 |
| main, `MLX_MAX_MB_PER_BUFFER=1000` | 5.10 | 5.44 |
| b4win, `MLX_MAX_MB_PER_BUFFER=1000` | 5.43 | 5.93 |

What the profile attributes, and what it rules out:

- **The pool reservation, through the graph byte budget (largest cost).** At `-c 65536` each layer's pool is one slab of 2048 blocks, 32 MiB per side. A fused paged launch registers both slabs as inputs, 64 MiB, and the CUDA backend commits the graph once its inputs pass 25 MiB on this device (`get_graph_limits`, compute capability 12.1). So the graph committed after every attention launch: 54 graph launches per step against main's 29. Raising the budget to 1000 MB on both arms closes most of the gap, which isolates the cause. The kernels read only the pages their page table names, so the `custom_kernel.cpp` overlay now leaves the inputs of `mlxcel_paged_attention*` kernels out of the byte count (`set_input_bytes_uncounted` in the `device.cpp` overlay). The reservation itself costs nothing else per step: no allocation, copy or synchronization falls in the decode window.
- **Per-launch helper kernels.** Each fused launch cast Q to f32 and built its scale with `full`, two extra kernels per layer per step. The kernel now widens Q as it stages it, like K and V, and the scale is built from host data. Widening fp16 to f32 is exact, so the bytes do not change (`an_fp16_query_reads_the_same_bytes_as_its_f32_cast`). The output cast back to fp16 remains (26 per step).
- **Pool write.** One in-place write per side per layer, the same 52 small copies main makes into its dense caches, about 15 us per step slower in total. There is no whole-slab copy in the decode window.
- **Page-table view and plan.** Host-only work: no upload copies appear in the window, since the index arrays are built from host data.
- **What remains.** The fused read splits a B=1 request into one chunk per page and then merges the chunks: 17 chunks per sliding layer and about 260 per full layer at 8192. That merge (26 kernels per step, 117 us at 256 and 267 us at 8192) and the output cast are the extra kernels left, 49 per step over main. Partial plus merge is cheaper than main's attention at 256 (403 against 501 us) and about level at 8192 (712 against 675 us). The rest of the gap is host time spent building and updating graphs with more nodes in them.

## B=4 correctness: diagnosed, not a window bug

The first full run aborted on the branch's 256 x 4 cell, where one row stopped at 598 of 1024 tokens with `ignore_eos` set. `repro_b4.py` reproduced it, and `divergence/diverge.py` reran it with token logprobs. `divergence/summary.json` holds the comparison and `divergence/cmp.py` produced it.

1. Membership is exact.
   - `paged_sliding_batch_tests.rs` checks every row of every step against attention computed on the host over that row's last 512 positions. It covers:
     - B=4 above the batched floor, with dropped head blocks and a mid-page window start;
     - B=4 below the floor, including rows that have not yet filled their window;
     - B=1.
   - On the tiny Gemma 3 fixture, paged and ring logits differ by at most 7e-7 against logits of about 0.9 over 48 decode steps past the window. That is summation order. A window that held one position too many or too few would show as a gross error.
2. The fused kernel is not the cause. With `MLXCEL_PAGED_ATTENTION_NATIVE=0`, which runs gather plus dense SDPA:
   - the branch still differs from main;
   - main does not differ from itself.
3. Where the outputs differ:
   - Branch and main logprobs are bit-identical until the row crosses the window (decode step 256 for a 257-token prompt). After that, B=4 row 3 differs at two steps by 0.36 and 0.11 nats, and the other differing steps across rows are at most 0.078 nats, most of them 0.031 or less. Attributing these to summation order is an inference from the tiny-fixture gap below, not a measurement on this model.
   - The ring and the paged window hold the same 512 keys in a different order (rotated against temporal), so SDPA sums them in a different order.
   - Every text divergence is a greedy flip at an exact or near-exact tie:
     - B=1, step 337: -1.5625 vs -1.5625;
     - B=4 row 1, step 262: -2.2421875 vs -2.2578125;
     - branch-gather, step 400: -0.859375 vs -0.859375.
   - The B=1 256-token prompt whose text differed from main (round 1 of the first run) is the same kind of tie flip.
4. The early stop is a `main` behavior. After row 1's flip, the text reaches a point where the model emits `<end_of_turn>` (id 106). `ignore_eos` suppresses only the declared EOS (id 1). `divergence/stopprobe.py` sends the branch's text to the main binary:
   - main also stops at 0 tokens;
   - `logit_bias {"106": -100}` lets it continue;
   - `{"1": -100}` does not.

Four full-length rows that match main byte for byte were not achieved on the bench's own prompts. After the window, the branch's rows can take a different greedy path at a tie, and one such path ends on a token that `ignore_eos` does not suppress on either binary.

## Not measured

- gemma-4-12b-it-4bit (Gemma 4 is not migrated).
- The prompt-cache replay.
- Engine parity.

The run stopped at the B=1 gate, as the issue requires.
