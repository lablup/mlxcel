# Paged sliding-window decode on GB10 (issue #3256)

Decode tok/s of Gemma 3 with its FP16 sliding layers stored as window-bounded paged blocks, measured against `main` at `56bfb92a`. The script is the interleaved null-arm `bench_decode.py`. Arms `main-a`, `branch` and `main-b` rotate each round, with prompt cache off, `--parallel 4` and `-c 65536`.

The copy in this directory adds `BENCH_ALLOW_EARLY_STOP=1`. With it, a row that ends before its token budget shortens the window in which every row streams, and no longer aborts the cell (see "The early stop" below). The rate was always computed over that window only.

Host: GB10, shared. `uptime` is recorded at the start and end of each run.

## Current gate: B=1 still fails

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

The 256 x 4 loss had a separate cause. Before a row fills its 512-token window, four rows hold fewer than the 2048 visible tokens the batched floor asks for, so each sliding layer gathered every row's window every step. With both floors forced to 1 (`gemma3-1b-b4-floor-1.json`, 2 x 2), the same cell gave main 389.63 / 390.32 against branch 408.65 (+4.7%). Commit `31803cae` therefore drops the batched floor for layers that are windowed on every row.

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

Both B=4 medians lie inside main's sample range. At 256 the branch median is one null spread below main's, and at 8192 it is above.

Across four separate runs, B=1 is 8% to 15% below main at 256 tokens and 14% to 18% below at 8192, far outside a null spread of 1 to 3 tok/s. Main's B=1 path keeps dense caches and never touches the pool. The branch pays a pool write, a page-table view and a fused launch per layer per step. The loss grows with context, which points at the full-attention layers: they also live in the pool on the branch and are dense on main.

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
   - Branch and main logprobs are bit-identical until the row crosses the window (decode step 256 for a 257-token prompt). After that they differ by at most 0.36 nats per token. Gemma 3 logits are bf16, so that is a few bf16 ulps.
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
