# Paged sliding-window decode on GB10 (issue #3256)

Decode tok/s of Gemma 3 with its FP16 sliding layers stored as window-bounded paged blocks (branch `feature/issue-3256-paged-sliding-kv` at `5fc1fc7b`) against `main` at `56bfb92a`, measured with the interleaved null-arm script `docs/benchmark_results/scheduler-owned-state-decode-gb10-2026-11-23/bench_decode.py` (arms `main-a`, `branch`, `main-b`, rotated each round, prompt cache off, `--parallel 4`, `-c 65536`).

Host: GB10 (shared, load average 3 to 5 during the runs).

## Result: the B=1 gate fails

`gemma3-1b-b1.json`, `BENCH_CELLS=256x1,8192x1`, 3 rounds x 3 reps, 512 generated tokens per request:

| model | prompt | B | main-a | main-b | branch | main range | null spread | branch vs main |
|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 1 | 194.13 | 190.56 | 161.82 | 182.62 to 199.37 | 3.57 | -16.6% |
| gemma-3-1b-it-4bit | 8192 | 1 | 188.92 | 184.24 | 142.34 | 179.79 to 193.28 | 4.68 | -23.6% |

Every branch sample is below every main sample in both cells. The loss grows with context (-16.6% at 256, -23.6% at 8192), so it is not a fixed per-step overhead alone: at B=1 every pool-backed layer gathers its visible keys out of the pool before the dense SDPA, and the full-attention layers gather O(len) per step.

The first full run (`gemma3-1b-first-run-partial.jsonl`) showed the same B=1 cell at 256 (162 to 164 against 192 to 194) before it aborted on the B=4 cell below.

## B=4 correctness: one row diverges and stops early

The first run aborted at the branch's 256x4 cell: one row ended at 598 of 1024 tokens with `ignore_eos` set. `repro_b4.py` reproduces it with the bench's prompts (salt 124 to 127), non-streaming, on both binaries (`b4-repro-main.json`, `b4-repro-branch.json`):

| row | main | branch | common prefix (chars) |
|---|---|---|---|
| B=1 | length 1024 | length 1024 | 5187 of 5187 |
| B=4 row 0 | length 1024 | length 1024 | 5187 of 5187 |
| B=4 row 1 | length 1024 | stop 598 | 1589 of 5157 |
| B=4 row 2 | length 1024 | length 1024 | 4726 of 4726 |
| B=4 row 3 | length 1024 | length 1024 | 6272 of 6272 |

Row 1 diverges from main after about 300 generated tokens, close to where the sequence crosses the 512-token sliding window, then emits a stop token. The batched step runs the fused paged v2 kernel once the batch reaches the 2048-visible-token floor; the unit test of the batched path (`a_batched_paged_decode_matches_each_row_alone`, window 32) stays below that floor and only covers the gather fallback. The cause is not isolated.

At B=1 the 256-token cell also produced different text from main for one of the three prompts (round 1, every rep), while the other two prompts and the 8192 cell matched main exactly.

## Not measured

The B=4 throughput cells (no valid branch number while one row stops early), gemma-4-12b-it-4bit (Gemma 4 is not migrated), and the prompt-cache replay. The run stopped at the B=1 gate as the issue requires.
