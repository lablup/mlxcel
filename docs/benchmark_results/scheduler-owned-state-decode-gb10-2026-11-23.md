# Scheduler-owned sequence state: decode rate, main against the branch (issue #3254), GB10, 2026-11-23

Issue #3254 moves the per-sequence state of Gemma 3 and Qwen 3.5 from the model wrappers into the scheduler's pool entry ([ADR 0010](../adr/0010-scheduler-owned-sequence-state.md)). The forward runs the same attention over the same caches, so the decode rate should not change. This page records the check.

Host: NVIDIA GB10 (driver 580.178.04), Linux, CUDA build. Base is `main` at `bd931f76`; the branch is `refactor/issue-3254-scheduler-owned-state` at `c98af10f`. Both are `cargo build --release --features cuda --bin mlxcel-server` builds, copied aside before measuring. The host is shared and other jobs ran on it during the session, which is what the null arm is for.

## Setup

[`bench_decode.py`](scheduler-owned-state-decode-gb10-2026-11-23/bench_decode.py) starts one server per arm and model (`--parallel 4 --no-prompt-cache -c 65536`) and sends `B` concurrent streaming `/v1/completions` requests of a `P`-token prompt, greedy, with `ignore_eos`. A cell is rejected unless every row reports a prompt within 5% of `P` and the full completion length.

- Arms: `main-a`, `branch`, `main-b`. `main-b` is the same binary as `main-a` (the null arm). The order rotates every round.
- Cells: `P` of 256 and 8192 tokens, `B` of 1 and 4.
- Rate: generated tokens per second summed over the rows, counted only inside the window where every row is streaming (latest first token to earliest last token). Rows prefill one after another, so a `B = 4` cell decodes 1,024 tokens per row at `P = 256` and 3,072 at `P = 8192` to make that window several seconds long; a `B = 1` cell decodes 512.
- The reply text of every `B = 1` request is hashed. It is identical across `main-a`, `main-b` and `branch` in every cell of every run.

"Null" below is the gap between the medians of `main-a` and `main-b`, as a share of the main median. "Branch" is the branch median against the median of both main arms. The last column says whether the branch median falls inside the range the two main arms covered.

## Run 1: three rounds, one sample per cell and round

Raw rows: [`run1.json`](scheduler-owned-state-decode-gb10-2026-11-23/run1.json).

| model | P | B | main-a | main-b | branch | main range | branch range | null | branch | in range |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 1 | 193.6 | 194.9 | 185.5 | 193.3 to 197.7 | 184.2 to 191.8 | 0.7% | -4.5% | no |
| gemma-3-1b-it-4bit | 256 | 4 | 387.4 | 395.4 | 381.1 | 381.1 to 401.5 | 374.1 to 396.1 | 2.0% | -2.6% | no |
| gemma-3-1b-it-4bit | 8192 | 1 | 188.0 | 186.0 | 184.7 | 184.6 to 188.6 | 180.1 to 186.7 | 1.1% | -0.8% | yes |
| gemma-3-1b-it-4bit | 8192 | 4 | 358.0 | 361.6 | 346.1 | 324.6 to 362.5 | 296.3 to 365.4 | 1.0% | -3.4% | yes |
| qwen3.5-0.8b-4bit | 256 | 1 | 189.4 | 182.0 | 174.8 | 177.7 to 200.2 | 171.5 to 183.1 | 4.0% | -5.9% | no |
| qwen3.5-0.8b-4bit | 256 | 4 | 446.1 | 442.8 | 443.8 | 438.5 to 459.6 | 373.7 to 446.6 | 0.8% | -0.1% | yes |
| qwen3.5-0.8b-4bit | 8192 | 1 | 171.1 | 167.0 | 160.1 | 166.5 to 174.9 | 151.6 to 173.6 | 2.4% | -5.3% | no |
| qwen3.5-0.8b-4bit | 8192 | 4 | 351.0 | 347.2 | 348.1 | 290.5 to 357.3 | 338.6 to 349.8 | 1.1% | -0.3% | yes |

Three samples per arm left four cells with a branch median below the main range, three of them at `B = 1`. A fixed per-step cost would show in both `B = 1` cells of a model at about the same size, and it does not for Gemma 3 (-4.5% at 256, -0.8% at 8192). The cells were repeated with more samples.

## Run 2: three rounds, two samples per cell and round

Raw rows: [`run2.json`](scheduler-owned-state-decode-gb10-2026-11-23/run2.json).

| model | P | B | main-a | main-b | branch | main range | branch range | null | branch | in range |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 1 | 193.2 | 192.9 | 190.1 | 168.8 to 196.8 | 187.9 to 197.5 | 0.2% | -1.5% | yes |
| gemma-3-1b-it-4bit | 256 | 4 | 391.3 | 384.1 | 378.8 | 370.2 to 398.5 | 369.7 to 387.4 | 1.9% | -2.2% | yes |
| gemma-3-1b-it-4bit | 8192 | 1 | 181.1 | 179.1 | 183.9 | 171.9 to 184.5 | 181.5 to 186.9 | 1.1% | +2.1% | yes |
| gemma-3-1b-it-4bit | 8192 | 4 | 358.1 | 350.5 | 350.7 | 306.9 to 365.6 | 333.8 to 364.4 | 2.1% | -1.9% | yes |
| qwen3.5-0.8b-4bit | 256 | 1 | 171.5 | 117.5 | 121.2 | 106.0 to 192.9 | 119.5 to 175.5 | 44.9% | +0.5% | yes |
| qwen3.5-0.8b-4bit | 256 | 4 | 332.4 | 296.3 | 325.1 | 168.7 to 473.5 | 316.5 to 425.1 | 10.9% | -1.7% | yes |
| qwen3.5-0.8b-4bit | 8192 | 1 | 123.8 | 112.9 | 112.5 | 106.1 to 175.6 | 105.8 to 152.0 | 9.2% | -5.2% | yes |
| qwen3.5-0.8b-4bit | 8192 | 4 | 319.7 | 286.5 | 275.3 | 270.7 to 360.4 | 194.3 to 353.8 | 11.5% | -4.6% | yes |

The Gemma 3 half ran on a quiet host: every branch median is inside the main range, and the fastest single sample of the 256-token `B = 1` cell is a branch one (197.5). Another job started on the host during the Qwen 3.5 half. The two main arms, which are one binary, differ there by 9% to 45%, so that half says nothing either way.

## Run 3: Qwen 3.5 at B = 1, three rounds, four samples per cell and round

Raw rows: [`run3-qwen-b1.json`](scheduler-owned-state-decode-gb10-2026-11-23/run3-qwen-b1.json).

| model | P | B | main-a | main-b | branch | main range | branch range | null | branch | in range |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b-4bit | 256 | 1 | 174.3 | 179.2 | 184.6 | 100.8 to 197.2 | 93.5 to 193.9 | 2.7% | +3.9% | yes |
| qwen3.5-0.8b-4bit | 8192 | 1 | 168.0 | 164.2 | 164.9 | 68.3 to 175.9 | 70.0 to 175.5 | 2.3% | -1.4% | yes |

Twelve samples per arm. The host was still disturbed for a few samples (the low ends of the ranges), and the medians are within the null. The three fastest samples are 193.0, 196.5 and 197.2 tok/s for main and 192.4, 193.4 and 193.9 for the branch at 256 tokens, and 172.6, 174.3 and 176.0 against 170.5, 171.5 and 175.5 at 8192.

## Run 4: Gemma 3 at 256 tokens, three rounds, four samples per cell and round

Raw rows: [`run4-gemma-256.json`](scheduler-owned-state-decode-gb10-2026-11-23/run4-gemma-256.json).

| model | P | B | main-a | main-b | branch | main range | branch range | null | branch | in range |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 1 | 184.8 | 182.2 | 183.3 | 171.1 to 186.5 | 176.2 to 186.3 | 1.4% | -0.1% | yes |
| gemma-3-1b-it-4bit | 256 | 4 | 314.5 | 322.4 | 308.2 | 188.9 to 332.3 | 193.7 to 330.6 | 2.5% | -4.1% | yes |

Twelve samples per arm, taken while the host carried more load than in runs 1 and 2 (every arm is about 5% slower at `B = 1` and 18% slower at `B = 4` than in run 2).

## Run 5: Gemma 3 at B = 4, quiet host, four rounds, three samples per cell and round

Raw rows: [`run5-gemma-b4.json`](scheduler-owned-state-decode-gb10-2026-11-23/run5-gemma-b4.json).

Gemma 3 1B 4-bit only, `P` of 256 and 8192 at `B = 4`, with `BENCH_CELLS=256x4,8192x4`. The branch binary was rebuilt at `4a879e9c`, the head of the branch after the review fixes (`cargo build --release --features cuda --bin mlxcel-server`); the main binary is the one from the earlier runs (`bd931f76`). The host load average was about 2.4 to 2.9 during the run, with no other job competing.

| model | P | B | main-a | main-b | branch | main range | branch range | null | branch | in range |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 4 | 366.21 | 363.91 | 388.52 | 254.41 to 390.13 | 361.87 to 398.38 | 0.6% | +6.4% | yes |
| gemma-3-1b-it-4bit | 8192 | 4 | 350.60 | 344.62 | 357.85 | 307.65 to 362.18 | 341.27 to 376.07 | 1.7% | +3.1% | yes |

Twelve samples per arm. The reply texts are identical across the three arms.

## Reading

At `B = 1` no cell shows a change that survives a repeat. Run 1's gaps of 4% to 6% came from three samples per arm. With twelve, Gemma 3 at 256 tokens is at -0.1% under a 1.4% null, and Qwen 3.5 is at +3.9% and -1.4% under 2.3% to 2.7%. Gemma 3 at 8192 tokens is at -0.8% and +2.1% in its two runs.

Qwen 3.5 at `B = 4` is at -0.1% and -0.3% in its one quiet run.

Gemma 3 at `B = 4` was below main in runs 1, 2 and 4 (-2.6%, -2.2% and -4.1% at 256 tokens, -3.4% and -1.9% at 8192). Run 5 repeats both cells on a quiet host with the branch rebuilt at its final commit, and the gap does not reproduce: the branch is at +6.4% (256) and +3.1% (8192), above both main arms, under nulls of 0.6% and 1.7%, and its median is inside the main range in both cells. The earlier negative sign therefore came from host load, not from the branch. The branch does the same device work per step as main, and no code difference was found.

The acceptance line of the issue, no decode regression, is met in every cell measured: `B = 1` and `B = 4`, both models, at 256 and 8192 tokens.

## Prompt-cache behaviour on real checkpoints

[`cache_compare.py`](scheduler-owned-state-decode-gb10-2026-11-23/cache_compare.py) runs each binary with the prompt cache on and `--ctx-checkpoints 4`, greedy: a three-turn chat (about 1,500, 2,500 and 3,400 prompt tokens), an edit of the second turn, the second turn again, then four concurrent fresh requests. It records every reply and `/v1/cache/stats` after each step ([`cache_compare.json`](scheduler-owned-state-decode-gb10-2026-11-23/cache_compare.json), branch at `365ee0b2`).

- Replies are identical between main and the branch at every step, for both models.
- `/v1/cache/stats` has the same fields on both, and the same values after every step of the chat, the edit and the replay (the only differing field is the wall-clock `last_reject_at_unix_ms`). That covers snapshot inserts, hits, supersedes, state-checkpoint inserts and hits, and the `cached_tokens` each reply reports.
- After the four concurrent requests, Qwen 3.5 showed one snapshot and one state checkpoint fewer on the branch. Repeating that step four times per binary gives the same split on both: one insert when a request is prefilled alone before the rest arrive, none when all four are prefilled as one cohort. It depends on request arrival, not on the binary.
