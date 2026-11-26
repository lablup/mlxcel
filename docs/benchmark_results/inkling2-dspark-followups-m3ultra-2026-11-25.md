# Inkling 2 DSpark follow-ups (M3 Ultra, 2026-11-25)

Issue #3410, after #3280 made the Inkling 2 verify block byte-identical to decode (`inkling2-verify-exactness-m3ultra-2026-11-24.md`). Host: Apple M3 Ultra, 512 GB. Every run is serialized under the shared GPU lock. Generation runs use the release CLI with 256 greedy tokens, `--temp 0 --show-reasoning`, and the two prompts below, in A/B/C order repeated twice inside one lock. Load average 3.2 to 4.2 for the Small block.

- Prose: `Explain why the sky is blue.`
- Code: `Write a Python function that returns the n-th Fibonacci number, with a short docstring.`

Arms:

- A: classic decode (no `--draft-model`)
- B: DSpark with the acceptance-aware verify width (the new default)
- C: DSpark at the fixed requested width 8 (`MLXCEL_DFLASH_ADAPTIVE_WIDTH=0`)

## Small: where the verify time goes

`models::inkling2::verify_bench_tests` (one verify forward from a 32-token prompt, median of 30, Small loaded through `initialize_runtime` so the wired limit matches the CLI). The grouped column is the grouped expert matvec off and on, two alternating blocks each:

| Rows | gather path (ms) | grouped (ms) |
|---|---|---|
| 1 | 55.8 (decode step) | not used |
| 2 | 69.2, 69.7 | 66.6, 66.5 |
| 3 | 80.8, 81.5 | 75.6, 75.0 |
| 4 | 93.2, 94.1 | 88.1, 88.0 |
| 6 | 122.7, 120.0 | 108.5, 106.3 |
| 8 | 152.3, 149.5 | 130.7, 129.1 |

The argmax digests of all rows are identical with and without the grouped kernel. The inexact plain forward takes 103.6 ms at 8 rows against 129.8 ms for the exact one, so the row-invariant dense products cost about 26 ms of an 8-row round; the rest is the 384-expert MoE, where every extra row routes to about 8 more experts whose NVFP4 weights must be read. Each row added about 10 ms (grouped) to 13 ms (gather) against a 52 to 56 ms decode step, so an 8-row round pays only if it emits about 2.4 tokens; prose emits 2.3 at width 8.

## Small: throughput

| Prompt | A classic (tok/s) | B adaptive (tok/s) | C fixed 8 (tok/s) | B / A | C / A |
|---|---|---|---|---|---|
| prose | 19.32, 19.34 | 21.42, 21.72 | 16.61, 16.66 | 1.12x | 0.86x |
| code | 19.23, 19.23 | 30.62, 30.16 | 28.32, 28.29 | 1.58x | 1.47x |

Before this change (main at 5111fd06, fixed width 8, gather path) the same prompts measured 0.90x on prose and 1.54x on code (`inkling2-verify-exactness-m3ultra-2026-11-24.md`, a different day). In this block the fixed-width arm, which now also takes the grouped kernel, measured 0.86x and 1.47x; the classic arm was 19.3 tok/s both days, so the fixed-width rounds were slower here than in that run, by an amount this block does not explain.

Round mix of arm B (`RUST_LOG=mlxcel_core::drafter::dflash=debug`, from the run before the ABAB block, same build): prose ran 145 rounds, 33 of them classic steps, mostly 3-row verify rounds at 84 ms (`widths [1:33@47.5 2:10@73.3 3:89@83.6 4:9@102.3 8:2@136.6]`); code ran 79 rounds, 8 classic, spread over 3 to 8 rows. The ABAB runs reported 144 and 145 rounds (prose) and 79 and 82 (code).

Drafter-on output equals classic output byte for byte in every run (both prompts, both drafter arms, both repetitions).

## Small: exactness with the grouped kernel

`models::inkling2::exactness_diag_tests::verify_block_vs_chain_per_visit` with `INKLING2_DIAG_MODEL=models/inkling-2-small`: 48 visits, 0 differing logits at widths 2, 4 and 8 after the default prompt, and 0 at width 8 after a 1100-token prompt.

## Mini: no regression

Same ABAB on `models/inkling-2-mini` (load average 3.2 at the start, 17.8 at the end, so read the ratios loosely):

| Prompt | A classic | B adaptive | C fixed 8 |
|---|---|---|---|
| prose | 37.07, 34.03 | 44.97, 43.40 | 39.37, 37.61 |
| code | 33.07, 31.82 | 58.27, 59.19 | 57.58, 59.83 |

Text is identical across all arms. On code the policy holds wide blocks (8 classic steps, which is the bootstrap measurement); on prose it moves to narrower blocks and gains over the fixed width.

`models/inkling-2-mini-4bit` ships no DSpark head, so `--draft-model` loads it as a full-model drafter, which this change does not touch.

## Mini 4-bit: cost of the per-row attention projections

Affine 4-bit attention projections cannot take the row-invariant matvec, so inside a verify block they run row by row (`row_invariant::per_row`). `verify_bench_tests` on `models/inkling-2-mini-4bit`, load average 2.8:

| Rows | exact verify (ms) | plain, inexact (ms) | cost of exactness |
|---|---|---|---|
| 1 | 28.30 | 28.29 | 0 |
| 2 | 34.97 | 36.16 | none measured |
| 4 | 41.32 | 41.44 | none measured |
| 8 | 56.69 | 52.63 | 4.06 ms (7.7%) |

The plain forward also skips the per-row sliding-window attention, so 4.06 ms is an upper bound on what the per-row projections cost. The block stays exact: the per-visit diagnostic shows 0 differing logits at widths 2, 4 and 8, and at width 8 after a 1100-token prompt. A quantized row-invariant kernel would save at most that 4 ms per 8-row round, so the per-row path stays.

## Server prefill chunk

The server DFlash burst now prefills Inkling 2 in the `--prefill-chunk-size` partition (`DFlashTargetModel::prefill_burst` to `Inkling2Model::prefill_speculative_planned`) instead of one forward over the prompt. Test: `models::inkling2::speculative::tests::round_tests::the_burst_prefill_follows_the_server_chunk_size` (a 40-token prompt at chunk 16 equals the classic 16-chunk prefill bytes and differs from the one-forward prefill).
