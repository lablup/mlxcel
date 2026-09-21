# Draft block width after the #1939 prefill fix (GB10, 2026-09-21)

Issue #1797 seeded a measured default of 4 for `(12, 1, Affine)` on a DFlash drafter, from a sweep taken before PR #1939 fixed the DFlash burst's prompt prefill. That fix changes the KV and gated-delta state every round starts from, which changes acceptance, which changes throughput per width. This re-measures the curve on current main so the shipped default rests on the tree that ships it.

Nothing here is differenced against the #1797 table. That sweep ran on a different binary and a different tree; it is the prior conclusion to confirm or overturn, not a set of rows to subtract from. The reference is this session's own classic brackets.

**The ordering moved. Width 3 is faster than the shipped default of 4, by more than this session's drift.**

## Host

```
host: spark-101
kernel: 7.0.0-1019-nvidia
arch: aarch64
nvidia_driver: 580.178.04
gpu: NVIDIA GB10, 12.1
MLX_CUDA_ARCHITECTURES: 121
cuda_toolkit: 13.0
mlx_pin: 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a
rustc_used_for_build: rustc 1.97.1 (8bab26f4f 2026-07-14)
git_commit: 6f9a0982d7854b653c7b354fe0f1f0b26af41a7f
git_dirty: no
mem_total_gib: 121.7
binary_sha256: f2ba16883e8aed7aecf57212f5a81f4516eaeddcc618252bc5a6f464f015248c
```

The binary is built from a detached worktree at `origin/main`, not from the issue #1935 branch whose exactness gate declines this burst. That distinction is not theoretical: the first attempt at this sweep copied a stale artifact out of the shared target directory and measured the gated branch, which the harness caught and reported as `DECLINED TO CLASSIC` on the first width rather than as a number. The binary used here was verified to carry none of that branch's gate before it ran.

## Method

`sweep_server_widths.py` from the #1797 harness, unchanged, on `models/mlx/qwen3.5-4b-4bit` with `models/mlx/qwen3.5-4b-dflash`. Widths 2, 3, 4 and 6, n = 3 per arm after a discarded warm-up, with the harness's own classic arm first and last as the drift control. One server per arm, `--ignore-eos --max-batch-size 1`, the harness's fixed 158-token prompt, 200 tokens per request, `MLX_ENABLE_TF32=1`.

The host gate is the #1820 one the harness imports: a sustained-quiet CPU predicate matching on `/proc/<pid>/comm` with stopped processes dropped, a foreign-model check, a memory floor, and the cumulative `NV_ERR_NO_MEMORY` trip wire. It held this sweep for several minutes while CI ran on the same host and released it when the runner's worker finished; no arm was timed against a compiler or a CI job. The driver count was 0 before and 0 after, per arm.

## Result

Classic bracket: opening 56.93 to 57.11, closing 56.71 to 56.75 tok/s. They DO NOT OVERLAP: the session drifted, treat every middle arm as suspect.

| arm | n | e2e tok/s mean (min to max) | vs classic | separates from classic | acceptance | emitted per verify | round device sync ms | NVRM delta | resolved block_size |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| classic-open | 3 | 57.00 (56.93 to 57.11) | 1.00x |  |  |  |  | +0 | classic |
| w2 | 3 | 66.74 (66.56 to 67.09) | 1.17x | yes, above every classic run | 0.754 | 1.75 | 19.9 | +0 | 2 |
| w3 | 3 | 67.81 (67.76 to 67.89) | 1.19x | yes, above every classic run | 0.599 | 2.19 | 24.8 | +0 | 3 |
| w4 | 3 | 66.50 (65.71 to 66.96) | 1.17x | yes, above every classic run | 0.511 | 2.52 | 30.3 | +0 | 4 |
| w6 | 3 | 56.23 (56.01 to 56.47) | 0.99x | yes, below every classic run | 0.432 | 3.16 | 47.4 | +0 | 6 |
| classic-close | 3 | 56.73 (56.71 to 56.75) | 1.00x |  |  |  |  | +0 | classic |

## Reading

**The brackets do not overlap, and the finding survives it anyway.** The opening classic arm ran at 56.93 to 57.11 and the closing one at 56.71 to 56.75, so the session drifted downward by about 0.3 tok/s, half a percent. That is the resolution floor for everything between them, and the harness says so rather than printing a clean table over a dirty run. Width 3's slowest run (67.76) is 0.80 tok/s above width 4's fastest (66.96), which is twice the full bracket spread, and width 4 ran after width 3, so the drift works against width 3 rather than for it. Correcting width 4 upward by the entire drift still leaves the two ranges disjoint.

**Width 3 is the best of the four, width 2 and width 4 do not separate from each other, and width 6 is a loss.** Width 3 at 1.19x classic separates from both its neighbours: its range clears width 2's fastest run (67.09) and width 4's (66.96). Width 2 and width 4 overlap (66.56 to 67.09 against 65.71 to 66.96) and this sweep does not order them. Width 6 at 0.99x is below every classic run: the verify block has stopped paying for itself there.

**Acceptance is what moves, and it moves monotonically while the block widens.** 0.754 at width 2, 0.599 at 3, 0.511 at 4, 0.432 at 6, against emitted-per-verify of 1.75, 2.19, 2.52 and 3.16 and a per-round device-sync cost of 19.9, 24.8, 30.3 and 47.4 ms. The product of those is what a width is worth, and on this tree it peaks at 3 rather than at 4. The shipped default sits one step past the peak.

**Every speculative arm's greedy text differs from classic**, at every width including 2 and 3, which is issue #1935 reproduced independently here: this is a `main` binary with no gate, and the harness's own identity check reports it.

```
Greedy identity: the two classic arms are byte-identical to each other.
Greedy identity: classic-close == classic
Greedy identity: classic-open == classic
Greedy identity: w2 != classic
Greedy identity: w3 != classic
Greedy identity: w4 != classic
Greedy identity: w6 != classic
```

## What follows

The measured default for `(12, 1, Affine)` on a DFlash drafter should be re-examined against 3. This record does not change it: the sweep covers one pairing on one host, and on current main the issue #1935 gate is about to decline this pairing on CUDA anyway, which makes the width academic for it until that is resolved. The finding is filed so the number rests on evidence rather than on a pre-#1939 measurement.

Data: `data/draft-block-width-post-1939-gb10-2026-09-21/`.
