# Bucketing the cuDNN SDPA plan-cache key: the measurement that decides against it (GB10, 2026-09-17)

Issue #1820. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64, MLX pin `81ba1c6a`, CUDA release build (`MLX_CUDA_ARCHITECTURES=121`), toolchain 1.97.1. Base tree `0ef0a1a4` (main, which carries PR #1817, the #1799 fix). Pairing: `models/mlx/laguna-xs-2.1-nvfp4` with `models/mlx/laguna-xs-2.1-dflash`. Harness: `data/sdpa-plan-bucket-gb10-2026-09-12/harness/`.

This supersedes `sdpa-plan-cache-bucket-gb10-2026-09-12.md`, which was measured on the pre-rebase tree `60341873`. That record's mechanism findings still hold and are not repeated here; its throughput table does not, and is superseded by the one below.

## Recommendation: do not ship bucketing, keep #1817's fallback alone

Bucketing is mechanically correct and does what the issue asked for. It is still not worth shipping.

Against the standing bar that a default-on change must not lose on any measured workload, bucketing fails on the workload it was designed for. At a 152-token prompt it is **behind** #1817's ops fallback at three of five block widths with disjoint ranges, and at 2634 tokens it is a wash. There is no measured context length at which it wins. A measured loss disqualifies it as a default regardless of what the unmeasured long-context rungs would have shown, so the recommendation does not depend on the rung that could not be run.

The change is left on the branch behind `MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES`, defaulting to off-by-composition rather than removed, because the open question below is answerable and the implementation is the expensive part.

## What bucketing achieves, and it is not nothing

Measured on the rebased binary with `MLXCEL_SDPA_PLAN_DEBUG=1`, block 4, 60 tokens, the 152-token prompt:

| | exact-shape cuDNN | bucketed |
|---|---|---|
| plan builds in the process | 82 | 12 |
| plan builds per verify round after warm-up | 3 (one per shape class) | 0 |

Three shape classes, three builds per round, becoming one build per class for the whole generation. The target's two classes take the free unslice arm (cache extents 256 and 768, `copy=0` in the trace); the drafter takes the copy arm, because it concatenates its proposal keys onto the cache window and so has no room past the live length.

Greedy output is byte-identical, compared as token ids and not as text, on the rebased tree:

| arm | binary | ids |
|---|---|---|
| exact-shape cuDNN | base `0ef0a1a4` | `7d044d8534201cb4` |
| exact-shape cuDNN | branch | `7d044d8534201cb4` |
| bucketed cuDNN (branch default) | branch | `7d044d8534201cb4` (3 repeats) |
| ops fallback (main's default, #1817) | base `0ef0a1a4` | `baa2c6b55d7f874d` |
| ops fallback via kill switch | branch | `baa2c6b55d7f874d` |

Both shas reproduce the pre-rebase record exactly. The fourth and fifth rows differ from the first three because #1817 moved these calls off cuDNN's flash kernel, which #1799 recorded as shifting the greedy path at ties; that is a dispatch difference, not a padding error, and there is no third answer.

Non-speculative control, `qwen3-1.7b-4bit` (dense, head_dim 128, cuDNN-eligible): 200 ids identical base against branch, stable across two repeats. With the debug trace on a 2634-token prompt, 1204 cuDNN SDPA calls, **zero bucketed**, three plan builds in the process, and no decline warning. The prefill chunks are 2048 and 586 query rows, both far above the 32-row bound, and the decode calls take MLX's own one-row canonicalization. A non-speculative generation does not enter the new path at all.

## Throughput: the measurement that decides it

Same binary on every arm, `fb-` sets `MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0` which is exactly main's dispatch. Widths interleaved round-robin with a rotating start, n = 3, one discarded warm-up. 152-token prompt, 200 tokens. Data: `data/sdpa-plan-bucket-gb10-2026-09-12/laguna_cli_rebased.jsonl`.

| width | bucketed, tok/s (min to max) | #1817 fallback, tok/s (min to max) | ratio | ranges | bucketed draft host ms/round | fallback draft host ms/round |
|---|---|---|---|---|---|---|
| classic | 33.33 (32.78 to 34.19) | 31.68 (28.60 to 33.56) | | | | |
| 2 | 33.10 (32.43 to 33.77) | 35.12 (34.65 to 35.79) | **0.943x** | disjoint | 10.5 | 7.6 |
| 4 | 37.82 (37.40 to 38.29) | 36.64 (32.88 to 38.68) | 1.032x | overlapping | 13.4 | 9.2 |
| 6 | 35.76 (34.48 to 36.80) | 37.54 (36.70 to 38.21) | 0.953x | overlapping | 14.6 | 10.9 |
| 8 | 31.15 (30.64 to 31.51) | 32.68 (32.36 to 33.04) | **0.953x** | disjoint | 15.6 | 10.6 |
| 16 | 22.21 (21.89 to 22.43) | 24.26 (23.91 to 24.45) | **0.915x** | disjoint | 16.2 | 10.3 |

Three of five widths are losses with non-overlapping ranges. That is the finding.

### The 2634-token rung

n = 3 per arm, 150 tokens, widths 4 and 8. Data: `laguna_ladder_rebased.jsonl`.

| width | bucketed tok/s | fallback tok/s | tok/s ratio | verify ms/round ratio | verdict |
|---|---|---|---|---|---|
| 4 | 33.66 (30.29 to 35.53) | 33.08 (30.92 to 34.50) | 1.017x | 1.052x | overlapping |
| 8 | 29.69 (29.35 to 30.12) | 29.37 (29.26 to 29.50) | 1.011x | 1.007x | overlapping |

A wash. Two metrics are reported because only one is a kernel measurement: the arms run different kernels, generate different text and need a different number of verify rounds to reach 150 tokens (64 against 66 at width 4), so throughput mixes kernel speed with acceptance luck. `verify ms/round` divides that out.

### An unattributed term

The fallback arm's drafter host build is flat against the pre-rebase run (7.4 to 7.6, 9.1 to 9.2, 10.1 to 10.3 ms per round at widths 2, 4 and 16), while the bucketed arm's rose about 3 ms at every width (7.3 to 10.5, 9.4 to 13.4, 13.0 to 16.2). That term is host kernel enqueues, which is what the widening adds and what CPU contention hits hardest, and this session ran at load1 1.00 to 3.14 against 0.72 to 2.29 before, with peer builds live throughout. Contention and a real regression are not separable from this run. It is recorded as unattributed. The retry that would settle it was not run, because the decision tree made it conditional on a long-context win that never materialised.

## What is still open, and what would settle it

The crossover question is **unmeasured on this host in its current state, not absent**. The mechanism that motivates it is real and worth recording: bucketing's cost is flat in context, because only 10 of the pairing's 45 attention calls scale with prompt length (the target's full-attention layers, which take the free unslice arm) while its 30 sliding layers and all 5 drafter layers are capped at a 512 window, whereas the ops fallback materializes a `[B, heads, q_len, k_len]` score matrix that grows linearly. A crossover should therefore exist somewhere above 2634 tokens.

Finding it would not produce an unconditional ship. It would produce a context-gated one, which then needs the short-context retry above to confirm the loss is real before a gate could be designed. The remaining question is larger than one rung.

The pre-rebase record's 16k rows (n = 1 to 2, `laguna_ladder_cli.jsonl`) point at a 10 to 14% per-round win for bucketing. They are too underpowered to rely on, which is exactly why re-running them was the task.

## Driver finding: a 16k run costs about 38 NVRM allocation failures on this host

Reported as a result rather than an incident, because it is a property of the host worth knowing independently of #1820.

| | value |
|---|---|
| cumulative `NVRM: NV_ERR_NO_MEMORY` before the session | 2, stable across 3 days of uptime |
| per-run cost at a 2634-token prompt | near zero |
| per-run cost at a 16k prompt | **about 38** (measured 184 to 222 across one run) |
| projected cost of the 13-run 16k rung | about 678 cumulative |
| budget the rung had to fit inside | 400 |

The step change with prompt length is in kind, not a noisy tail, and the runs themselves complete successfully with valid throughput. That is not evidence the failures are benign: it is the shape the 2026-07-06 hard freezes took on this host, where cumulative accumulation under spiky delivery preceded a wedged kernel while individual runs looked fine. The rung was stopped after two runs with the count at 222.

**Follow-up should require a fresh boot.** A rebooted host starts at a clean driver count, which is the only condition under which a 13-run 16k rung fits inside a 400 budget. Not filed here.

### `peak_rss_kib` is not a valid footprint proxy on GB10

The harness samples `VmHWM` per run. It reports **1.9 GiB** for a run whose weights alone are 20.97 GiB, because CUDA unified allocations on this host do not appear in the process's resident set. The field is recorded but must not be trusted as a footprint, and the harness's 32 GiB memory floor stays **derived** (20.97 GiB of weights measured on disk plus 0.69 GiB of KV computed from `config.json`, plus headroom) rather than measured. Marked broken rather than left as a plausible number.

## Blast radius

**Measured**: the Laguna DFlash pairing, and `qwen3-1.7b-4bit` as a dense head_dim-128 non-speculative control where the trace shows the path is not entered.

**Shares the path, unmeasured**: every other CUDA model whose array-masked SDPA with 2 to 32 query rows over a longer key sequence reaches this file on Ampere or later. In tree that is the speculative verify of any head_dim-128-or-less pairing (Muse Glimmer DFlash, LFM2 and LFM2.5 DSpark, Inkling MTP, Qwen 3.5's own DFlash drafter), a short incremental prefill over a reused prefix-cache prefix, and the trailing chunk of a chunked prefill when it lands under 32 rows.

**Untouched**: the one-row decode step, the backward primitive, Metal, ROCm and CPU. Attention sinks are not exercised: this checkpoint carries none. The `metal,accelerate` gate is not runnable on this Linux/CUDA host and was not run.

## Plan-cache growth bound

Shape classes x query widths x buckets crossed, not rounds. A class builds one plan per distinct `(q_len, bucket)`, so a generation of `N` new keys crosses `ceil(N / 256)` buckets per class. Measured: 12 plans in a 25-round block-4 generation against 82 exact-shape. Against the `2 * MLX_CUDA_SDPA_CACHE_SIZE` lifetime-miss abort (4000 at mlxcel's CUDA default of 2000), exact-shape cuDNN reaches it in about 1300 rounds and at MLX's own default of 256 in about 170, which is the abort #1799 hit; bucketed, the same budget covers roughly 340k new key positions per shape class. Note that #1817 also removes that abort for these shapes, by routing them off cuDNN entirely, so this is not a unique advantage of bucketing.
