# Laguna DFlash verify round: where the fixed and per-row cost goes (GB10, 2026-09-11)

Issue #1799. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64, MLX pin `81ba1c6a`, CUDA release build (`MLX_CUDA_ARCHITECTURES=121`), toolchain 1.97.1. Source tree `2deae307` (the squash merge of PR #1771; its tree is byte-identical to the PR head `dab18fdf`, `git diff --stat dab18fdf 2deae307` is empty). Warm PTX cache and warm page cache (one discarded warm-up run per sweep). `MLX_ENABLE_TF32` left at MLX's default; no numerical comparison here depends on it (identity below compares token ids).

Pairing: `models/mlx/laguna-xs-2.1-nvfp4` (target: 40 layers, 10 full plus 30 sliding softmax attention layers with per-head sinks on sliding layers, head_dim 128, 48 query and 8 KV heads, 256 experts top-8 with `moe_intermediate_size` 512 plus a dense shared expert of 8192, everything NVFP4 through `gather_qmm` for the experts and `fp_qmv` / `qmm_sm80` for the dense projections) with `models/mlx/laguna-xs-2.1-dflash` (5-layer bf16 DFlash drafter, `block_size 16`, `target_layer_ids [1, 13, 25, 33, 39]`, 64 query heads with per-head gating).

## Host idleness

The #1771 sweep ran with the GPU exclusive but another session's `cargo build` at load 2.6 to 3.9. Every sweep here ran on an otherwise idle host: the sweep driver gates its start on a 1-minute load average under 0.6 in two samples 20 s apart (the baseline started at load 0.42), the 1-minute load average is recorded before every run, and `ps` during the runs showed the `mlxcel` process under test as the only consumer above 2% CPU (it runs at 130% during its own model load, which is what lifts the recorded load1 to 1.0 to 1.3 once the sweep is under way). The self-hosted CI runner was up but idle. GPU held under the scratchpad lock for every run. One transient spike to load1 2.37 was recorded before the block 12 round-2 run; that run's value (17.17 tok/s) sits inside its other two (17.08, 17.51) and is kept.

## Method

`mlxcel generate -m <target> -p <prompt> -n 200 --temp 0 [--draft-model <drafter> --draft-kind dflash --draft-block-size N]`, the offline arm the #1771 sweep used, `MLXCEL_MTP_ALLOW_INEXACT=1` on the speculative arms (the exactness probe declines this host; the gate itself is not touched by this work), `MLXCEL_PRINT_TOKEN_IDS=1` for identity. Prompt: the 152-token raw Python source header (`retry_with_backoff`) used by the #1782 record, no chat template. Every run reaches the 200-token budget. Same binary on every arm; the classic arm is the same command without `--draft-model`. Widths are interleaved round-robin with a rotating start (the `scripts/bench_block_width.sh` rationale) so drift spreads across the table. Rate is decode tok/s as the CLI reports it. The per-round split comes from the CLI's `DFlash:` line: `draft_ms` (host graph build of the drafter, including the `async_eval` enqueue), `verify_graph_ms` (host graph build of the target verify forward), `verify_sync_ms` (the wait for all of the round's device work), `decode_ms` (the whole round loop). Harness: `bench_cli.py` (scratchpad), n = 3 per configuration after one discarded warm-up.

## Baseline (idle host, tree `2deae307`)

| config | n | tok/s mean (min to max) | vs off | accepted/round | round wall ms | device sync ms/round (min to max) | draft host ms/round | verify host ms/round | load1 (min to max) |
|---|---|---|---|---|---|---|---|---|---|
| off (classic) | 3 | 29.28 (29.07 to 29.59) | | | 34.2 per token | | | | 0.58 to 1.30 |
| block 2 | 3 | 15.37 (15.05 to 15.93) | 0.52x | 0.77 | 115.3 | 80.4 (78.7 to 82.9) | 31.1 | 3.7 | 0.70 to 1.21 |
| block 4 | 3 | 20.57 (20.15 to 20.91) | 0.70x | 1.44 | 118.6 | 84.9 (83.8 to 86.3) | 30.1 | 3.6 | 1.01 to 1.14 |
| block 6 | 3 | 21.06 (20.62 to 21.30) | 0.72x | 1.67 | 126.7 | 92.5 (91.2 to 94.2) | 30.4 | 3.7 | 1.00 to 1.26 |
| block 8 | 3 | 19.04 (18.84 to 19.26) | 0.65x | 1.62 | 138.4 | 103.7 (102.2 to 104.8) | 30.7 | 3.9 | 1.05 to 1.30 |
| block 10 | 3 | 18.03 (17.82 to 18.18) | 0.62x | 1.62 | 145.9 | 111.1 (110.3 to 112.4) | 30.9 | 3.8 | 1.03 to 1.26 |
| block 12 | 3 | 17.25 (17.08 to 17.51) | 0.59x | 1.62 | 152.6 | 117.4 (115.8 to 118.7) | 31.1 | 4.0 | 1.16 to 2.37 |
| block 16 (checkpoint default) | 3 | 15.37 (15.28 to 15.43) | 0.52x | 1.55 | 166.9 | 131.3 (130.7 to 132.1) | 31.5 | 4.0 | 1.03 to 1.82 |

What the baseline says before any profile:

- The idle host does not rescue the pairing. The classic step is 34.2 ms; the round's device sync fits `71.6 ms + 3.79 ms per row` by ordinary least squares over all 21 speculative runs (the issue's `72 + 3.55` shape). Within-configuration spread is under 6% everywhere, so this is not the #755 bimodality.
- The round is a serial chain and the columns add up: at block 2, 31.1 (draft host) + 3.7 (verify host) + 80.4 (device sync) = 115.2 against a measured 115.3 ms round wall. At 0.77 accepted (1.77 emitted per round) that is 65 ms per emitted token, 1.9x the classic step.
- Acceptance on this prompt is lower than the #1771 sweep's (1.62 against 3.33 at block 8), and saturates at width 8, so the throughput ratios here are worse than that table's; the per-round cost columns, which are what this issue attributes, agree with it (80 to 131 ms against its 80 to 127).
- Token identity: the three classic runs are not identical to each other on this pairing (two of four classic runs, warm-up included, flip one late token, at positions 114 and 155). Every speculative width is deterministic across its own three runs and diverges from classic at a fixed early position (44 at widths 2, 4, 6 and 16; 15 at widths 8, 10 and 12, where the dense projections take `qmm_sm80`). This is the out-of-scope exactness question recorded in #1799 and is not used as a gate here; the new fact for that issue is that the classic arm itself is not run-to-run deterministic on this host.

## Zero-code controls: the graph-side candidates (idle host, n = 3 each, same binary)

Environment variables read by MLX, no code path difference. `mb400` is `MLX_MAX_MB_PER_BUFFER=400` (GB10's default is 25), `ops100` is `MLX_MAX_OPS_PER_BUFFER=100` (default 20), `both` is `MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=1000`, `nograph` is `MLX_USE_CUDA_GRAPHS=0`. Sweep ran 19:30 to 19:48, load1 0.82 to 1.75 with the process under test the only consumer.

| config | n | tok/s mean (min to max) | vs default off | accepted/round | round wall ms | device sync ms/round (min to max) | draft host ms/round | verify host ms/round |
|---|---|---|---|---|---|---|---|---|
| off (default budgets) | 3 | 29.28 (29.07 to 29.59) | | | 34.2 per token | | | |
| mb400, off | 3 | 31.70 (31.47 to 32.08) | 1.08x | | 31.5 per token | | | |
| ops100, off | 3 | 29.91 (29.77 to 30.15) | 1.02x | | 33.4 per token | | | |
| both, off | 3 | 33.94 (33.53 to 34.27) | 1.16x | | 29.5 per token | | | |
| nograph, off | 3 | 32.67 (32.56 to 32.83) | 1.12x | | 30.6 per token | | | |
| block 2 (default) | 3 | 15.37 (15.05 to 15.93) | 0.52x | 0.77 | 115.3 | 80.4 (78.7 to 82.9) | 31.1 | 3.7 |
| mb400, block 2 | 3 | 15.94 (15.73 to 16.33) | 0.54x | 0.77 | 111.1 | 78.1 (76.5 to 79.0) | 29.2 | 3.7 |
| ops100, block 2 | 3 | 15.63 (15.36 to 16.16) | 0.53x | 0.77 | 113.3 | 80.5 (77.8 to 82.0) | 28.8 | 3.9 |
| both, block 2 | 3 | 15.98 (15.68 to 16.48) | 0.55x | 0.77 | 110.8 | 78.2 (76.1 to 79.4) | 28.4 | 4.1 |
| block 8 (default) | 3 | 19.04 (18.84 to 19.26) | 0.65x | 1.62 | 138.3 | 103.7 (102.2 to 104.8) | 30.7 | 3.9 |
| mb400, block 8 | 3 | 19.47 (19.21 to 19.76) | 0.66x | 1.62 | 135.2 | 100.6 (99.0 to 102.4) | 30.7 | 3.8 |
| ops100, block 8 | 3 | 18.61 (18.54 to 18.72) | 0.64x | 1.62 | 141.4 | 106.2 (105.2 to 106.8) | 31.2 | 4.0 |
| both, block 8 | 3 | 19.17 (18.86 to 19.70) | 0.65x | 1.62 | 137.3 | 102.9 (100.1 to 104.5) | 30.1 | 4.3 |
| nograph, block 8 | 3 | 20.40 (19.96 to 20.63) | 0.70x | 1.62 | 129.1 | 95.7 (94.8 to 97.0) | 29.4 | 4.0 |
| block 16 (default) | 3 | 15.37 (15.28 to 15.43) | 0.52x | 1.55 | 166.9 | 131.3 (130.7 to 132.1) | 31.5 | 4.0 |
| mb400, block 16 | 3 | 15.84 (15.49 to 16.08) | 0.54x | 1.55 | 162.0 | 126.7 (123.8 to 130.0) | 31.2 | 3.9 |
| ops100, block 16 | 3 | 15.33 (15.25 to 15.44) | 0.52x | 1.55 | 167.3 | 132.2 (131.1 to 132.9) | 30.9 | 4.1 |
| both, block 16 | 3 | 15.29 (15.02 to 15.63) | 0.52x | 1.55 | 167.7 | 132.7 (130.3 to 135.1) | 30.7 | 4.3 |

What the controls say about the verify round:

- No graph-side knob owns the floor. The best any of them does to the 2-row device sync is 80.4 to 78.1 ms (`mb400` or `both`, about 3%), to the drafter host build 31.1 to 28.4 ms, and to the whole round 115.3 to 110.8 ms (4%). Turning capture off entirely leaves block 8 at 95.7 ms of device sync against 103.7 with graphs on, so the multi-row round gains nothing from capture, the same sign as on Qwen 3.5 in #1782. The 72 ms floor survives every one of them. Byte budget, op budget and capture itself are therefore ruled out as the mechanism, with these bounds.
- The byte-budget mechanism is real but small here. MLX commits a graph when `bytes_in_graph_` (which sums `data_size()`, an element count, over the graph's input arrays) exceeds the budget; each 256-expert nvfp4 stack is 33.5M packed elements and the lm_head 25.7M, so on the default 25 "MB" every `gather_qmm` (120 per token) and the lm_head commit their own graph on both arms. Raising it is worth 8% on the classic arm and about 3% on the verify round; it does not separate the arms.

## Separate finding for #1798: on this MoE pairing the default GB10 graph budgets make capture a net loss on the classic arm

Recorded here because the controls established it with repeats and it goes beyond #1799's question. Classic decode of `laguna-xs-2.1-nvfp4`, same binary, n = 3, ranges disjoint:

| classic arm | tok/s mean (min to max) | vs default |
|---|---|---|
| default (graphs on, 20 ops, 25 "MB") | 29.28 (29.07 to 29.59) | |
| `MLX_USE_CUDA_GRAPHS=0` | 32.67 (32.56 to 32.83) | +12% |
| `MLX_MAX_MB_PER_BUFFER=400` | 31.70 (31.47 to 32.08) | +8% |
| `MLX_MAX_OPS_PER_BUFFER=100` | 29.91 (29.77 to 30.15) | +2% |
| both raised (100 ops, 1000 "MB") | 33.94 (33.53 to 34.27) | +16% |

Two things follow. First, with GB10's default budgets, CUDA graph capture costs this model more than it saves: graphs off beats graphs on by 12%, and capture only pulls ahead of no capture (33.94 against 32.67) once both budgets are raised. Second, the two knobs interact: the op cap alone is worth 2% because the byte cap commits the graph long before 20 ops accumulate (every expert stack and the lm_head exceed 25 "MB" on their own), so the op cap only binds once the byte cap is lifted. This is a per-model-shape result, not a host default: on Qwen 3.5 in #1782 (`qwen3.5-4b-4bit`, affine 4-bit, no matrix over the byte cap) graphs off cost the classic arm 8% and `MLX_MAX_OPS_PER_BUFFER=100` gave +3%, the opposite sign on capture. Any change to the GB10 defaults in `hardware.rs` needs a measurement per model shape (dense against MoE, and matrix size against the byte cap), and it is not made here.
