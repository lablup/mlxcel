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

## Attribution: kernel tables per round and the host side of the round

`nsys profile -t cuda,nvtx --cuda-graph-trace=node` (rule 4 of the Volta record). Because `mlxcel generate` also profiles the model load, the prefill and the exactness probe (57 forwards at width 16), each configuration was profiled twice, at 400 and at 200 tokens, and the two were differenced: load, prefill and probe are identical in both, so the delta is exactly the decode work of the extra tokens, normalized by the extra rounds from the runs' own `DFlash:` lines (87 rounds at block 8, 83 at block 16, 200 tokens on the classic arm). Profiling inflated these runs by 3 to 4% only (classic 28.4 and 28.8 tok/s profiled against 29.28; block 8 18.5 against 19.0), which is far less than #1782's 15 to 46% because this pairing's kernels are large. Block 2 could not be differenced this way at first: its 400-token profile (and later a 300-token one) aborted, see below.

### GPU kernel time per round, by category

| category (kernel names) | classic, per token | block 8, per round | block 16, per round |
|---|---|---|---|
| bf16 dense projections and lm_head: `gemv_single<bf16>` at one row, `cutlass_80_wmma_tensorop_bf16_s161616gemm_16x16_128x2` at 2 or more rows (q, k, v, o, g, router, lm_head; also the drafter's bf16 matmuls) | 21.16 ms, 243 launches | 33.37 ms, 318 | 33.06 ms, 353 |
| routed experts, `qmm_sm80_kernel` via `gather_qmm` (`M = 1, B = 8 x rows`, so this family at every width including classic); at 8 rows and above the NVFP4 shared expert joins it | 6.14 ms, 117 | 39.37 ms, 233 | 59.35 ms, 234 |
| shared expert, `fp_qmv_single` (NVFP4, `M < 8`) | 0.90 ms, 117 | 0.02 ms, 1 | 0 |
| attention kernels (`kernel_sdpav_1pass` on classic; `cudnn_generated_fort_native_sdpa_sm80_flash_fprop` on the blocks) | 0.66 ms, 42 | 1.09 ms, 45 | 1.16 ms, 45 |
| elementwise, copies, gather, sort, norms, rope, reductions, arange | 5.28 ms, 1844 | 12.84 ms, 2666 | 14.19 ms, 2669 |
| **total** | **34.14 ms, 2363 launches** | **86.69 ms, 3262 launches** | **107.70 ms, 3301 launches** |
| wall per token or per round under nsys (from the runs' own timers) | 35.6 ms | 142.6 ms (device sync 112.7, drafter host build 26.0, verify host build 3.7) | 170.4 ms (136.8, 29.8, 3.7) |
| GPU busy fraction | 96% | 61% | 63% |

Two corrections to the issue's premises come out of the kernel names. This checkpoint's attention projections, per-head gate, router and lm_head are plain bf16 (`model.layers.*.self_attn.{q,k,v,o}_proj.weight` are BF16 in the safetensors headers; only the routed and shared experts are NVFP4), so the `fp_qmv` to `qmm_sm80` "family switch on the dense projections" does not exist on this pairing: the dense projections go from `gemv_single` at one row to a cutlass bf16 GEMM at two or more, which is what the `dense` row above shows. And `SwitchGLU::forward` expands `x` to `[n, 1, 1, K]`, so `gather_qmm` sees `M = 1, B = 8n` and the routed experts take `qmm_sm80` at every width including classic; the switch at 8 rows only moves the NVFP4 shared expert (117 `fp_qmv` launches per token) into the same family.

The per-row term is GPU work in the routed experts: `qmm_sm80` goes from 39.4 ms at 8 rows to 59.4 at 16, 2.5 ms per row, against a bandwidth floor of about 1.8 ms per row for reading each selected expert once per (row, expert) pair (8 experts x 3 matrices x 0.5 MB x 40 layers). It is inherent to a MoE verify: rows route to different experts, so each row adds its own expert reads. The bf16 dense term is flat from 8 to 16 rows (33.4 against 33.1 ms) and is a fixed 12 ms over the classic step's 21 ms, the price of the cutlass 16x16-tile GEMM over the one-row gemv; the drafter's own bf16 matmuls are inside it.

### The fixed floor is host time in `ScaledDotProductAttention::eval_gpu`

The kernel time per round (86.7 ms at block 8) is far below the round (142.6 ms) and even below the device sync alone (112.7 ms), so the floor is not on the GPU. MLX's NVTX ranges, differenced the same way, put it in one primitive:

| NVTX host range | classic, per token | block 8, per round | block 16, per round |
|---|---|---|---|
| `ScaledDotProductAttention::eval_gpu` | 2.82 ms (40 instances) | **67.09 ms (45)** | **75.83 ms (45)** |
| `CommandEncoder::commit` | 9.91 ms (207) | 21.59 ms (294) | 24.29 ms (310) |
| `Matmul::eval_gpu` | 0.60 ms (243) | 4.69 ms (280) | 5.37 ms (280) |
| `cu::CudaEvent::wait` | 0 | 4.57 ms | 5.58 ms |
| every other primitive | under 1 ms each | under 2 ms each | under 2 ms each |

The 45 instances are the target's 40 attention layers plus the drafter's 5. On the classic arm a one-row call takes the `sdpa_vector` kernel and costs 70 us of host time. On a multi-row masked call with head_dim 128 and bf16 the CUDA backend selects cuDNN (`supports_sdpa_cudnn`: Ampere or later, head_dim at most 128, f16 or bf16; `supports_sdpa_vector` refuses any array mask and any `q_len >= 4`), and MLX caches the cuDNN execution plan in an LRU keyed on the exact q, k, v and mask shapes and strides (`build_sdpa_cache_key`, `scaled_dot_product_attention.cpp:142-176`). A verify round appends rows to the KV cache, so `k_len` and the mask shape are new every round for each of the three shape classes (target full layers, target sliding layers with sinks, drafter layers): three cache misses per round, each running `build_sdpa_graph` (cuDNN frontend validate, build_operation_graph, create_execution_plans, check_support, build_plans), about 22 ms each on this host. That is the fixed term: 67 ms at 8 rows, 76 at 16, and it does not depend on the row count. It also explains why every graph-side control left the floor alone (the plan build is not graph work) and why the drafter's host build was 30 ms at every width: its five layers pay one of the three builds, and it fell to 9 to 11 ms once cuDNN was taken out of the round (control below).

The same LRU (`lru_cache.h`) keeps a lifetime miss counter and throws `Cache thrashing is happening, please set the environment variable MLX_CUDA_SDPA_CACHE_SIZE to a larger value than 256` once it passes twice the capacity, 512. At three misses per round that is about 170 rounds per process: the block 2 profile at 400 tokens (226 rounds) aborted on exactly that throw, twice, and so did a 300-token retry (about 170 rounds), while the 200-token runs (113 rounds) and the block 8 and 16 runs at 400 tokens (163 and 161 rounds) completed. Before this issue, a Laguna DFlash generation longer than about 170 rounds ended the process. `nsys -t cudnn` was also tried to count the builds directly; this nsys (2025.3.2) captured no cuDNN events from the MLX binary, so the count rests on the NVTX ranges and the abort arithmetic.

Graph replay itself is healthy on every arm, as on Qwen 3.5: `cudaGraphInstantiate` 0.1 per token on classic against 199.9 `cudaGraphExecUpdate`, and 0.2 per round against 259.4 at block 8 and 259.8 at block 16. The instantiate-to-update ratio is under 1:1000 on both verify widths, so re-instantiation is ruled out.

### Control: `MLX_CUDA_USE_CUDNN_SDPA=0` (zero code, n = 3)

| config | tok/s mean (min to max) | vs default off | accepted/round | round wall ms | device sync ms/round (min to max) | draft host ms/round | verify host ms/round |
|---|---|---|---|---|---|---|---|
| cuDNN off, classic | 29.49 (28.98 to 30.19) | 1.01x | | 33.9 per token | | | |
| cuDNN off, block 2 | 32.24 (31.44 to 32.95) | **1.10x** (was 0.52x) | 0.74 | 54.0 (was 115.3) | 41.3 (40.5 to 42.7) (was 80.4) | 8.8 (was 31.1) | 3.8 |
| cuDNN off, block 4 | 39.15 (38.68 to 39.73) | **1.34x** (was 0.70x) | 1.47 | 63.1 | 48.7 (48.3 to 49.1) | 10.3 | 4.1 |
| cuDNN off, block 8 | 33.58 (33.13 to 33.90) | **1.15x** (was 0.65x) | 1.67 | 79.4 | 64.6 (63.9 to 65.9) | 10.7 | 4.0 |
| cuDNN off, block 16 | 25.27 (25.09 to 25.52) | 0.86x (was 0.52x) | 1.69 | 106.9 | 91.4 (89.9 to 92.6) | 11.1 | 4.3 |

The classic arm is unchanged (it never enters cuDNN at decode) and the verify round loses about 39 ms of device sync and 21 ms of drafter host build at every width: the two links of the serial chain that held a plan build each. Caveat on this one sweep: the host was not clean for it. My own `nsys stats` exports (CPU-bound sqlite conversions) overlapped it, and load1 reached 3.2 before some runs; the direction and size of the move are not in doubt (every range is disjoint from the baseline by a wide margin), but the after-fix numbers that follow, taken on a quiet host with the same binary and a code-level kill switch, are the ones to quote. The acceptance per round also moves slightly (0.74 against 0.77 at block 2, 1.69 against 1.55 at 16) because the fallback's arithmetic differs from cuDNN's and the greedy path shifts at a tie; that is the numerics side of the exactness question, out of scope here.

Process-wide `MLX_CUDA_USE_CUDNN_SDPA=0` is not the fix: it also takes prefill off cuDNN, which is where cuDNN's flash kernels earn their keep on long prompts (mlxcel's chunked materializing path exists for that case but is slower). The fix routes only the verify shape away from cuDNN, see the next section.
