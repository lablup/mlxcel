# bf16 quantized matmul route on ROCm: Radeon 8060S (gfx1151), 2026-09-30

Why `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible` failed for bf16 on gfx1151, what was measured to choose the fix, and what the fix did to prefill (issue #2081, part of #1801). The fix is `patches-rocm/LOCAL_FIXES.md` item 29 plus a route check in `prefill_dense_gemm_eligible`.

## Environment

Same host as the [decode baseline](rocm-baseline-gfx1151-2026-09-30.md): AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), ROCm 10.0.0 with HIP 7.15.26333, Debian 13. mlxcel at `7278397a` (main) plus this change, MLX pin `81ba1c6a`, overlay NripeshN/mlx `rocm-support` at `75915908` plus `LOCAL_FIXES.md` items 1 to 29. `--features rocm`. Every measured run waited until `/sys/class/kfd/kfd/proc` had been empty for 30 s, and a sampler checked it once per second during the run; one op-level run that another GPU process overlapped was discarded. Compiler activity was not logged.

## The mismatch

bf16, no bias, x `[1, 1024, 2048]`, 4-bit g64 affine weight `[1024, 2048]`: 499 of 1,048,576 outputs of `dequantize` + `matmul` differed from `quantized_matmul` (465 by 1 ULP, 10 by 2, 24 by more, all on outputs below 0.01 in magnitude). `quantized_matmul` ran `qmm_wmma_dense_kernel`, which dequantizes with the same rounding as `affine_dequantize` but accumulates through rocWMMA 16 x 16 x 16 tiles in its own K order; the dense side ran hipBLASLt. f16 has no WMMA route, so its `quantized_matmul` also dequantizes and calls hipBLASLt, and it matched.

## Op level: WMMA kernel against dequantize + hipBLASLt

bf16, 4-bit g64, one `[1, M, K]` input against an `[N, K]` weight. `qmm` is `quantized_matmul` on main (always the WMMA kernel for these shapes); `dense` is `dequantize` then `matmul`, which re-dequantizes on every call. Mean of 40 calls per arm in two alternated rounds; `differ` counts outputs whose bytes differ.

| M | K x N | qmm ms | dense ms | dense / qmm | differ |
|---:|---|---:|---:|---:|---:|
| 16 | 4096 x 4096 | 0.388 | 0.337 | 0.87 | 79 / 65,536 |
| 16 | 4096 x 14336 | 1.329 | 1.649 | 1.24 | 261 / 229,376 |
| 64 | 4096 x 4096 | 0.413 | 0.488 | 1.18 | 305 / 262,144 |
| 64 | 14336 x 4096 | 1.472 | 1.995 | 1.36 | 1,217 / 262,144 |
| 128 | 4096 x 4096 | 0.805 | 0.570 | 0.71 | 572 / 524,288 |
| 128 | 14336 x 4096 | 2.519 | 2.479 | 0.98 | 2,309 / 524,288 |
| 256 | 4096 x 14336 | 4.022 | 2.274 | 0.57 | 4,153 / 3,670,016 |
| 512 | 4096 x 14336 | 9.187 | 3.769 | 0.41 | 4,242 / 7,340,032 |
| 1024 | 4096 x 1024 | 1.403 | 0.320 | 0.23 | 1,164 / 1,048,576 |
| 1024 | 4096 x 14336 | 15.775 | 4.644 | 0.29 | 12,251 / 14,680,064 |
| 2048 | 14336 x 4096 | 51.491 | 10.046 | 0.20 | 29,524 / 8,388,608 |

Across the 50 cells of the sweep (M 2 to 2048; K x N of 4096 x 4096, 4096 x 1024, 4096 x 14336, 14336 x 4096, 1024 x 3072) the WMMA kernel won only on some shapes at 64 rows or fewer, and dense was faster on every shape from 128 rows up. So byte identity and speed pointed the same way: route large bf16 GEMMs to dequantize + hipBLASLt. With the change, `quantized_matmul` differs from dense in 0 outputs of every cell from 128 rows up, and below 128 rows it is unchanged.

## Model prefill

`mlxcel-bench-decode --prompt-tokens {512, 2048} -n 8 --warmup-tokens 4 --ignore-eos`, one binary, the default against `MLX_ROCM_WMMA_QMM=1` (which removes the ceiling, the old dispatch on this device), three ABBA runs per cell. Prefill tok/s, mean (range):

| Model | Scales | pp512 before | pp512 after | pp2048 before | pp2048 after |
|---|---|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | 1146 (1046-1199) | 2289 (2197-2359) | 961 (948-972) | 2815 (2805-2824) |
| Qwen3-0.6B-4bit | bf16 | 4581 (4230-5024) | 7633 (7147-8012) | 3059 (2894-3219) | 4236 (4027-4566) |
| Qwen3-30B-A3B-4bit | bf16 | 311 (305-315) | 326 (321-334) | 283 (282-284) | 297 (296-299) |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | 969 (879-1032) | 1008 (994-1026) | 1149 (1143-1153) | 1132 (1124-1137) |

Decode is unchanged in every cell (M is 1, which never took the WMMA kernel). MLX peak memory is within 0.1 GB of before except Qwen3-0.6B at 512 tokens, 1.04 to 1.46 GB, because the dequantize route allocates a bf16 copy of each weight matrix, which `LOCAL_FIXES.md` item 28 bounds. Qwen3-30B-A3B gains least because only its attention projections take this path; its experts go through `gather_qmm`. Llama 3.1 8B is the control: its scales are f16, so no GEMM of it reaches the changed code, and its -1.5% at 2048 tokens has no path to explain it. No dense bf16-scale checkpoint above 4B was available on the host, so peak memory for one is not measured here; the dequantized-weight cache holds 8 matrices or 256 MB, so every model measured here dequantizes each projection again per prefill chunk, and the transient copies grow with the projection size.

## The dense prefill path on ROCm

`MLXCEL_PREFILL_DEQUANT_MIN_M=1024` against unset, same binary, pp2048, three ABBA runs: gemma-3-4b-it-4bit 2871 against 2852 tok/s, Qwen3-0.6B-4bit 4174 against 4185. With the route change the dense path is eligible on ROCm only where `quantized_matmul` already runs the same GEMM, so turning it on changes neither bytes nor speed there.

## 2026-10-07: the dequantized-weight cache above the WMMA ceiling (issue #2151)

The route above keeps every dequantized weight in an LRU of 8 matrices or 256 MB (`MLX_ROCM_QMM_DEQUANT_CACHE_SIZE`, `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`). A forward pass runs more distinct projections than that, so the cache never hits, and its last entries stay alive into decode. GEMMs routed above the ceiling (`QmmRoute::DequantGemmAboveWmmaCeiling`) now dequantize into a temporary and leave the cache alone; f16 and the other shapes on the same GEMM keep it. `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1` prints the cache counters at exit.

Same host. Before is main at `87538835`; after is that commit plus the change. The before and after medians were taken on 2026-10-07, except the third Gemma before run, which was taken on 2026-10-08 (the first failed in the harness, not the bench). `scripts/bench_decode.sh <model> --prompt-tokens 2048 --max-tokens 128` (20-token same-process warmup, then the measured pass), each run in its own `scripts/rocm_gpu_guard.sh --idle-secs 60` with before and after alternated, every run CLEAN. Medians of three runs (range). "Active after prefill" is the `[Memory] after warmup (prefill + warmup tokens)` line, read once the warmup prefill and its decode tokens are done; "active at end" is the `after measured pass` line.

| Model | Scales | Arm | Prefill tok/s | Decode tok/s | MLX peak | Active after prefill | Active at end |
|---|---|---|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | before | 2872 (2819-2910) | 62.75 (62.68-62.76) | 4.52 GB | 2.80 GB | 2.97 GB |
| gemma-3-4b-it-4bit | bf16 | after | 2868 (2839-2945) | 62.71 (62.55-62.73) | 4.52 GB | 2.56 GB | 2.73 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | before | 1136 (1106-1136) | 36.63 (32.36-36.84) | 6.92 GB | 4.75 GB | 4.75 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | after | 1132 (1048-1143) | 36.67 (36.58-36.92) | 6.92 GB | 4.75 GB | 4.75 GB |

Cache counters for the same run (warmup and measured prefill together). The before counts come from a build of the counters alone, which routes and caches exactly as `87538835` does; one run each, since the counts are deterministic and matched across the three after runs.

| Model | Arm | Hits | Misses | Inserts | Evictions | Bypassed | Entries at exit | Bytes at exit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | before (counters build) | 0 | 340 | 340 | 334 | 0 | 6 | 241,172,480 |
| gemma-3-4b-it-4bit | after | 0 | 0 | 0 | 0 | 340 | 0 | 0 |
| Meta-Llama-3.1-8B-Instruct-4bit | before (counters build) | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |
| Meta-Llama-3.1-8B-Instruct-4bit | after | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |

For Gemma 3 4B the cache held 230 MiB of bf16 weights after prefill and never hit; skipping it lowers active memory after prefill and at the end of the run by 0.24 GB. MLX peak memory does not move, because the peak comes from the per-call dequantized copies inside prefill, which the cache never removed. Prefill and decode differ by less than the run-to-run range. Llama 3.1 8B is the f16 control and reads the same in both arms: its GEMMs never take the ceiling route, and its cache also gets zero hits while holding 224 MiB after prefill; turning its default off is lablup/mlxcel#2232. Decode at batch 1 runs one row, which takes qmv and never reaches the cache.

Logits: `examples/logit_trace` on gemma-3-4b-it-4bit (README.md as the corpus, 4 chunks of 256 tokens, so every projection runs at 256 rows and takes the ceiling route), before and after, gives byte-identical traces (sha256 `e470fd1c...`), and `scripts/compare_logit_traces.py --decided 2.0` reports 0 of 1024 disagreements and an identical perplexity.

Confirmation on the change rebased onto main `ad844354` (2026-10-08), one guarded run per model: gemma-3-4b-it-4bit 2885 tok/s prefill, 62.14 tok/s decode, MLX peak 4.52 GB, active after prefill 2.56 GB, 340 bypasses and an empty cache; Meta-Llama-3.1-8B-Instruct-4bit 1119 tok/s prefill, 36.66 tok/s decode, MLX peak 6.92 GB, active 4.75 GB, and the same 320 misses, 0 hits and 2 cached entries. Both agree with the after runs above. Gemma's active memory at the end of the run reads 2.56 GB there against 2.73 GB on `87538835`; main moved between the two bases and that difference was not isolated.
