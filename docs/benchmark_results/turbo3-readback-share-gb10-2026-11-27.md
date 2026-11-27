# Turbo3Asym KV host readbacks and the on-device codec (issue #3274), GB10, 2026-11-27

`--kv-cache-mode turbo3-asym` (fp16 K, 3-bit PolarQuant V) read back to the host in both halves of its codec. The quantizer evaluated the rotated coordinates and looked the centroids up on the CPU. The dequantizer evaluated the whole visible packed window, unpacked it on one CPU thread and uploaded the indices again, once per Turbo3 layer per decode step and once per layer per prefill chunk. Issue #3274 asks for the share of that cost first and for device ops second. This page records both: `main` before the change, and the branch that replaces the two host segments with `boundary_indices_on_device`, `pack_3bit_on_device` and `unpack_3bit_on_device`.

## Setup

- Host: NVIDIA GB10 (DGX Spark), 121 GB unified memory, 20 cores, driver 580.178.04, CUDA 13.0.88, Linux 7.0.0-1019-nvidia aarch64. One model process ran at a time, never beside a build.
- The host is shared. The `pre` round on `main` ran with no other GPU process. During the A/B block another job started GPU processes of its own from time to time. A sampler logged the GPU compute processes of other jobs and their CPU time every 3 s; a run started only after 15 s (45 s for the first runs) with no such work, and a run during which another job's GPU process started, stopped or used more CPU than 2 s or 2% of the run (3 s or 5% for the B=4 server runs, after a sleeping process of the other job turned out to idle at about 2% of a core) was set aside and repeated. Five runs taken before the sampler existed, in the minutes when that job became active, were set aside as well (one of them, fp16 at 32768 on `main`, read 11.3 tok/s against 24.5). Only runs that passed are in the tables and the CSVs.
- Builds: `cargo build --locked --release --features cuda`, rustc 1.97.1, MLX pin `fc344c98`, mlxcel 0.9.0-beta.2. `main` is `944dbfc2d`. The branch is the code of the commit that adds this page; its rows carry `944dbfc2-dirty` in `mlxcel_commit` because they were measured before that commit existed. The measured binaries were copied aside with their sha256 before the runs.
- Environment: product defaults (`MLX_ENABLE_TF32` and `MLXCEL_SDPA_DETERMINISTIC` unset).
- Checkpoint: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (revision `241a666d`, 4-bit, group size 64): 32 layers, 32 query heads, 8 KV heads, head_dim 128. It decodes through `KVCache::attend`. The boundary policy keeps the first two and the last two layers in fp16, so 28 layers hold Turbo3 V.

## Method

Three measurements, each run on both trees, and an nsys trace described with its table.

1. **Per-layer stage timers**, `examples/turbo3_readback_share_bench.rs`. The example uses only API present on both trees, so the same source was built on `main` (the tree of `944dbfc2d` plus that one file) and on the branch:

   ```bash
   cargo run --release --features cuda --example turbo3_readback_share_bench -- \
     --model models/Meta-Llama-3.1-8B-Instruct-4bit --contexts 4096,16384,32768 \
     --quant-tokens 1,2048,4096,16384 --csv out.csv
   ```

   Every timed region starts after `synchronize_default()` and ends after `eval` on its outputs and another `synchronize_default()`. A row is the median of 20 iterations after 5 untimed ones. `dequant_host_segment` is the readback of the packed window, the host unpack and the upload of the index tensor, reproduced from the public host helpers; `quant_host_segment` is the readback of the rotated coordinates, the host centroid lookup, the host pack and the upload. `dequant_full_call` and `quant_full_call` time `dequantize_v_turbo3` and `quantize_v_turbo3` as built, so they contain the host segment on `main` and not on the branch. `layer_call` is `dequantize_v_turbo3` and `layers::attention` for one decode token in one graph. 2048 tokens was added to the quantize sizes because it is the default prefill chunk.

2. **End-to-end decode**, B=1:

   ```bash
   for mode in fp16 turbo3-asym; do
     ./target/release/mlxcel-bench-decode -m models/Meta-Llama-3.1-8B-Instruct-4bit -p x \
       --prompt-tokens <256|1024|4096|16384|32768> -n 128 --ignore-eos --kv-cache-mode "$mode"
   done
   ```

   One process per run (load, a warm-up pass of the same prompt with 20 tokens, the measured pass). Three runs per cell. The prefill figure of the 32768-token run is the long prefill. The 256- and 1024-token contexts were added to see the cost of the one-token quantize where the window is short.

3. **Server**, for reply text and for B=4. `mlxcel-server -m <model> --parallel <1|4> --no-prompt-cache -c <ctx> --kv-cache-mode <mode>`, streaming `/v1/completions`, `temperature` 0, `ignore_eos`. The text of every reply is hashed. For the reply-text check the prompts are prose (slices of this repository's docs with a request for a summary at the end). For B=4 four requests of a 4096-token word-list prompt (eight short words chosen by index and a salt) are sent at once, and a batched step is looked for as the window in which all four rows are streaming.

The readback share of a decode step is `28 x dequant_host_segment / decode step`, with the step taken from the end-to-end run of the same tree, and is cross-checked by the end-to-end change between the trees.

## Before: `main` at `944dbfc2d`

### Readback share of the decode step

Decode step is 1000 / (median decode tok/s) of the `turbo3-asym` runs on `main`. The host segment is the example's `dequant_host_segment` on `main`, times the 28 Turbo3 layers.

| context | fp16 tok/s | turbo3-asym tok/s | turbo3 / fp16 | decode step | host segment x 28 | share of the step | readback / unpack / upload x 28 |
|---|---|---|---|---|---|---|---|
| 4096 | 44.09 | 4.35 | 0.099x | 229.9 ms | 175.3 ms | 76.2% | 151.9 / 51.8 / 21.3 ms |
| 16384 | 33.07 | 1.25 | 0.038x | 800.0 ms | 620.9 ms | 77.6% | 535.1 / 97.8 / 29.9 ms |
| 32768 | 24.49 | 0.57 | 0.023x | 1754.4 ms | 1433.3 ms | 81.7% | 1038.1 / 365.3 / 45.1 ms |

The three parts were timed separately and do not sum to the segment exactly. The readback is the largest: `array_to_raw_bytes` copies the packed window out through the bridge.

### Readback share of a long prefill

The default prefill chunk is 2048 tokens. A 32768-token prefill is 16 chunks; in each one every Turbo3 layer quantizes 2048 new tokens and dequantizes the whole window visible so far (2048, 4096, ... 32768 tokens). The quantize host segment is `quant_host_segment` at 2048 tokens. The dequantize host segment grows with the window; it is taken as linear in the visible length, scaled from the 32768-token row and summed over the 16 window lengths.

| | per call | calls in a 32768-token prefill | total | share of the prefill |
|---|---|---|---|---|
| quantize host segment (2048 new tokens) | 37.3 ms | 448 | 16.69 s | 38.6% |
| dequantize host segment (window 2048 to 32768) | 3.2 to 51.2 ms | 448 | 12.18 s | 28.2% |
| both | | | 28.87 s | 66.8% |

The measured prefill of the 32768-token run on `main` is 43.21 s (758.4 tok/s; fp16 2226.1 tok/s). The share is an estimate built from per-call times; the end-to-end prefill change below is the measured counterpart.

## After: the branch

### Decode and prefill, B=1

Median of the runs, with the range. The `runs` column is the number of runs that passed the disturbance check.

| context | mode | arm | runs | decode tok/s | range | ms per token | prefill tok/s | range | MLX peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 256 | fp16 | main | 3 | 48.24 | 48.19 to 48.36 | 20.7 | 3132.2 | 3109.9 to 3148.8 | 4.88 |
| 256 | fp16 | branch | 3 | 48.30 | 48.19 to 48.35 | 20.7 | 3137.7 | 3132.6 to 3150.3 | 4.88 |
| 256 | turbo3-asym | main | 3 | 15.87 | 15.81 to 16.08 | 63.0 | 918.7 | 913.6 to 934.0 | 4.73 |
| 256 | turbo3-asym | branch | 3 | 37.02 | 35.99 to 37.34 | 27.0 | 2761.9 | 2747.3 to 2785.3 | 4.76 |
| 1024 | fp16 | main | 3 | 47.05 | 46.95 to 47.12 | 21.3 | 3638.0 | 3628.0 to 3656.2 | 5.71 |
| 1024 | fp16 | branch | 3 | 47.04 | 46.92 to 47.08 | 21.3 | 3629.0 | 3623.1 to 3637.2 | 5.70 |
| 1024 | turbo3-asym | main | 3 | 9.45 | 9.36 to 9.51 | 105.8 | 1272.7 | 1270.0 to 1280.2 | 5.31 |
| 1024 | turbo3-asym | branch | 3 | 31.93 | 31.92 to 31.94 | 31.3 | 3269.6 | 3181.4 to 3269.7 | 5.39 |
| 4096 | fp16 | main | 3 | 44.09 | 43.66 to 45.19 | 22.7 | 3331.3 | 3291.0 to 3374.1 | 6.81 |
| 4096 | fp16 | branch | 3 | 43.82 | 42.80 to 45.36 | 22.8 | 3308.2 | 3125.6 to 3359.5 | 6.81 |
| 4096 | turbo3-asym | main | 3 | 4.35 | 4.33 to 4.36 | 229.9 | 1178.8 | 1173.5 to 1191.6 | 5.99 |
| 4096 | turbo3-asym | branch | 3 | 18.01 | 17.83 to 18.63 | 55.5 | 2987.6 | 2911.1 to 3035.8 | 6.07 |
| 16384 | fp16 | main | 3 | 33.07 | 32.86 to 33.99 | 30.2 | 2791.7 | 2749.4 to 2824.1 | 7.18 |
| 16384 | fp16 | branch | 3 | 32.98 | 32.86 to 33.93 | 30.3 | 2769.9 | 2764.9 to 2822.8 | 7.18 |
| 16384 | turbo3-asym | main | 3 | 1.25 | 1.24 to 1.25 | 800.0 | 964.5 | 957.5 to 965.6 | 6.42 |
| 16384 | turbo3-asym | branch | 3 | 4.75 | 4.72 to 4.76 | 210.5 | 2352.5 | 2350.1 to 2360.1 | 6.56 |
| 32768 | fp16 | main | 3 | 24.49 | 24.39 to 24.68 | 40.8 | 2226.1 | 2212.1 to 2249.5 | 9.35 |
| 32768 | fp16 | branch | 3 | 24.44 | 24.39 to 24.53 | 40.9 | 2214.8 | 2208.1 to 2217.8 | 9.34 |
| 32768 | turbo3-asym | main | 3 | 0.57 | 0.57 to 0.57 | 1754.4 | 758.4 | 754.1 to 767.1 | 7.92 |
| 32768 | turbo3-asym | branch | 3 | 2.48 | 2.48 to 2.49 | 403.2 | 1828.9 | 1827.7 to 1831.5 | 8.06 |

| context | turbo3 / fp16 on main | turbo3 / fp16 on the branch | turbo3 decode, branch / main | turbo3 prefill, branch / main | fp16 decode, branch / main |
|---|---|---|---|---|---|
| 256 | 0.329x | 0.766x | 2.333x | 3.006x | 1.001x |
| 1024 | 0.201x | 0.679x | 3.379x | 2.569x | 1.000x |
| 4096 | 0.099x | 0.411x | 4.140x | 2.534x | 0.994x |
| 16384 | 0.038x | 0.144x | 3.800x | 2.439x | 0.997x |
| 32768 | 0.023x | 0.101x | 4.351x | 2.412x | 0.998x |

### Readback share of the decode step, before and after

"Measured drop" is the end-to-end change of the decode step between the trees, as a share of the step on `main`. On the branch the codec has no host segment: `dequantize_v_turbo3` and `quantize_v_turbo3` contain no `eval`, no `array_to_raw_bytes` and no `from_bytes` (a source-guard test holds that).

| context | step on main | host segment x 28 on main | share (timers) | step on the branch | measured drop | host segment on the branch |
|---|---|---|---|---|---|---|
| 4096 | 229.9 ms | 175.3 ms | 76.2% | 55.5 ms | 75.8% | 0 |
| 16384 | 800.0 ms | 620.9 ms | 77.6% | 210.5 ms | 73.7% | 0 |
| 32768 | 1754.4 ms | 1433.3 ms | 81.7% | 403.2 ms | 77.0% | 0 |

### Per-layer stages

`turbo3_readback_share_bench`, microseconds: the median over the runs of each run's 20-iteration median, with the range of those per-run medians. The host-segment rows are the same reproduction on both trees, so they agree; the full-call rows are what changed.

| dequantize, by visible tokens | arm | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| `dequant_host_segment` | main | 6260 (5692 to 6464, n=4) | 22174 (21647 to 22798, n=4) | 51188 (50999 to 51819, n=4) |
| `dequant_host_segment` | branch | 6394 (5872 to 6579, n=3) | 22641 (21763 to 22782, n=3) | 51021 (50825 to 51035, n=3) |
| `dequant_host_readback` | main | 5423 (5002 to 5810, n=4) | 19112 (18499 to 19499, n=4) | 37075 (36772 to 37764, n=4) |
| `dequant_host_readback` | branch | 5543 (4849 to 5604, n=3) | 19198 (18500 to 19533, n=3) | 36815 (36743 to 37458, n=3) |
| `dequant_host_unpack` | main | 1851 (993 to 2649, n=4) | 3494 (3016 to 3819, n=4) | 13046 (12773 to 13436, n=4) |
| `dequant_host_unpack` | branch | 1624 (782 to 2687, n=3) | 3702 (3078 to 4013, n=3) | 12764 (12732 to 12826, n=3) |
| `dequant_host_upload` | main | 762 (251 to 1306, n=4) | 1068 (744 to 1482, n=4) | 1610 (1123 to 2488, n=4) |
| `dequant_host_upload` | branch | 985 (98 to 1114, n=3) | 1435 (785 to 1691, n=3) | 1718 (1413 to 2249, n=3) |
| `dequant_device_rest` | main | 1858 (1388 to 2338, n=4) | 5746 (5219 to 6103, n=4) | 10900 (10603 to 11144, n=4) |
| `dequant_device_rest` | branch | 2014 (1457 to 2141, n=3) | 5730 (5281 to 6166, n=3) | 10486 (9896 to 10487, n=3) |
| `dequant_full_call` | main | 7920 (6815 to 9254, n=4) | 27410 (26657 to 27755, n=4) | 60852 (60456 to 61065, n=4) |
| `dequant_full_call` | branch | 2380 (2035 to 2513, n=3) | 7080 (6414 to 7125, n=3) | 13624 (13064 to 14356, n=3) |
| `attention_fp16` | main | 482 (358 to 1096, n=4) | 1028 (386 to 1718, n=4) | 1100 (910 to 1317, n=4) |
| `attention_fp16` | branch | 183 (175 to 377, n=3) | 739 (565 to 1410, n=3) | 1352 (932 to 1980, n=3) |
| `layer_call` | main | 7366 (6858 to 8127, n=4) | 27634 (27020 to 28121, n=4) | 61744 (61258 to 62047, n=4) |
| `layer_call` | branch | 1689 (1384 to 1850, n=3) | 7327 (6962 to 7607, n=3) | 13847 (13720 to 14724, n=3) |

| quantize, by new tokens | arm | 1 | 2048 | 4096 | 16384 |
|---|---|---|---|---|---|
| `quant_host_segment` | main | 489 (345 to 767, n=4) | 37254 (36785 to 37506, n=3) | 70077 (65080 to 75348, n=4) | 298050 (294231 to 304158, n=4) |
| `quant_host_segment` | branch | 355 (58 to 395, n=3) | 37904 (37308 to 38390, n=3) | 75512 (75364 to 75527, n=3) | 299073 (296262 to 299260, n=3) |
| `quant_device_rest` | main | 695 (489 to 1192, n=4) | 769 (592 to 1678, n=3) | 1543 (1311 to 2179, n=4) | 4544 (4062 to 4647, n=4) |
| `quant_device_rest` | branch | 476 (177 to 505, n=3) | 1008 (654 to 1029, n=3) | 1031 (1021 to 2200, n=3) | 4543 (3906 to 5159, n=3) |
| `quant_full_call` | main | 433 (210 to 972, n=4) | 40089 (39982 to 40131, n=3) | 75535 (70355 to 81207, n=4) | 314835 (312779 to 315782, n=4) |
| `quant_full_call` | branch | 531 (162 to 724, n=3) | 2322 (2311 to 2916, n=3) | 4311 (4241 to 5316, n=3) | 19785 (19605 to 21611, n=3) |

### Reply text, main against the branch

Server at `--parallel 1`, two prompts per context for `turbo3-asym`, one for the fp16 control, 64 generated tokens, greedy. The prompts are slices of this repository's docs (six files of `944dbfc2d`, concatenated) followed by a one-line request for a summary, so the reply depends on the context; the token counts are what the server reported. `null` is a second server process of the `main` binary, at the two shorter contexts. A cell shows the first 12 hex digits of the sha256 of the reply text. Only the text is taken from these runs.

| mode | context | prompt | prompt tokens | main | main (null) | branch | identical |
|---|---|---|---|---|---|---|---|
| fp16 | 4096 | 1 | 3974 | 651787171288 | - | 651787171288 | yes |
| fp16 | 16384 | 1 | 16430 | 50f723229586 | - | 50f723229586 | yes |
| fp16 | 32768 | 1 | 33970 | aaaade247bcb | - | aaaade247bcb | yes |
| turbo3-asym | 4096 | 1 | 3974 | b03e7b2b92be | b03e7b2b92be | b03e7b2b92be | yes |
| turbo3-asym | 4096 | 2 | 4044 | 02daa1a88f2f | 02daa1a88f2f | 02daa1a88f2f | yes |
| turbo3-asym | 16384 | 1 | 16430 | 50f723229586 | 50f723229586 | 50f723229586 | yes |
| turbo3-asym | 16384 | 2 | 16372 | 2739c13261fe | 2739c13261fe | 2739c13261fe | yes |
| turbo3-asym | 32768 | 1 | 33970 | 44fb873d9ee0 | - | 44fb873d9ee0 | yes |
| turbo3-asym | 32768 | 2 | 31915 | f3bcbb12a3e5 | - | f3bcbb12a3e5 | yes |

### Four concurrent requests through the server

`--parallel 4`, four requests of a 4096-token word-list prompt sent at once. The batched step calls `KVCache::attend` once per row, so a row of a batch would pay its own codec. Whether a batch formed is read from the window in which all four rows stream: a negative window means the first row to finish was done before the last one started.

| mode | arm | run | tokens per row | first token of each row, s after the requests | all-rows window | tok/s in the window | driver wall time |
|---|---|---|---|---|---|---|---|
| fp16 | branch | b4-fix-fp16 | 256 | 1.4 / 6.3 / 11.1 / 16.2 | -4.2 s | none | 28 s |
| fp16 | main | b4-main-fp16 | 256 | 3.2 / 7.7 / 12.5 / 17.8 | -3.5 s | none | 29 s |
| turbo3-asym | branch | b4-fix-turbo3-r1 | 96 | 1.4 / 9.0 / 16.3 / 23.6 | -15.0 s | none | 32 s |
| turbo3-asym | branch | b4-fix-turbo3-r2 | 96 | 1.4 / 9.0 / 16.4 / 23.6 | -15.0 s | none | 32 s |
| turbo3-asym | branch | b4-fix-turbo3-r3 | 96 | 1.4 / 9.0 / 16.3 / 23.6 | -15.1 s | none | 33 s |
| turbo3-asym | main | b4-main-turbo3-r1 | 96 | 4.0 / 30.3 / 56.5 / 82.6 | -53.5 s | none | 109 s |
| turbo3-asym | main | b4-main-turbo3-r2 | 96 | 3.8 / 30.0 / 56.2 / 82.3 | -53.3 s | none | 108 s |
| turbo3-asym | main | b4-main-turbo3-r3 | 96 | 3.6 / 29.9 / 55.9 / 82.1 | -53.3 s | none | 108 s |

With `turbo3-asym` the four requests were served one after another on both trees: in every run the first tokens are a full request apart and the all-rows window is negative. So this server configuration never ran a batched `turbo3-asym` decode step, and there is no B=4 step to report; each request paid the B=1 cost above. The driver's wall time for the cell (server start, a 64-token warm-up and the four requests) was 108 s on `main` (108 to 109, n=3) and 32 s on the branch (32 to 33, n=3).

The fp16 control was served one after another as well, so the sequential service is not specific to the quantized mode here.

### CUDA runtime calls per decode token (nsys)

`nsys profile --trace=cuda` over a whole `mlxcel-bench-decode` run at a 4096-token context, once with 32 and once with 96 generated tokens (warm-up of 4 tokens). The two runs differ only in 64 decode steps, so the difference divided by 64 is the count per decode token.

| per decode token | main | branch |
|---|---|---|
| `cudaGraphAddKernelNode_v10000` | 1266.0 | 2078.2 |
| `cudaEventRecord_v3020` | 289.0 | 148.0 |
| `cudaGraphAddDependencies_v12030` | 92.0 | 147.0 |
| `cudaEventSynchronize_v3020` | 54.4 | 0.0 |
| `cudaLaunchHostFunc_v10000` | 176.0 | 147.0 |
| `cudaStreamWaitEvent_v3020` | 176.0 | 147.0 |
| `cudaGraphAddEmptyNode_v10000` | 0.0 | 28.0 |
| `cudaEventQuery_v3020` | 120.0 | 93.0 |
| `cudaFreeAsync_v11020` | 1.0 | 4.0 |
| `cudaMallocAsync_v11020` | 1.0 | 4.0 |
| `cudaMemPoolGetAttribute_v11020` | 1.1 | 3.8 |
| `cudaGraphExecUpdate_v10020` | 148.0 | 146.9 |
| `cudaGraphCreate_v10000` | 148.0 | 147.0 |
| `cudaGraphDestroy_v10000` | 148.0 | 147.0 |
| `cudaGraphLaunch_v10000` | 148.0 | 147.0 |

Only calls whose count differs by at least 0.5 per token are listed, plus the graph launches. The forced synchronizations go from 54.4 per token to 0.0: about two per Turbo3 layer on `main`, one for each `eval` in the codec. The number of graph launches does not move (148 and 147): the graph byte budget already cuts the step's graph that often at this context, so removing the forced evaluations did not reduce launches. The step holds more kernel nodes (1266 to 2078), which are the index, pack and unpack ops. `cudaMemcpy` copies per decode token: 0.0 on `main`, 0.0 on the branch. On GB10's unified memory the host reads and writes the mapped buffers directly, so the readback was never a `cudaMemcpy`; it shows as host time in the stage timers and as the synchronizations here.

## Reading

- **Before.** The host segment of the dequantizer was 76% / 78% / 82% of a decode step at 4096 / 16384 / 32768 tokens, and `turbo3-asym` decoded at 0.099x / 0.038x / 0.023x of fp16. By the per-call estimate the two host segments were about 67% of a 32768-token prefill.
- **After.** `turbo3-asym` decode is 4.1x / 3.8x / 4.4x of `main` at 4096 / 16384 / 32768 tokens and prefill 2.5x / 2.4x / 2.4x. The end-to-end drop of the decode step (76% / 74% / 77%) matches the share the timers gave. fp16 is unchanged (0.994x / 0.997x / 0.998x), as it does not touch the codec.
- **What is left.** The dequantize is still O(visible) on the device (unpack, centroid gather, norm correction, inverse rotation, rescale). Timed alone, `dequant_full_call` x 28 is 67 / 198 / 381 ms on the branch, the same order as the whole decode step (56 / 211 / 403 ms; a call timed alone pays an evaluation of its own, which is why it can read above the step at the shortest context), against 5 / 21 / 38 ms for the fp16 attention itself. On the branch `turbo3-asym` decodes at 0.41x / 0.14x / 0.10x of fp16. A dequant-free Turbo3 attention route is what the issue leaves to a follow-up; these rows are the input for that decision.
- **One-token quantize.** At decode the quantizer sees one new token. In isolation that call is 433 us on `main` (210 to 972 over 4 runs) and 531 us on the branch (162 to 724 over 3): the device form is a few dozen small graph nodes where the host form was one forced evaluation and a 1024-float loop, and the per-run ranges overlap. End to end at short contexts, where this call weighs most, the branch decodes at 2.33x / 3.38x of `main` at 256 / 1024 tokens.
- **Memory.** MLX peak of the `turbo3-asym` runs: 5.99 to 6.07 GB at 4096, 6.42 to 6.56 GB at 16384, 7.92 to 8.06 GB at 32768. The index stage holds `tokens x head_dim x 7` booleans per quantize call until the step is evaluated.
- **Text.** 9 B=1 reply pairs were compared between `main` and the branch; all are identical.

## Not measured

- Apple Silicon. No Metal build was made for this change, so there is no Metal record and the device forms have not run on Metal.
- ROCm.
- Any checkpoint other than Llama 3.1 8B 4-bit end to end. Head dims 64 and 256 are covered by the unit tests only.
- A batched decode step. Four concurrent requests were served one after another in the server configuration tried, for fp16 as well, so no step with more than one row was timed. Why that server did not batch them was not investigated.
- `mlxcel generate`; the end-to-end rows are `mlxcel-bench-decode` (the same engine loop) and the server.

## Data

- [`bench_decode.csv`](data/turbo3-readback-share-gb10-2026-11-27/bench_decode.csv): one row per `mlxcel-bench-decode` run.
- [`example.csv`](data/turbo3-readback-share-gb10-2026-11-27/example.csv): the example's rows for every run of both trees.
- [`server.csv`](data/turbo3-readback-share-gb10-2026-11-27/server.csv): one row per server request, with the sha256 of its reply text.
