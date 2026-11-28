# Turbo4 KV quantize and decode attention on GB10

The baseline is mlxcel `944dbfc2d`, with MLX pinned to
`fc344c986c4169d99f76b5d95ae6b4a58a359a63`. The checkpoint is
Meta-Llama-3.1-8B-Instruct-4bit, revision
`241a666dad6cb93c8ff213d39a7f34a36bf26db4`, affine 4-bit. It has 32 query heads, eight KV
heads, head dimension 128 and 32 layers. The default boundary policy keeps two
layers on each end in fp16, leaving 28 Turbo layers. This record covers one
NVIDIA GB10 with 121 GiB unified memory, Ubuntu 24.04.5, driver 580.178.04 and
CUDA toolkit 13.0.3. Apple Silicon was unavailable.

## Method

The op benchmark uses B=1, a single query token, four query heads per KV head,
fixed-seed inputs and one resident input set (warm buffers). Each row contains
20 new graphs after five warmups. The timer starts after synchronizing the
default stream and stops after evaluating the result arrays and synchronizing
again. The dequant row includes packed V and, for symmetric Turbo4, packed K.
For delegated it includes cold V only. Its full-call row also rotates a
128-token hot tail, concatenates it with cold V, runs SDPA and inverse-rotates
the result. The fp16 control times native SDPA.

The decode executable uses a synthetic prompt of exactly 4096, 16384 or 32768
tokens, greedy sampling and 128 generated tokens with EOS suppressed. The
first generated token falls inside prefill, leaving 127 timed decode steps.
Each process loads the checkpoint, warms up, and then measures a fresh
sequence. The MLX buffer cache limit is 4 GiB. An external process-group
watchdog enforces a 58 GiB RSS ceiling and 25 GiB available-memory floor. GPU
compute processes and builds were checked before each launch.

```sh
cargo run --locked --release --features cuda --example turbo_kv_dequant_share_bench -- \
  --model "$MODEL" --contexts 4096,16384,32768 --csv layer.csv
for context in 4096 16384 32768; do
  for mode in fp16 turbo4-asym turbo4 turbo4-delegated; do
    mlxcel-bench-decode -m "$MODEL" -p x --prompt-tokens "$context" \
      -n 128 --ignore-eos --kv-cache-mode "$mode"
  done
done
```

The stock model-family resolver substitutes Turbo4Asym for a symmetric
`turbo4` request on Llama. The requested and effective modes are both
recorded in the decode rows. For the true symmetric engine diagnostic, the
baseline and fix each used a scratch copy of `src/bin/bench_decode.rs` that
passed `KVCacheMode::Turbo4` directly to DirectEngine at the CLI resolver call
site. Tokenization, cache construction and attention dispatch used that
revision's library. The stock Qwen3.5 family accepts symmetric Turbo4, but its
full-attention layers reuse `Qwen3NextAttention`, which calls
`update_and_fetch`. That family does not call the three dense
`KVCache::attend` paths covered by #3272.

[End-to-end decode rows](data/turbo-kv-dequant-share-gb10-2026-11-28/decode.csv)

## Binary provenance

The baseline used the stock executables from the same clean main tree as the
op harness. The fix used the branch based on `15c341671` (#3497). This table
identifies the binaries independently of the worktree's Git metadata.

| Executable | Baseline SHA-256 | Fix SHA-256 |
|---|---|---|
| `mlxcel` | `04ccd29843806d990af18af2b559f92adca4dd9ea1d21c12fa6cd314dc2f198d` | `20b2d8ad188b7bed6624123561de63bc947ff30990e40c490878ccd7954f2d61` |
| `mlxcel-server` | `8123fb94bcfde8abc3902c788fad8d133bd38a37618ded8a01515b2bd1b850d6` | `50c2f35e99d286264fd03a2faa1d342e77abce785e8276140060f074c72245b2` |
| `mlxcel-bench-decode` | `3369d712ee50f9def230f6379466db1d788f265b83f182fffe079617425212a2` | `4dd8d5b83056ece989b42785302dcfd240e8fe086dcc7046b12521d9258b5fac` |

## Gate and readback instrumentation

The baseline op harness linked a scratch CUDA prototype to the unchanged main library. It compared 8, 16 and 32 warps per query head in the same process. The prototype includes the symmetric Q rotation, delegated hot-tail rotation and output inverse rotation in its full-call comparison. Every run compared its output to dequant-first SDPA before timing. The 81 comparisons cover three modes, three context lengths, three warp counts and three processes. Maximum RMS was 0.000004145. The decision uses the 32-warp variant, which the production CUDA port implements. [Prototype comparisons](data/turbo-kv-dequant-share-gb10-2026-11-28/prototype-parity.csv)

The main and fix readback probes wrapped the C++ bridge symbols for `eval` and `array_to_raw_bytes` with the GNU linker's `--wrap` option. Thread-local counters and steady-clock timers covered the call to `quantize_v_turbo4`. Each probe stopped those counters before evaluating its three outputs, then recorded the total after stream synchronization. The probe used B=1, Hkv=8, D=128 and T=1, 4096 and 16384. The separate uninstrumented quantizer and the copy-only control run in the checked-in example.

```sh
cargo rustc --release --locked --features cuda --example turbo_probe -- \
  -C "link-arg=$R/scripts/prototype.o"
cargo rustc --release --locked --features cuda --example turbo_readback_probe -- \
  -C "link-arg=$R/scripts/readback-wrap.o" \
  -C link-arg=-Wl,--wrap=_ZN7mlx_cxx4evalERKNS_8MlxArrayE \
  -C link-arg=-Wl,--wrap=_ZN7mlx_cxx18array_to_raw_bytesERKNS_8MlxArrayE
```

The two extra example names above identify the scratch instrumentation. The production benchmark in this revision is `turbo_kv_dequant_share_bench`.

## Numerical parity

The scalar host reference and the device codec produced identical packed K/V bytes and FP16 norm bits over the fixed-seed D=64/128/256, T=1/4096 suite. Twelve of 24,582 FP16 rescale values rounded differently. The maximum relative error was 0.0008976661 (limit 0.001). The boundary tests cover all 15 ties and the adjacent representable floats, plus zeros, NaN and infinities. Empty axes preserve the stored shapes.

The two-cache test used B=2, four query heads and two KV heads. It called `KVCache::attend` for prefill and 64 decode steps. A neutral additive mask selected dequant-first SDPA on the reference cache. The packed route executed exactly 64 times in each case. Maximum RMS across steps was 0.0002128040 for asymmetric D=64, 0.0002252933 for symmetric D=128 and 0.0002054454 for delegated D=256 (limit 0.005).

## GB10 decision

|Effective mode|Gate A, layer at 16K|Gate A, step at 16K|Gate B, latency saved at 16K|At 32K|
|---|---:|---:|---:|---:|
|turbo4-asym|86.74%|81.52%|53.68%|44.75%|
|turbo4|110.43%|99.83%|73.34%|76.82%|
|turbo4-delegated|98.01%|90.98%|35.88%|49.12%|

Gate A requires at least 15% dequant/full and 5% of the decode step at 16K. The step estimate uses 28 Turbo layers and the median decode rate. Gate B is the reduction in latency of the 32-warp prototype relative to the full reference call, including the basis changes. Gate B requires at least 10% savings at both lengths. All three CUDA modes cleared both gates. The same packed kernel reads dense K in asymmetric and delegated mode and packed K in symmetric mode.

## Layer and engine results

|Effective mode|Tokens|Main decode tok/s [min, max]|Fix decode tok/s [min, max]|Reference dequant/full, main|Reference dequant/full, fix|Main dequant / decode step|Fix packed call, µs|
|---|---:|---:|---:|---:|---:|---:|---:|
|fp16|4096|42.92 [42.84, 43.15]|42.93 [42.89, 42.95]|—|—|—|—|
|fp16|16384|32.37 [32.35, 32.45]|32.34 [32.32, 32.38]|—|—|—|—|
|fp16|32768|24.28 [24.24, 24.44]|24.21 [24.17, 24.29]|—|—|—|—|
|turbo4-asym|4096|17.22 [17.03, 19.30]|34.13 [34.11, 34.18]|109.65% [77.00, 143.15]|120.08% [74.14, 125.55]|80.81%|1042.4 [748.3, 1056.4]|
|turbo4-asym|16384|8.89 [8.87, 8.91]|20.22 [20.15, 20.23]|86.74% [78.08, 113.30]|88.40% [87.74, 96.33]|81.52%|2182.4 [1111.7, 2234.0]|
|turbo4-asym|32768|5.55 [5.54, 5.58]|12.86 [12.86, 12.88]|96.76% [90.81, 102.89]|95.29% [83.29, 104.33]|89.12%|2773.2 [2530.5, 2884.4]|
|turbo4|4096|11.03 [10.91, 11.11]|29.98 [29.87, 30.01]|138.54% [102.20, 146.26]|123.71% [110.25, 171.25]|86.91%|866.6 [743.7, 889.9]|
|turbo4|16384|4.57 [4.57, 4.58]|16.52 [16.51, 16.60]|110.43% [109.91, 111.73]|117.55% [104.99, 120.59]|99.83%|1302.7 [1261.9, 1753.1]|
|turbo4|32768|2.62 [2.62, 2.62]|10.50 [10.46, 10.51]|101.96% [100.80, 105.25]|103.47% [100.91, 104.08]|96.53%|3336.2 [2635.3, 3629.2]|
|turbo4-delegated|4096|27.96 [27.96, 28.01]|34.44 [34.41, 34.47]|136.78% [61.89, 163.24]|63.58% [32.26, 75.29]|114.08%|1040.8 [852.6, 1766.1]|
|turbo4-delegated|16384|10.40 [10.34, 10.42]|19.15 [19.11, 19.16]|98.01% [85.40, 98.01]|82.23% [73.21, 88.59]|90.98%|1844.7 [973.9, 1905.1]|
|turbo4-delegated|32768|5.74 [5.67, 5.76]|11.92 [11.91, 11.94]|86.92% [76.28, 90.74]|82.85% [78.57, 85.11]|88.43%|2622.7 [2056.3, 3697.0]|

Symmetric engine rows use the direct-mode diagnostic. The stock Llama binary printed a fallback warning. Its requested and effective modes are identified separately in the CSV. Reference dequant/full is the ratio for the independently evaluated reference operations on each revision. The estimated share of a main decode step multiplies isolated dequant time by 28 and divides by measured time per token. Both estimates can exceed 100% because separate measurements impose materialization and synchronization. On a taken packed route the time spent in a standalone cold-window dequant stage is zero. Inline unpack and scale are included in the packed call. The fix packed-call column includes the basis changes. Decode and packed-call intervals compare the three process results. For dequant/full, the central value is the ratio of the stage medians and the interval spans the three within-run ratios.

## Host readbacks during quantization

|Arm|Tokens|Calls to eval / raw per invocation|Full codec, µs|Eval + raw, µs|Eval + raw / full|Raw copy alone / full|
|---|---:|---:|---:|---:|---:|---:|
|main|1|2 / 2|813.4|181.6|31.43%|3.66%|
|main|4096|2 / 2|73685.8|50231.1|68.08%|66.20%|
|main|16384|2 / 2|328174.9|213068.5|64.95%|63.68%|
|fix|1|0 / 0|851.0|0.0|0.00%|0.00%|
|fix|4096|0 / 0|5419.5|0.0|0.00%|0.00%|
|fix|16384|0 / 0|18065.8|0.0|0.00%|0.00%|

The linker-wrapped FFI entry points count calls only inside the codec, before evaluating its returned outputs. Full latency includes evaluating all three outputs and synchronizing the stream. Each ratio is calculated for each invocation and then summarized by the median. The separate readback-control row in the op CSV copies already-evaluated arrays. It is a positive control for transfer cost.

[Layer measurements](data/turbo-kv-dequant-share-gb10-2026-11-28/layer.csv) · [Readback measurements](data/turbo-kv-dequant-share-gb10-2026-11-28/readback.csv)

## Uninstrumented quantization

The table gives the median and range of the three process medians. Each process timed 20 calls. The wrapped latencies above include instrumentation.

| Tokens | Main quantize, µs [min, max] | Fix quantize, µs [min, max] |
| ---: | ---: | ---: |
| 1 | 624.2 [385.8, 714.0] | 767.0 [617.1, 1,279.0] |
| 4096 | 73,317.5 [65,637.2, 73,610.8] | 6,421.5 [6,173.8, 6,608.4] |
| 16384 | 327,621.8 [308,883.3, 327,652.4] | 18,148.8 [17,698.9, 19,083.5] |

## Traced decode routes

Nsight Systems counted CUDA launches whose demangled names contain `mlxcel_turbo_packed_attention`. Each switch setting ran in a fresh process and generated eight tokens. The server trace records a loopback `/completion` request. Llama uses Meta-Llama-3.1-8B-Instruct-4bit. The additional Qwen3.5 trace uses Qwen3.5-4B-4bit. Each trace is a single profiled process. The throughput comparison uses the separate three-run 128-token benchmark. [Launch counts and CUDA activity totals](data/turbo-kv-dequant-share-gb10-2026-11-28/route.csv)

| Checkpoint | Entry point | Requested | Effective | Packed launches on | Packed launches off |
| --- | --- | --- | --- | ---: | ---: |
| llama | direct | turbo4 | turbo4 | 196 | 0 |
| llama | generate | turbo4 | turbo4-asym | 196 | 0 |
| llama | generate | turbo4-asym | turbo4-asym | 196 | 0 |
| llama | generate | turbo4-delegated | turbo4-delegated | 196 | 0 |
| llama | server | turbo4 | turbo4-asym | 196 | 0 |
| llama | server | turbo4-asym | turbo4-asym | 196 | 0 |
| llama | server | turbo4-delegated | turbo4-delegated | 196 | 0 |
| qwen35 | generate | turbo4 | turbo4 | 0 | 0 |
| qwen35 | server | turbo4 | turbo4 | 0 | 0 |

The CLI traces used `nsys profile --trace=cuda --cuda-graph-trace=node --cpuctxsw=none --sample=none` around `mlxcel generate -m "$MODEL" -p "$PROMPT" --no-chat-template --profile --temp 0 -n 8 --kv-cache-mode "$MODE"`. Server traces launched `mlxcel-server -m "$MODEL" --host 127.0.0.1 --port "$PORT" --no-prompt-cache --parallel 1 --kv-cache-mode "$MODE"` through `nsys launch`. Capture started after health readiness and stopped after the completion response. Both used fresh processes with `MLXCEL_TURBO_PACKED_ATTENTION` set to 1 and 0.

With `MLXCEL_TURBO_PACKED_ATTENTION=0`, the fix stock asymmetric decode benchmark at 16,384 prompt tokens produced 10.99 tok/s (one 128-token run).

## Generated output

Two prompts ask for a summary and for two details from a passage. The same prompt bytes or model token IDs were used on each arm at exactly 4,096, 16,384 and 32,768 input tokens. Sampling used temperature zero, EOS suppression and 64 output tokens. A pair counts as exact when both its decoded UTF-8 and token IDs match. The fix-off rows repeat comparisons with the packed switch disabled. For stock modes that differed, all six pairs were repeated. The direct diagnostic repeats each differing pair. The issue defines its numerical tolerance on layer outputs. It specifies no token-sequence tolerance. [Prompt, completion and token hashes with first-divergence positions](data/turbo-kv-dequant-share-gb10-2026-11-28/generated.csv)

| Entry point | Requested mode | Comparison | 4K exact / pairs | 16K exact / pairs | 32K exact / pairs |
| --- | --- | --- | ---: | ---: | ---: |
| direct | turbo4 | fix | 1/2 | 1/2 | 0/2 |
| direct | turbo4 | fix-off | 0/1 | 0/1 | 0/2 |
| server | fp16 | fix | 2/2 | 2/2 | 2/2 |
| server | turbo4 | fix | 1/2 | 1/2 | 0/2 |
| server | turbo4 | fix-off | 2/2 | 1/2 | 0/2 |
| server | turbo4-asym | fix | 1/2 | 1/2 | 0/2 |
| server | turbo4-asym | fix-off | 2/2 | 1/2 | 0/2 |
| server | turbo4-delegated | fix | 2/2 | 2/2 | 2/2 |

For each of main, fix and fix with the switch off, the two stock flags that resolved to asymmetric Turbo4 produced identical text and IDs on all six prompts. Across revisions, disabling the packed kernel restored one asymmetric comparison. Three other asymmetric comparisons still differed. All four differing direct symmetric pairs also differed with the switch off.
